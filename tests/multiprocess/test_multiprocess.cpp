// Atomic multiprocess proof: one coordinator OS process + two worker OS
// processes + two replicas over the real framed TCP transport. Exercises
// provision/warm/promote/execute, kills the authoritative worker, fails over to
// the standby, restarts the worker with a new boot id, replays stale authority,
// and proves no double-serving and a coherent final state.

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include <replicafabric/core/identity_kinds.hpp>
#include <replicafabric/core/enums.hpp>
#include <replicafabric/distributed/transport.hpp>
#include <replicafabric/distributed/protocol.hpp>
#include <replicafabric/persistence/snapshot.hpp>

#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace replicafabric;

static int g_failures = 0;
static void check(bool cond, const std::string& msg) {
  if (!cond) { std::cerr << "PROOF FAIL: " << msg << "\n"; ++g_failures; }
  else { std::cout << "PROOF OK:   " << msg << "\n"; }
}

std::mt19937_64 g_rng(0xDEADBEEFULL);
template <typename T> T rnd() { return T::random(g_rng); }

std::uint16_t free_port() {
  SOCKET s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_ANY); addr.sin_port = 0;
  ::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  int len = sizeof(addr);
  ::getsockname(s, reinterpret_cast<sockaddr*>(&addr), &len);
  const std::uint16_t port = ntohs(addr.sin_port);
  ::closesocket(s);
  return port;
}

struct Proc {
  HANDLE h = INVALID_HANDLE_VALUE;
  bool spawn(const std::string& exe, const std::string& args) {
    std::string cmd = "\"" + exe + "\" " + args;
    std::vector<char> buf(cmd.begin(), cmd.end());
    buf.push_back('\0');
    STARTUPINFOA si{}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    BOOL ok = ::CreateProcessA(exe.c_str(), buf.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi);
    if (!ok) return false;
    h = pi.hProcess;
    ::CloseHandle(pi.hThread);
    return true;
  }
  void kill() { if (h != INVALID_HANDLE_VALUE) { ::TerminateProcess(h, 1); ::CloseHandle(h); h = INVALID_HANDLE_VALUE; } }
};

bool send_msg(TcpConnection& c, const Message& m) { return c.write_frame(encode_message(m)); }
bool recv_msg(TcpConnection& c, Message& out) { std::vector<std::uint8_t> b; if (!c.read_frame(b)) return false; return decode_message(b, out); }

// A framed connection the test can half-close: finish() returns once the
// coordinator has closed its side, i.e. after everything it does on losing the
// connection has happened.
struct RawConn {
  SOCKET s = INVALID_SOCKET;
  bool open(std::uint16_t port) {
    s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return false;
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); addr.sin_port = htons(port);
    return ::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
  }
  bool send_all(const std::uint8_t* p, std::size_t n) {
    std::size_t done = 0;
    while (done < n) { const int r = ::send(s, reinterpret_cast<const char*>(p + done), static_cast<int>(n - done), 0); if (r <= 0) return false; done += static_cast<std::size_t>(r); }
    return true;
  }
  bool recv_all(std::uint8_t* p, std::size_t n) {
    std::size_t done = 0;
    while (done < n) { const int r = ::recv(s, reinterpret_cast<char*>(p + done), static_cast<int>(n - done), 0); if (r <= 0) return false; done += static_cast<std::size_t>(r); }
    return true;
  }
  bool send(const Message& m) {
    const std::vector<std::uint8_t> b = encode_message(m);
    const std::uint32_t len = static_cast<std::uint32_t>(b.size());
    const std::uint8_t hdr[4] = {static_cast<std::uint8_t>(len >> 24), static_cast<std::uint8_t>(len >> 16), static_cast<std::uint8_t>(len >> 8), static_cast<std::uint8_t>(len)};
    return send_all(hdr, 4) && send_all(b.data(), b.size());
  }
  bool recv(Message& out) {
    std::uint8_t hdr[4];
    if (!recv_all(hdr, 4)) return false;
    const std::uint32_t len = (static_cast<std::uint32_t>(hdr[0]) << 24) | (static_cast<std::uint32_t>(hdr[1]) << 16) | (static_cast<std::uint32_t>(hdr[2]) << 8) | static_cast<std::uint32_t>(hdr[3]);
    std::vector<std::uint8_t> b(len);
    if (len > 0 && !recv_all(b.data(), len)) return false;
    return decode_message(b, out);
  }
  bool finish() {
    if (::shutdown(s, SD_SEND) != 0) return false;
    char c = 0;
    for (;;) { const int r = ::recv(s, &c, 1, 0); if (r == 0) return true; if (r < 0) return false; }
  }
  void close() { if (s != INVALID_SOCKET) { ::closesocket(s); s = INVALID_SOCKET; } }
};

int main(int argc, char** argv) {
  if (!tcp_init()) { std::cerr << "tcp_init failed\n"; return 1; }
  if (argc < 3) { std::cerr << "usage: rf_multiprocess <coordinator-exe> <worker-exe>\n"; return 1; }
  const std::string coord_exe = argv[1];
  const std::string worker_exe = argv[2];

  const std::uint16_t port = free_port();
  const std::string portstr = std::to_string(port);

  Proc coord; Proc wa; Proc wb;
  check(coord.spawn(coord_exe, "--port " + portstr), "spawn coordinator");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  // Fixed well-known ids.
  ReplicaSetId setid = rnd<ReplicaSetId>();
  ModelId model = rnd<ModelId>();
  ArtifactId artifact = rnd<ArtifactId>();
  ArtifactGeneration agen = rnd<ArtifactGeneration>();
  ReplicaSetGeneration sgen = rnd<ReplicaSetGeneration>();
  PolicyGeneration pgen = rnd<PolicyGeneration>();
  WorkerId wa_id = rnd<WorkerId>(); WorkerBootId wa_boot = rnd<WorkerBootId>(); NodeId wa_node = rnd<NodeId>();
  WorkerId wb_id = rnd<WorkerId>(); WorkerBootId wb_boot = rnd<WorkerBootId>(); NodeId wb_node = rnd<NodeId>();
  ReplicaId ra = rnd<ReplicaId>(); ReplicaId rb = rnd<ReplicaId>();

  check(wa.spawn(worker_exe, " --port " + portstr + " --worker " + wa_id.str() + " --boot " + wa_boot.str() + " --node " + wa_node.str()), "spawn worker A");
  check(wb.spawn(worker_exe, " --port " + portstr + " --worker " + wb_id.str() + " --boot " + wb_boot.str() + " --node " + wb_node.str()), "spawn worker B");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  TcpConnection control;
  check(control.connect_to("127.0.0.1", port), "control connects to coordinator");

  Message m; m.epoch = CoordinatorEpoch(); m.kind = MessageKind::CREATE_SET; m.seq = 1; m.set = setid;
  m.payload = "model=" + model.str() + "\nartifact=" + artifact.str() + "\nagen=" + agen.str() +
              "\nsgen=" + sgen.str() + "\npgen=" + pgen.str() + "\nworkload=" + rnd<WorkloadId>().str() +
              "\ntenant=" + rnd<TenantId>().str() + "\ndesired=2\nmin=1\nmax=2\nbackend=2\nccmaj=8\nccmin=0";
  send_msg(control, m); Message ack;
  check(recv_msg(control, ack) && ack.kind == MessageKind::CREATE_SET_ACK && ack.payload.rfind("ok",0)==0, "create replica set");

  auto provision = [&](const ReplicaId& rid, const std::string& wstr) {
    Message pm; pm.kind = MessageKind::PROVISION; pm.epoch = CoordinatorEpoch(); pm.set = setid; pm.replica = rid; pm.seq = 1;
    pm.payload = "worker=" + wstr + "\nmodel=" + model.str() + "\nagen=" + agen.str() + "\nccmaj=c\ncapacity=0x1000000\npolicyfp=fp-1";
    send_msg(control, pm); Message pa;
    return recv_msg(control, pa) && pa.kind == MessageKind::PROVISION_ACK && pa.payload.rfind("ok",0)==0;
  };
  check(provision(ra, wa_id.str()), "provision replica A");
  check(provision(rb, wb_id.str()), "provision replica B");

  auto wait_state = [&](const ReplicaId& rid, const std::string& want, int attempts) {
    for (int i = 0; i < attempts; ++i) {
      Message q; q.kind = MessageKind::QUERY; q.epoch = CoordinatorEpoch(); q.replica = rid; q.payload = "q=state";
      send_msg(control, q); Message qa;
      if (recv_msg(control, qa)) {
        if (qa.payload.find(want) != std::string::npos) return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    return false;
  };
  // Wait for both replicas to become READY (healed after warm + health + readiness).
  check(wait_state(ra, "lifecycle=READY", 100), "replica A reaches READY");
  check(wait_state(rb, "lifecycle=READY", 100), "replica B reaches READY");

  auto promote = [&](const ReplicaId& rid, const std::string& target) {
    Message pm; pm.kind = MessageKind::PROMOTE; pm.epoch = CoordinatorEpoch(); pm.replica = rid; pm.payload = "target=" + target;
    send_msg(control, pm); Message a; if (recv_msg(control, a)) { std::cerr << "  PROMOTE " << rid.str() << " target=" << target << " -> " << a.payload << "\n"; return a.payload.rfind("ok",0)==0; } return false;
  };
  check(promote(ra, "5"), "promote replica A to PRIMARY");
  check(promote(rb, "1"), "promote replica B to STANDBY");

  // Execute real work on the serving primary A.
  Message ex; ex.kind = MessageKind::EXECUTE; ex.epoch = CoordinatorEpoch(); ex.replica = ra; ex.attempt = rnd<AttemptId>(); ex.payload = "req=reqA";
  send_msg(control, ex); Message exa; check(recv_msg(control, exa) && exa.payload.rfind("ok",0)==0, "submit work to A");
  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  { Message q; q.kind = MessageKind::QUERY; q.epoch = CoordinatorEpoch(); q.replica = ra; q.payload = "q=result";
    send_msg(control, q); Message qa; check(recv_msg(control, qa) && qa.payload.find("outcome=1") != std::string::npos, "work on A completed successfully"); }

  // Work is forwarded only to a replica that holds serving authority. The
  // standby's worker is connected, but B may not serve.
  { Message xb; xb.kind = MessageKind::EXECUTE; xb.epoch = CoordinatorEpoch(); xb.replica = rb; xb.attempt = rnd<AttemptId>(); xb.payload = "req=reqB";
    send_msg(control, xb); Message xa; check(recv_msg(control, xa) && xa.payload.rfind("err",0)==0, "work for standby B is refused"); }

  // Kill worker A while it is authoritative; fail over to B.
  wa.kill();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  { Message f; f.kind = MessageKind::FAILOVER; f.epoch = CoordinatorEpoch(); f.set = setid; f.payload = "failed=" + ra.str() + "\nreason=worker A died";
    send_msg(control, f); Message fa; check(recv_msg(control, fa) && fa.payload.rfind("ok",0)==0, "failover to B after killing A"); }
  check(wait_state(rb, "serving=1", 100), "replica B is now serving");

  // Restart worker A with a fresh boot id; the old authority must not return.
  WorkerBootId wa_boot2 = rnd<WorkerBootId>();
  Proc wa2; check(wa2.spawn(worker_exe, " --port " + portstr + " --worker " + wa_id.str() + " --boot " + wa_boot2.str() + " --node " + wa_node.str()), "restart worker A with new boot");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  // Exactly one replica must be serving (no double-serving).
  { Message q; q.kind = MessageKind::QUERY; q.epoch = CoordinatorEpoch(); q.replica = ra; q.payload = "q=state"; send_msg(control, q); Message qa;
    if (recv_msg(control, qa)) { check(qa.payload.find("serving=0") != std::string::npos, "restarted replica A does not serve (serving=0)"); } else check(false, "query A state"); }
  { Message q; q.kind = MessageKind::QUERY; q.epoch = CoordinatorEpoch(); q.replica = rb; q.payload = "q=state"; send_msg(control, q); Message qa;
    if (recv_msg(control, qa)) { check(qa.payload.find("serving=1") != std::string::npos, "replica B still serves (serving=1)"); } else check(false, "query B state"); }

  // Snapshot the coordinator state and validate it decodes + is coherent.
  { Message s; s.kind = MessageKind::SNAPSHOT; s.epoch = CoordinatorEpoch(); send_msg(control, s); Message sa;
    if (recv_msg(control, sa) && sa.kind == MessageKind::SNAPSHOT) {
      std::string hex = get_payload(sa.payload, "hex", "");
      std::vector<std::uint8_t> bytes;
      for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        auto hx = [](char c)->std::uint8_t { if (c >= '0' && c <= '9') return static_cast<std::uint8_t>(c - '0'); if (c >= 'a' && c <= 'f') return static_cast<std::uint8_t>(c - 'a' + 10); if (c >= 'A' && c <= 'F') return static_cast<std::uint8_t>(c - 'A' + 10); return 0; };
        bytes.push_back(static_cast<std::uint8_t>((hx(hex[i]) << 4) | hx(hex[i+1])));
      }
      std::string err;
      auto rec = decode_snapshot(bytes, &err);
      check(rec.has_value(), "snapshot decodes (" + err + ")");
      if (rec) {
        std::uint32_t serving_cnt = 0;
        for (const auto& [rid, rr] : rec->replicas) { if (rr.serving_eligible) ++serving_cnt; (void)rid; }
        check(serving_cnt == 1, "exactly one serving replica in final state (" + std::to_string(serving_cnt) + ")");
      }
    } else { check(false, "snapshot received"); } }

  // A worker that registers again on a new connection keeps its registration
  // when the connection it replaced goes away.
  {
    const WorkerId wc_id = rnd<WorkerId>(); const NodeId wc_node = rnd<NodeId>();
    const WorkerBootId wc_boot1 = rnd<WorkerBootId>(); const WorkerBootId wc_boot2 = rnd<WorkerBootId>();
    auto registration = [&](const WorkerBootId& boot) {
      Message r; r.kind = MessageKind::REGISTER; r.worker = wc_id; r.boot = boot;
      r.payload = "node_hex=" + wc_node.str() + "\nbackend=1\nmem=0x800000000\ndevice=cuda:0\nccmaj=c\ndevmem=0x800000000";
      return r;
    };
    RawConn first; RawConn second; Message a1; Message a2;
    check(first.open(port) && first.send(registration(wc_boot1)) && first.recv(a1) && a1.kind == MessageKind::REGISTER_ACK && a1.payload.rfind("ok",0)==0, "worker C registers");
    check(second.open(port) && second.send(registration(wc_boot2)) && second.recv(a2) && a2.kind == MessageKind::REGISTER_ACK && a2.payload.rfind("ok",0)==0, "worker C registers again on a new connection");
    check(first.finish(), "coordinator let go of the replaced connection");
    first.close();

    const ReplicaSetId set2 = rnd<ReplicaSetId>(); const ReplicaId rc = rnd<ReplicaId>();
    Message cs; cs.kind = MessageKind::CREATE_SET; cs.epoch = CoordinatorEpoch(); cs.seq = 2; cs.set = set2;
    cs.payload = "model=" + rnd<ModelId>().str() + "\nartifact=" + rnd<ArtifactId>().str() + "\nagen=" + rnd<ArtifactGeneration>().str() +
                 "\nsgen=" + rnd<ReplicaSetGeneration>().str() + "\npgen=" + rnd<PolicyGeneration>().str() + "\nworkload=" + rnd<WorkloadId>().str() +
                 "\ntenant=" + rnd<TenantId>().str() + "\ndesired=2\nmin=1\nmax=2\nbackend=2\nccmaj=8\nccmin=0";
    send_msg(control, cs); Message csa;
    check(recv_msg(control, csa) && csa.kind == MessageKind::CREATE_SET_ACK && csa.payload.rfind("ok",0)==0, "create a second replica set");
    Message pm; pm.kind = MessageKind::PROVISION; pm.epoch = CoordinatorEpoch(); pm.set = set2; pm.replica = rc; pm.seq = 2;
    pm.payload = "worker=" + wc_id.str() + "\nccmaj=c\ncapacity=0x1000000";
    send_msg(control, pm); Message pa;
    const bool provisioned = recv_msg(control, pa) && pa.kind == MessageKind::PROVISION_ACK && pa.payload.rfind("ok",0)==0;
    check(provisioned, "re-registered worker C is still registered (" + pa.payload + ")");
    if (provisioned) {
      Message warm;
      check(second.recv(warm) && warm.kind == MessageKind::WARM && warm.replica == rc, "re-registered worker C receives the warm command");
      // A worker report with an unparsable payload is discarded; the connection keeps working.
      Message hb; hb.kind = MessageKind::HEALTH; hb.worker = wc_id; hb.boot = wc_boot2; hb.replica = rc; hb.set = set2; hb.payload = "state=zz";
      Message ping; ping.kind = MessageKind::PING; Message pong;
      check(second.send(hb) && second.send(ping) && second.recv(pong) && pong.kind == MessageKind::PONG, "malformed worker report is discarded");
    }
    second.close();
  }

  // Malformed control requests are answered with an error and leave the
  // coordinator running.
  {
    Message bad; bad.kind = MessageKind::PROMOTE; bad.epoch = CoordinatorEpoch(); bad.replica = rb; bad.payload = "target=9";
    send_msg(control, bad); Message ba;
    check(recv_msg(control, ba) && ba.kind == MessageKind::MSG_ERROR && ba.payload.rfind("err",0)==0, "promotion to an unknown target is refused");
    Message bad2; bad2.kind = MessageKind::CREATE_SET; bad2.epoch = CoordinatorEpoch(); bad2.set = rnd<ReplicaSetId>(); bad2.payload = "model=not-an-identity";
    send_msg(control, bad2); Message bb;
    check(recv_msg(control, bb) && bb.kind == MessageKind::MSG_ERROR && bb.payload.rfind("err",0)==0, "replica set with an unparsable identity is refused");
    Message bad3; bad3.kind = MessageKind::PROVISION; bad3.epoch = CoordinatorEpoch(); bad3.set = setid; bad3.replica = rnd<ReplicaId>(); bad3.payload = "worker=" + wb_id.str() + "\ncapacity=zz";
    send_msg(control, bad3); Message bc;
    check(recv_msg(control, bc) && bc.kind == MessageKind::MSG_ERROR && bc.payload.rfind("err",0)==0, "provision with an unparsable number is refused");
    check(wait_state(rb, "serving=1", 1), "coordinator still answers and B still serves");
  }

  tcp_shutdown();
  wa2.kill(); wb.kill(); coord.kill();
  if (g_failures == 0) { std::cout << "\nMULTIPROCESS PROOF: ALL CHECKS PASSED\n"; return 0; }
  std::cout << "\nMULTIPROCESS PROOF: " << g_failures << " FAILURES\n";
  return 1;
}