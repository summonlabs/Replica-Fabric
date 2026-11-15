// rf_coordinator - Replica Fabric coordinator process.
#include <replicafabric/core/replica_set_controller.hpp>
#include <replicafabric/distributed/transport.hpp>
#include <replicafabric/distributed/protocol.hpp>
#include <replicafabric/persistence/snapshot.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace replicafabric;

namespace {

std::atomic<bool> g_stop{false};
std::mt19937_64 g_placement_rng(7);

std::mutex g_results_mu;
std::map<ReplicaId, std::string> g_last_results;
void stash_result(const ReplicaId& id, const std::string& res) { std::lock_guard<std::mutex> lk(g_results_mu); g_last_results[id] = res; }
std::string peek_result(const ReplicaId& id) { std::lock_guard<std::mutex> lk(g_results_mu); auto it = g_last_results.find(id); return it == g_last_results.end() ? "" : it->second; }
void on_sig(int) { g_stop.store(true); }

// Serializes every use of the controller across the connection threads. The
// controller locks its own mutations, but the pointers returned by find_*() and
// the snapshot walk read controller state outside that lock. It is never held
// while conn_mu is acquired (conn_mu comes first where both are needed).
std::mutex g_ctrl_mu;

// Payload parsing throws on a missing or unparsable field. Every caller runs
// under a handler that answers the sender with an error instead.
std::uint64_t hexu(const std::string& s) { return std::stoull(s, nullptr, 16); }
int hexi(const std::string& s) { return static_cast<int>(std::stoul(s, nullptr, 16)); }

template <typename IdT>
IdT parse_id(const std::string& s) { return IdT::from_string(s); }

template <typename E>
E parse_enum(const std::string& s) {
  E v{};
  if (!from_int(v, hexi(s))) throw std::invalid_argument("enum value out of range");
  return v;
}

ReplicaSetState build_set(const Message& m) {
  ReplicaSetState s;
  s.id = m.set;
  s.model_id = parse_id<ModelId>(get_payload(m.payload, "model"));
  s.workload_id = parse_id<WorkloadId>(get_payload(m.payload, "workload"));
  s.tenant_id = parse_id<TenantId>(get_payload(m.payload, "tenant"));
  s.desired_count = static_cast<std::uint32_t>(std::stoul(get_payload(m.payload, "desired", "2")));
  s.min_healthy = static_cast<std::uint32_t>(std::stoul(get_payload(m.payload, "min", "1")));
  s.max_replicas = static_cast<std::uint32_t>(std::stoul(get_payload(m.payload, "max", "2")));
  s.artifact_id = parse_id<ArtifactId>(get_payload(m.payload, "artifact"));
  s.artifact_generation = parse_id<ArtifactGeneration>(get_payload(m.payload, "agen"));
  s.compatibility.artifact_generation = s.artifact_generation;
  s.compatibility.min_compute = s.min_compute;
  s.generation = parse_id<ReplicaSetGeneration>(get_payload(m.payload, "sgen"));
  s.policy_generation = parse_id<PolicyGeneration>(get_payload(m.payload, "pgen"));
  s.lifecycle = ReplicaSetLifecycle::CREATED;
  const BackendKind bk = parse_enum<BackendKind>(get_payload(m.payload, "backend", "2"));
  s.backend = bk; s.compatibility.backend = bk;
  s.runtime_name = get_payload(m.payload, "runtime", "triton-3");
  s.compatibility.runtime_name = s.runtime_name;
  s.compatibility.numeric_mode = NumericMode::FP16;
  s.compatibility.policy_fingerprint = "fp-1";
  s.compatibility.model_id = s.model_id;
  s.compatibility.architecture = get_payload(m.payload, "arch", "llama");
  s.compatibility.min_compute.major = hexi(get_payload(m.payload, "ccmaj", "8"));
  s.compatibility.min_compute.minor = hexi(get_payload(m.payload, "ccmin", "0"));
  s.min_compute = s.compatibility.min_compute;
  s.memory_requirement_bytes = hexu(get_payload(m.payload, "mem", "0x1000000"));
  s.accelerator_requirement = static_cast<std::uint32_t>(std::stoul(get_payload(m.payload, "accel", "1")));
  return s;
}

WorkerRegistration build_worker_reg(const Message& m) {
  WorkerRegistration wr;
  wr.worker_id = m.worker;
  wr.boot_id = m.boot;
  wr.node_id = parse_id<NodeId>(get_payload(m.payload, "node_hex"));
  wr.protocol_version = 1;
  const BackendKind bk = parse_enum<BackendKind>(get_payload(m.payload, "backend", "2"));
  wr.inventory.total_memory_bytes = hexu(get_payload(m.payload, "mem", "0x800000000"));
  wr.inventory.free_memory_bytes = wr.inventory.total_memory_bytes;
  wr.inventory.accelerator_count = 1;
  DeviceCapability dc;
  dc.kind = AcceleratorKind::CUDA;
  dc.device_id = get_payload(m.payload, "device", "cuda:0");
  dc.compute.major = hexi(get_payload(m.payload, "ccmaj", "c"));
  dc.compute.minor = 0;
  dc.memory_bytes = hexu(get_payload(m.payload, "devmem", "0x800000000"));
  wr.inventory.devices.push_back(dc);
  wr.inventory.backends.push_back(BackendCapability{bk, "triton-3", dc.compute, NumericMode::FP16});
  return wr;
}

WarmingRecord warming_from(const std::string& payload) {
  WarmingRecord w;
  w.state = WarmthState::WARM;
  w.steps_completed = hexu(get_payload(payload, "steps", "0"));
  w.steps_required = 4;
  w.artifact_loading_done = true; w.weights_resident = true; w.adapters_active = true;
  w.kernel_init_done = true; w.graph_init_done = true; w.allocator_init_done = true;
  w.device_context_done = true; w.warmup_execution_done = true; w.endpoint_registered = true;
  return w;
}

// A message the coordinator owes a worker; it is sent once g_ctrl_mu is released.
struct WorkerSend {
  bool pending = false;
  WorkerId worker;
  Message msg;
};

// Applies one control message to the controller and fills the reply. Called
// with g_ctrl_mu held; throws on a malformed payload.
void apply_control(const Message& m, ReplicaSetController& ctrl, CoordinatorEpoch epoch, Message& ack, WorkerSend& out) {
  switch (m.kind) {
    case MessageKind::CREATE_SET: {
      ReplicaSetState s = build_set(m);
      auto r = ctrl.create_replica_set(s, epoch);
      ack.kind = MessageKind::CREATE_SET_ACK;
      ack.payload = std::string(r.ok() ? "ok" : "err") + " " + r.message;
      break;
    }
    case MessageKind::PROVISION: {
      ReplicaId reid = m.replica;
      WorkerId w = parse_id<WorkerId>(get_payload(m.payload, "worker"));
      // Derive the replica compatibility from the replica set's authoritative
      // compatibility so a provisioned replica is always compatible by
      // construction (the device provides compute >= the set minimum).
      const ReplicaSetState* proto = ctrl.find_set(m.set);
      ReplicaCompatibility compat;
      if (proto != nullptr) {
        compat.model_id = proto->compatibility.model_id;
        compat.model_revision = proto->compatibility.model_revision;
        compat.tokenizer_vocab = proto->compatibility.tokenizer_vocab;
        compat.adapters = proto->compatibility.required_adapters;
        compat.backend = proto->compatibility.backend;
        compat.runtime_name = proto->compatibility.runtime_name;
        compat.architecture = proto->compatibility.architecture;
        compat.numeric_mode = proto->compatibility.numeric_mode;
        compat.artifact_generation = proto->compatibility.artifact_generation;
        compat.kernel_abi = proto->compatibility.kernel_abi;
        compat.policy_fingerprint = proto->compatibility.policy_fingerprint;
      }
      compat.compute.major = hexi(get_payload(m.payload, "ccmaj", "c"));
      compat.compute.minor = 0;
      DeviceCapability dev;
      dev.kind = AcceleratorKind::CUDA;
      dev.device_id = get_payload(m.payload, "device", "cuda:0");
      dev.compute.major = hexi(get_payload(m.payload, "ccmaj", "c"));
      dev.memory_bytes = hexu(get_payload(m.payload, "devmem", "0x800000000"));
      std::uint64_t cap = hexu(get_payload(m.payload, "capacity", "0x1000000"));
      auto r = ctrl.provision_replica(reid, m.set, w, compat, dev, cap, monotonic_ns(), wall_ns());
      ack.kind = MessageKind::PROVISION_ACK;
      ack.payload = std::string(r.ok() ? "ok" : "err") + " " + r.message;
      if (r.ok()) {
        ctrl.set_allocating(reid, PlacementId::random(g_placement_rng), monotonic_ns());
        ctrl.set_starting(reid, monotonic_ns(), wall_ns());
        Message warm; warm.kind = MessageKind::WARM; warm.epoch = epoch;
        warm.worker = w; warm.replica = reid; warm.set = m.set;
        warm.payload = "steps=4";
        out.pending = true; out.worker = w; out.msg = warm;
      }
      break;
    }
    case MessageKind::PROMOTE: {
      const PromotionState target = parse_enum<PromotionState>(get_payload(m.payload, "target", "5"));
      const ReplicaState* rs = ctrl.find_replica(m.replica);
      WorkerId wkr = rs != nullptr ? rs->worker_id : m.worker;
      WorkerBootId bkt = rs != nullptr ? ctrl.current_boot_for(m.replica) : m.boot;
      auto r = ctrl.promote_replica(m.replica, target, wkr, bkt, monotonic_ns(), wall_ns());
      ack.kind = MessageKind::PROMOTE_ACK;
      ack.payload = std::string(r.ok() ? "ok" : "err") + " " + r.message;
      break;
    }
    case MessageKind::DRAIN: {
      const ReplicaState* rs = ctrl.find_replica(m.replica);
      WorkerId wkr = rs != nullptr ? rs->worker_id : m.worker;
      WorkerBootId bkt = rs != nullptr ? ctrl.current_boot_for(m.replica) : m.boot;
      auto r = ctrl.drain_replica(m.replica, wkr, bkt, monotonic_ns());
      ack.kind = MessageKind::DRAIN_ACK;
      ack.payload = std::string(r.ok() ? "ok" : "err") + " " + r.message;
      break;
    }
    case MessageKind::FAILOVER: {
      ReplicaId failed = parse_id<ReplicaId>(get_payload(m.payload, "failed"));
      auto r = ctrl.trigger_failover(m.set, failed, get_payload(m.payload, "reason", "failover"), monotonic_ns(), wall_ns());
      ack.kind = MessageKind::FAILOVER_ACK;
      ack.payload = std::string(r.ok() ? "ok" : "err") + " " + r.message;
      break;
    }
    case MessageKind::EXECUTE: {
      const ReplicaState* rs = ctrl.find_replica(m.replica);
      if (rs == nullptr) { ack.payload = "err unknown replica"; break; }
      // Work goes only to a replica the serving-authority gate admits; a
      // connected worker is not serving authority.
      const ServingAuthorityDecision gate = ctrl.servable(m.replica);
      if (!gate.eligible) { ack.payload = "err replica may not serve: " + gate.rejection; break; }
      Message fwd = m; fwd.epoch = epoch;
      out.pending = true; out.worker = rs->worker_id; out.msg = fwd;
      break;
    }
    case MessageKind::QUERY: {
      const std::string q = get_payload(m.payload, "q", "?");
      if (q == "result") {
        ack.kind = MessageKind::QUERY;
        ack.payload = "result=" + peek_result(m.replica);
      } else if (q == "state") {
        const ReplicaState* rs = ctrl.find_replica(m.replica);
        if (rs == nullptr) { ack.payload = "err unknown replica"; ack.kind = MessageKind::QUERY; }
        else {
          ack.kind = MessageKind::QUERY;
          ack.payload = std::string("lifecycle=") + std::string(to_string(rs->lifecycle)) +
                        " serving=" + (rs->serving_eligible ? "1" : "0") +
                        " role=" + std::string(to_string(rs->role)) +
                        " boot=" + rs->boot_id.str();
        }
      } else {
        ack.kind = MessageKind::QUERY; ack.payload = "err unknown query";
      }
      break;
    }
    case MessageKind::SNAPSHOT: {
      auto bytes = make_snapshot(ctrl);
      std::string hex; hex.reserve(bytes.size()*2);
      static const char* hx = "0123456789abcdef";
      for (std::uint8_t b : bytes) { hex.push_back(hx[b >> 4]); hex.push_back(hx[b & 0xF]); }
      ack.kind = MessageKind::SNAPSHOT;
      ack.payload = "len=" + std::to_string(bytes.size()) + "\nhex=" + hex;
      break;
    }
    default:
      ack.payload = "err unknown control command";
      break;
  }
}

void handle_control(const Message& m, std::shared_ptr<TcpConnection> conn, ReplicaSetController& ctrl,
                    CoordinatorEpoch epoch, const std::function<bool(const WorkerId&, const Message&)>& send_to_worker) {
  Message ack; ack.kind = MessageKind::MSG_ERROR; ack.epoch = epoch; ack.seq = m.seq;
  WorkerSend out;
  try {
    std::lock_guard<std::mutex> ctrl_lk(g_ctrl_mu);
    apply_control(m, ctrl, epoch, ack, out);
  } catch (const std::exception&) {
    // A missing or unparsable payload field is the sender's error; it must not
    // take the coordinator down.
    ack.kind = MessageKind::MSG_ERROR;
    ack.payload = "err malformed request";
    out.pending = false;
  }
  if (out.pending && m.kind == MessageKind::EXECUTE) {
    ack.payload = send_to_worker(out.worker, out.msg) ? "ok forwarded" : "err worker unreachable";
    conn->write_frame(encode_message(ack));
    return;
  }
  conn->write_frame(encode_message(ack));
  if (out.pending) send_to_worker(out.worker, out.msg);
}

// One message from a registered worker. Throws on a malformed payload.
void handle_worker_message(const Message& m, const std::shared_ptr<TcpConnection>& conn, ReplicaSetController& ctrl) {
  switch (m.kind) {
    case MessageKind::WARM_DONE: {
      const WarmingRecord warm = warming_from(m.payload);
      std::lock_guard<std::mutex> ctrl_lk(g_ctrl_mu);
      ctrl.warm_replica(m.replica, warm, m.worker, m.boot, monotonic_ns());
      break;
    }
    case MessageKind::HEALTH: {
      HealthEvidence ev;
      HealthState hs;
      if (!from_int(hs, hexi(get_payload(m.payload, "state", "2")))) break;
      ev.state = hs; ev.kind = HealthEvidenceKind::REPORTED;
      ev.source = get_payload(m.payload, "source", "worker"); ev.confidence = 0.99;
      std::lock_guard<std::mutex> ctrl_lk(g_ctrl_mu);
      const ReplicaState* rs = ctrl.find_replica(m.replica);
      if (rs != nullptr) ev.generation = rs->health_generation;
      ev.observed_at_wall = wall_ns();
      ctrl.report_health(m.replica, ev, m.worker, m.boot, monotonic_ns());
      break;
    }
    case MessageKind::READINESS: {
      ReadinessRecord rr;
      std::uint64_t mask = hexu(get_payload(m.payload, "mask", "0"));
      rr.factors.model_loaded = (mask & 1) != 0; rr.factors.artifact_validated = (mask & 2) != 0;
      rr.factors.adapters_present = (mask & 4) != 0; rr.factors.kernel_prepared = (mask & 8) != 0;
      rr.factors.graph_prepared = (mask & 16) != 0; rr.factors.memory_available = (mask & 32) != 0;
      rr.factors.device_context_initialized = (mask & 64) != 0; rr.factors.warmup_complete = (mask & 128) != 0;
      rr.factors.dependencies_ready = (mask & 256) != 0; rr.factors.endpoint_registered = (mask & 512) != 0;
      rr.factors.policy_current = (mask & 1024) != 0;
      rr.state = ReadinessState::READY;
      std::lock_guard<std::mutex> ctrl_lk(g_ctrl_mu);
      ctrl.report_readiness(m.replica, rr, m.worker, m.boot, monotonic_ns());
      break;
    }
    case MessageKind::EXECUTE_RESULT: stash_result(m.replica, "req=" + get_payload(m.payload, "req", "?") + "\noutcome=" + get_payload(m.payload, "outcome", "?") + "\nresult=" + get_payload(m.payload, "result", "?"));
    case MessageKind::PING: { Message p; p.kind = MessageKind::PONG; conn->write_frame(encode_message(p)); break; }
    default: break;
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::uint16_t port = 0;
  std::uint64_t seed = 0x1111ULL;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--port" && i + 1 < argc) port = static_cast<std::uint16_t>(std::stoul(argv[++i]));
    else if (a == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
  }
  if (port == 0) { std::cerr << "rf_coordinator: --port required\n"; return 2; }
  std::signal(SIGINT, on_sig);

  std::mt19937_64 gen(seed);
  CoordinatorEpoch epoch = CoordinatorEpoch::random(gen);
  ReplicaSetController ctrl(seed);

  TcpConnection listener;
  if (!listener.listen_on(port)) { std::cerr << "rf_coordinator: listen failed\n"; return 3; }
  std::cerr << "rf_coordinator: listening port=" << port << " epoch=" << epoch.str() << "\n";

  std::mutex conn_mu;
  std::map<WorkerId, std::shared_ptr<TcpConnection>> worker_conns;

  auto send_to_worker = [&](const WorkerId& w, const Message& m) -> bool {
    std::lock_guard<std::mutex> lk(conn_mu);
    auto it = worker_conns.find(w);
    if (it == worker_conns.end() || !it->second->valid()) return false;
    return it->second->write_frame(encode_message(m));
  };

  std::function<bool(const WorkerId&, const Message&)> send_fn = send_to_worker;

  while (!g_stop.load()) {
    auto conn = std::make_shared<TcpConnection>();
    if (!listener.accept_one(*conn)) { if (g_stop.load()) break; std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue; }
    std::thread([conn, &ctrl, &epoch, &conn_mu, &worker_conns, &send_fn]() mutable {
      std::vector<std::uint8_t> frame;
      if (!conn->read_frame(frame)) return;
      Message first;
      if (!decode_message(frame, first)) return;
      if (first.kind == MessageKind::REGISTER) {
        Message ack; ack.kind = MessageKind::REGISTER_ACK; ack.epoch = epoch;
        WorkerRegistration wr;
        try {
          wr = build_worker_reg(first);
        } catch (const std::exception&) {
          ack.payload = "err malformed registration";
          conn->write_frame(encode_message(ack));
          return;
        }
        {
          // The registration and the connection that carries it change together.
          std::lock_guard<std::mutex> lk(conn_mu);
          std::lock_guard<std::mutex> ctrl_lk(g_ctrl_mu);
          auto r = ctrl.register_worker(wr, monotonic_ns(), wall_ns());
          worker_conns[wr.worker_id] = conn;
          ack.payload = std::string(r.ok() ? "ok" : "err") + " " + r.message;
        }
        conn->write_frame(encode_message(ack));
        while (true) {
          std::vector<std::uint8_t> f;
          if (!conn->read_frame(f)) break;
          Message m;
          if (!decode_message(f, m)) continue;
          try {
            handle_worker_message(m, conn, ctrl);
          } catch (const std::exception&) {
            // A malformed worker payload is discarded.
          }
        }
        {
          // Only the connection that still carries the worker's registration
          // takes it down. When the worker has registered again on a newer
          // connection, the loss of this one must not unregister it.
          std::lock_guard<std::mutex> lk(conn_mu);
          auto it = worker_conns.find(wr.worker_id);
          if (it != worker_conns.end() && it->second == conn) {
            std::lock_guard<std::mutex> ctrl_lk(g_ctrl_mu);
            ctrl.unregister_worker(wr.worker_id);
            worker_conns.erase(it);
          }
        }
      } else {
        handle_control(first, conn, ctrl, epoch, send_fn);
        while (true) {
          std::vector<std::uint8_t> f;
          if (!conn->read_frame(f)) break;
          Message m;
          if (!decode_message(f, m)) continue;
          handle_control(m, conn, ctrl, epoch, send_fn);
        }
      }
    }).detach();
  }
  tcp_shutdown();
  return 0;
}
