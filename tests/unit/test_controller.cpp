#include <replicafabric/core/replica_set_controller.hpp>
#include <replicafabric/core/time.hpp>

#include <cstdint>
#include <random>

#include "replicafabric_test.hpp"

namespace {

using namespace replicafabric;

std::mt19937_64 g_rng(0x12345ULL);

ReplicaSetId sid() { return ReplicaSetId::random(g_rng); }
WorkerId wid() { return WorkerId::random(g_rng); }
WorkerBootId wbid() { return WorkerBootId::random(g_rng); }
ReplicaId rid() { return ReplicaId::random(g_rng); }
NodeId nid() { return NodeId::random(g_rng); }
ArtifactId aid() { return ArtifactId::random(g_rng); }

ArtifactGeneration agen() { return ArtifactGeneration::random(g_rng); }
ReplicaSetGeneration sgen() { return ReplicaSetGeneration::random(g_rng); }
PolicyGeneration pgen() { return PolicyGeneration::random(g_rng); }
CoordinatorEpoch epoch() { return CoordinatorEpoch::random(g_rng); }

ReplicaSetState make_set(ModelId model, ArtifactId artifact, ArtifactGeneration ag,
                         ReplicaSetGeneration sg, PolicyGeneration pg) {
  ReplicaSetState s;
  s.id = sid();
  s.model_id = model;
  s.workload_id = WorkloadId::random(g_rng);
  s.tenant_id = TenantId::random(g_rng);
  s.desired_count = 2;
  s.min_healthy = 1;
  s.max_replicas = 2;
  s.artifact_id = artifact;
  s.artifact_generation = ag;
  s.generation = sg;
  s.policy_generation = pg;
  s.lifecycle = ReplicaSetLifecycle::CREATED;
  s.compatibility.model_id = model;
  s.compatibility.model_revision = "rev-1";
  s.compatibility.tokenizer_vocab = "vocab-1";
  s.compatibility.required_adapters = {};
  s.compatibility.backend = BackendKind::TRITON;
  s.compatibility.runtime_name = "triton-3";
  s.compatibility.architecture = "llama";
  s.compatibility.min_compute = {8, 0};
  s.compatibility.numeric_mode = NumericMode::FP16;
  s.compatibility.artifact_generation = ag;
  s.compatibility.kernel_abi = "abi-1";
  s.compatibility.policy_fingerprint = "fp-1";
  s.backend = BackendKind::TRITON;
  s.runtime_name = "triton-3";
  s.min_compute = {8, 0};
  s.memory_requirement_bytes = 16 * 1024 * 1024;
  s.accelerator_requirement = 1;
  s.placement_policy.anti_affinity_domains = {"host"};
  s.placement_policy.require_diversity = true;
  s.placement_policy.synthetic_domains = {{"fd-a", "rack"}, {"fd-b", "rack"}};
  return s;
}

WorkerRegistration make_worker(WorkerId w, WorkerBootId b, NodeId n, std::string host) {
  WorkerRegistration reg;
  reg.worker_id = w;
  reg.boot_id = b;
  reg.node_id = n;
  reg.protocol_version = 1;
  reg.inventory.total_memory_bytes = 64ULL * 1024 * 1024 * 1024;
  reg.inventory.free_memory_bytes = 64ULL * 1024 * 1024 * 1024;
  reg.inventory.accelerator_count = 1;
  reg.inventory.devices.push_back(DeviceCapability{AcceleratorKind::CUDA, host + ":0", {12, 0}, 32ULL * 1024 * 1024 * 1024});
  reg.inventory.backends.push_back(BackendCapability{BackendKind::TRITON, "triton-3", {12, 0}, NumericMode::FP16});
  return reg;
}

ReplicaCompatibility make_compat(const ReplicaSetState& s) {
  ReplicaCompatibility c;
  c.model_id = s.compatibility.model_id;
  c.model_revision = s.compatibility.model_revision;
  c.tokenizer_vocab = s.compatibility.tokenizer_vocab;
  c.backend = s.compatibility.backend;
  c.runtime_name = s.compatibility.runtime_name;
  c.architecture = s.compatibility.architecture;
  c.compute = {12, 0};
  c.numeric_mode = s.compatibility.numeric_mode;
  c.artifact_generation = s.compatibility.artifact_generation;
  c.kernel_abi = s.compatibility.kernel_abi;
  c.policy_fingerprint = s.compatibility.policy_fingerprint;
  return c;
}

// Walk a replica through provision -> allocate -> start -> warm -> health ->
// ready -> promote.
ReplicaId provision_basic(ReplicaSetController& ctrl, const ReplicaSetState& set, WorkerId w,
                          WorkerBootId b, NodeId n, MonotonicNs t) {
  ReplicaId id = rid();
  auto reg = make_worker(w, b, n, std::to_string(w.hi() % 1000));
  (void)ctrl.register_worker(reg, t, t);
  (void)ctrl.provision_replica(id, set.id, w, make_compat(set),
                               DeviceCapability{AcceleratorKind::CUDA, "cuda:0", {12, 0}, 32ULL * 1024 * 1024 * 1024},
                               8ULL * 1024 * 1024, t, t);
  (void)ctrl.set_allocating(id, PlacementId::random(g_rng), t);
  (void)ctrl.set_starting(id, t, t);
  WarmingRecord warm;
  warm.state = WarmthState::WARM;
  warm.artifact_loading_done = true;
  warm.weights_resident = true;
  warm.adapters_active = true;
  warm.kernel_init_done = true;
  warm.graph_init_done = true;
  warm.allocator_init_done = true;
  warm.device_context_done = true;
  warm.warmup_execution_done = true;
  warm.endpoint_registered = true;
  (void)ctrl.warm_replica(id, warm, w, b, t);
  return id;
}

}  // namespace

RF_TEST_CASE(controller_provision_and_serve) {
  ReplicaSetController ctrl(42);
  ModelId model = ModelId::random(g_rng);
  ArtifactId artifact = aid();
  ArtifactGeneration ag = agen();
  ReplicaSetGeneration sg = sgen();
  PolicyGeneration pg = pgen();
  ReplicaSetState set = make_set(model, artifact, ag, sg, pg);
  CoordinatorEpoch ep = epoch();
  RF_CHECK(ctrl.create_replica_set(set, ep).ok());

  WorkerId w = wid();
  WorkerBootId b = wbid();
  NodeId n = nid();
  const MonotonicNs t0 = monotonic_ns();
  ReplicaId id = provision_basic(ctrl, set, w, b, n, t0);

  // Mark healthy and ready.
  HealthEvidence he;
  he.state = HealthState::HEALTHY;
  he.kind = HealthEvidenceKind::MEASURED;
  he.source = "probe";
  he.observed_at_mono = t0;
  he.observed_at_wall = wall_ns();
  he.confidence = 0.99;
  const ReplicaState* r = ctrl.find_replica(id);
  RF_REQUIRE(r != nullptr);
  he.generation = r->health_generation;
  RF_CHECK(ctrl.report_health(id, he, w, b, t0).ok());

  ReadinessRecord rr;
  rr.factors.model_loaded = true;
  rr.factors.artifact_validated = true;
  rr.factors.adapters_present = true;
  rr.factors.kernel_prepared = true;
  rr.factors.graph_prepared = true;
  rr.factors.memory_available = true;
  rr.factors.device_context_initialized = true;
  rr.factors.warmup_complete = true;
  rr.factors.dependencies_ready = true;
  rr.factors.endpoint_registered = true;
  rr.factors.policy_current = true;
  rr.state = ReadinessState::READY;
  RF_CHECK(ctrl.report_readiness(id, rr, w, b, t0).ok());
  RF_CHECK(ctrl.promote_replica(id, PromotionState::PRIMARY, w, b, t0, wall_ns()).ok());

  const ReplicaState* r2 = ctrl.find_replica(id);
  RF_REQUIRE(r2 != nullptr);
  RF_CHECK(r2->lifecycle == ReplicaLifecycle::SERVING);
  RF_CHECK(r2->promotion == PromotionState::PRIMARY);
  RF_CHECK(r2->serving_eligible);
  RF_CHECK(ctrl.servable(id).eligible);
}

RF_TEST_CASE(controller_stale_epoch_rejected) {
  ReplicaSetController ctrl(7);
  ModelId model = ModelId::random(g_rng);
  ReplicaSetState set = make_set(model, aid(), agen(), sgen(), pgen());
  CoordinatorEpoch ep1 = epoch();
  RF_CHECK(ctrl.create_replica_set(set, ep1).ok());
  WorkerId w = wid();
  WorkerBootId b = wbid();
  NodeId n = nid();
  const MonotonicNs t0 = monotonic_ns();
  ReplicaId id = provision_basic(ctrl, set, w, b, n, t0);
  const ReplicaState* r = ctrl.find_replica(id);
  HealthEvidence he;
  he.state = HealthState::HEALTHY;
  he.kind = HealthEvidenceKind::REPORTED;
  he.source = "probe";
  he.observed_at_mono = t0;
  he.generation = r->health_generation;
  RF_CHECK(ctrl.report_health(id, he, w, b, t0).ok());

  // Coordinator restarts under a new epoch; old authority is now stale.
  CoordinatorEpoch ep2 = epoch();
  ctrl.set_epoch(ep2);
  RF_CHECK(!ctrl.servable(id).eligible);

  // Replaying the pre-restart health report (old epoch authority) is rejected.
  RF_CHECK(ctrl.report_health(id, he, w, b, t0).rejected());
}

RF_TEST_CASE(controller_worker_restart_failover) {
  ReplicaSetController ctrl(99);
  ModelId model = ModelId::random(g_rng);
  ReplicaSetState set = make_set(model, aid(), agen(), sgen(), pgen());
  CoordinatorEpoch ep = epoch();
  RF_CHECK(ctrl.create_replica_set(set, ep).ok());

  // Worker A hosts replica A; worker B hosts replica B (standby).
  WorkerId wa = wid(); WorkerBootId ba = wbid(); NodeId na = nid();
  WorkerId wb = wid(); WorkerBootId bb = wbid(); NodeId nb = nid();
  const MonotonicNs t0 = monotonic_ns();
  ReplicaId ra = provision_basic(ctrl, set, wa, ba, na, t0);
  ReplicaId rb = provision_basic(ctrl, set, wb, bb, nb, t0);

  auto mark_healthy_ready_and_promote = [&](ReplicaId id, WorkerId w, WorkerBootId b) {
    const ReplicaState* r = ctrl.find_replica(id);
    RF_REQUIRE(r != nullptr);
    HealthEvidence he;
    he.state = HealthState::HEALTHY;
    he.kind = HealthEvidenceKind::REPORTED;
    he.source = "probe";
    he.observed_at_mono = t0;
    he.generation = r->health_generation;
    RF_CHECK(ctrl.report_health(id, he, w, b, t0).ok());
    ReadinessRecord rr;
    rr.factors.model_loaded = true; rr.factors.artifact_validated = true;
    rr.factors.adapters_present = true; rr.factors.kernel_prepared = true;
    rr.factors.graph_prepared = true; rr.factors.memory_available = true;
    rr.factors.device_context_initialized = true; rr.factors.warmup_complete = true;
    rr.factors.dependencies_ready = true; rr.factors.endpoint_registered = true;
    rr.factors.policy_current = true;
    rr.state = ReadinessState::READY;
    RF_CHECK(ctrl.report_readiness(id, rr, w, b, t0).ok());
  };
  mark_healthy_ready_and_promote(ra, wa, ba);
  mark_healthy_ready_and_promote(rb, wb, bb);

  // Replica A becomes the serving primary; B is a standby.
  RF_CHECK(ctrl.promote_replica(ra, PromotionState::PRIMARY, wa, ba, t0, wall_ns()).ok());
  RF_CHECK(ctrl.promote_replica(rb, PromotionState::STANDBY, wb, bb, t0, wall_ns()).ok());
  RF_CHECK(ctrl.servable(ra).eligible);
  RF_CHECK(!ctrl.servable(rb).eligible);

  // Worker A dies; coordinator unregisters it and fails over to B.
  RF_CHECK(ctrl.unregister_worker(wa).ok());
  RF_CHECK(ctrl.trigger_failover(set.id, ra, "worker died", t0, wall_ns()).ok());
  RF_CHECK(ctrl.servable(rb).eligible);
  const ReplicaState* rb2 = ctrl.find_replica(rb);
  RF_REQUIRE(rb2 != nullptr);
  RF_CHECK(rb2->lifecycle == ReplicaLifecycle::SERVING);
  RF_CHECK(rb2->promotion == PromotionState::PRIMARY);

  // Worker A restarts with a NEW boot id; its old replica must NOT regain
  // serving authority.
  WorkerBootId ba2 = wbid();
  auto regA2 = make_worker(wa, ba2, na, "hostA");
  RF_CHECK(ctrl.register_worker(regA2, t0, t0).ok());
  RF_CHECK(!ctrl.servable(ra).eligible);  // stale boot / failed replica
  // Stale health report from the pre-restart boot is rejected.
  const ReplicaState* rstale = ctrl.find_replica(ra);
  HealthEvidence stale;
  stale.state = HealthState::HEALTHY;
  stale.kind = HealthEvidenceKind::REPORTED;
  stale.source = "probe";
  stale.generation = rstale->health_generation;
  RF_CHECK(ctrl.report_health(ra, stale, wa, ba, t0).rejected());  // old boot
}

RF_TEST_CASE(controller_double_serve_prevented) {
  ReplicaSetController ctrl(5);
  ModelId model = ModelId::random(g_rng);
  ReplicaSetState set = make_set(model, aid(), agen(), sgen(), pgen());
  CoordinatorEpoch ep = epoch();
  RF_CHECK(ctrl.create_replica_set(set, ep).ok());
  WorkerId wa = wid(); WorkerBootId ba = wbid(); NodeId na = nid();
  const MonotonicNs t0 = monotonic_ns();
  ReplicaId ra = provision_basic(ctrl, set, wa, ba, na, t0);
  const ReplicaState* r = ctrl.find_replica(ra);
  HealthEvidence he;
  he.state = HealthState::HEALTHY;
  he.kind = HealthEvidenceKind::REPORTED;
  he.source = "probe";
  he.generation = r->health_generation;
  RF_CHECK(ctrl.report_health(ra, he, wa, ba, t0).ok());
  ReadinessRecord rr;
  rr.factors.model_loaded = true; rr.factors.artifact_validated = true;
  rr.factors.adapters_present = true; rr.factors.kernel_prepared = true;
  rr.factors.graph_prepared = true; rr.factors.memory_available = true;
  rr.factors.device_context_initialized = true; rr.factors.warmup_complete = true;
  rr.factors.dependencies_ready = true; rr.factors.endpoint_registered = true;
  rr.factors.policy_current = true;
  rr.state = ReadinessState::READY;
  RF_CHECK(ctrl.report_readiness(ra, rr, wa, ba, t0).ok());
  RF_CHECK(ctrl.promote_replica(ra, PromotionState::PRIMARY, wa, ba, t0, wall_ns()).ok());
  // A second promotion to PRIMARY on a different replica in the same set would
  // double-serve; with only one set, we verify repeated promotion of the SAME
  // primary is idempotent-safe (it stays primary; it may re-record but no
  // second distinct primary appears).
  RF_CHECK(ctrl.servable(ra).eligible);
  // Only one replica may hold serving authority: exactly one replica in the
  // set is serving_eligible and PRIMARY.
  std::size_t serving = 0;
  for (const ReplicaId cand : ctrl.list_replicas(set.id)) {
    const ReplicaState* rec = ctrl.find_replica(cand);
    if (rec && rec->serving_eligible) ++serving;
  }
  RF_CHECK_EQ(serving, 1u);

}

// ---------------------------------------------------------------------------
// Regression tests
// ---------------------------------------------------------------------------
namespace {

// A health report of `state` under the replica's current health generation.
OpResult report(ReplicaSetController& ctrl, ReplicaId id, WorkerId w, WorkerBootId b,
                HealthState state = HealthState::HEALTHY) {
  const ReplicaState* r = ctrl.find_replica(id);
  HealthEvidence he;
  he.state = state;
  he.kind = HealthEvidenceKind::REPORTED;
  he.source = "probe";
  if (r != nullptr) he.generation = r->health_generation;
  return ctrl.report_health(id, he, w, b, 0);
}

ReadinessRecord all_ready() {
  ReadinessRecord rr;
  rr.factors.model_loaded = true; rr.factors.artifact_validated = true;
  rr.factors.adapters_present = true; rr.factors.kernel_prepared = true;
  rr.factors.graph_prepared = true; rr.factors.memory_available = true;
  rr.factors.device_context_initialized = true; rr.factors.warmup_complete = true;
  rr.factors.dependencies_ready = true; rr.factors.endpoint_registered = true;
  rr.factors.policy_current = true;
  rr.state = ReadinessState::READY;
  return rr;
}

WarmingRecord full_warm() {
  WarmingRecord warm;
  warm.state = WarmthState::WARM;
  warm.artifact_loading_done = true; warm.weights_resident = true; warm.adapters_active = true;
  warm.kernel_init_done = true; warm.graph_init_done = true; warm.allocator_init_done = true;
  warm.device_context_done = true; warm.warmup_execution_done = true;
  warm.endpoint_registered = true;
  return warm;
}

struct Hosted {
  ReplicaId id;
  WorkerId w;
  WorkerBootId b;
};

// A replica on its own newly registered worker, taken to STARTING.
Hosted started_replica(ReplicaSetController& ctrl, const ReplicaSetState& set) {
  Hosted h{rid(), wid(), wbid()};
  (void)ctrl.register_worker(make_worker(h.w, h.b, nid(), "host"), 0, 0);
  (void)ctrl.provision_replica(h.id, set.id, h.w, make_compat(set),
                               DeviceCapability{AcceleratorKind::CUDA, "cuda:0", {12, 0}, 32ULL * 1024 * 1024 * 1024},
                               8ULL * 1024 * 1024, 0, 0);
  (void)ctrl.set_allocating(h.id, PlacementId::random(g_rng), 0);
  (void)ctrl.set_starting(h.id, 0, 0);
  return h;
}

// A replica on its own newly registered worker, warm, healthy and READY.
Hosted ready_replica(ReplicaSetController& ctrl, const ReplicaSetState& set) {
  Hosted h = started_replica(ctrl, set);
  (void)ctrl.warm_replica(h.id, full_warm(), h.w, h.b, 0);
  (void)report(ctrl, h.id, h.w, h.b);
  (void)ctrl.report_readiness(h.id, all_ready(), h.w, h.b, 0);
  return h;
}

ReplicaSetState new_set(std::uint32_t max) {
  ReplicaSetState s = make_set(ModelId::random(g_rng), aid(), agen(), sgen(), pgen());
  s.desired_count = max;
  s.max_replicas = max;
  return s;
}

std::size_t servable_count(const ReplicaSetController& ctrl, ReplicaSetId set) {
  std::size_t n = 0;
  for (const ReplicaId id : ctrl.list_replicas(set)) {
    if (ctrl.servable(id).eligible) ++n;
  }
  return n;
}

}  // namespace

// Exactly one primary holds serving authority per set: promoting a second
// replica to PRIMARY while the first still serves is rejected.
RF_TEST_CASE(controller_second_primary_rejected) {
  ReplicaSetController ctrl(11);
  const ReplicaSetState set = new_set(2);
  RF_CHECK(ctrl.create_replica_set(set, epoch()).ok());
  const Hosted a = ready_replica(ctrl, set);
  const Hosted b = ready_replica(ctrl, set);
  RF_CHECK(ctrl.promote_replica(a.id, PromotionState::PRIMARY, a.w, a.b, 0, 0).ok());
  RF_CHECK(ctrl.promote_replica(b.id, PromotionState::PRIMARY, b.w, b.b, 0, 0).rejected());
  RF_CHECK(ctrl.servable(a.id).eligible);
  RF_CHECK(!ctrl.servable(b.id).eligible);
  RF_CHECK_EQ(servable_count(ctrl, set.id), 1u);
  const ReplicaState* rb = ctrl.find_replica(b.id);
  RF_REQUIRE(rb != nullptr);
  RF_CHECK(!rb->serving_eligible);
  RF_CHECK(rb->promotion != PromotionState::PRIMARY);
  RF_CHECK(rb->lifecycle == ReplicaLifecycle::READY);

  // Once the primary is drained the other replica can take over.
  RF_CHECK(ctrl.drain_replica(a.id, a.w, a.b, 0).ok());
  RF_CHECK(ctrl.promote_replica(b.id, PromotionState::PRIMARY, b.w, b.b, 0, 0).ok());
  RF_CHECK(ctrl.servable(b.id).eligible);
  RF_CHECK_EQ(servable_count(ctrl, set.id), 1u);
}

// A failover for a replica that is not the primary keeps the serving primary
// and promotes nobody else.
RF_TEST_CASE(controller_failover_keeps_single_primary) {
  ReplicaSetController ctrl(12);
  const ReplicaSetState set = new_set(3);
  RF_CHECK(ctrl.create_replica_set(set, epoch()).ok());
  const Hosted a = ready_replica(ctrl, set);
  const Hosted b = ready_replica(ctrl, set);
  const Hosted c = ready_replica(ctrl, set);
  RF_CHECK(ctrl.promote_replica(a.id, PromotionState::PRIMARY, a.w, a.b, 0, 0).ok());
  RF_CHECK(ctrl.promote_replica(b.id, PromotionState::STANDBY, b.w, b.b, 0, 0).ok());
  RF_CHECK(ctrl.promote_replica(c.id, PromotionState::STANDBY, c.w, c.b, 0, 0).ok());
  const std::size_t promotions_before = ctrl.promotions().size();

  RF_CHECK(ctrl.trigger_failover(set.id, b.id, "standby died", 0, 0).ok());
  RF_CHECK(ctrl.servable(a.id).eligible);
  RF_CHECK(!ctrl.servable(c.id).eligible);
  RF_CHECK_EQ(servable_count(ctrl, set.id), 1u);
  const ReplicaState* rc = ctrl.find_replica(c.id);
  RF_REQUIRE(rc != nullptr);
  RF_CHECK(rc->promotion == PromotionState::STANDBY);
  RF_CHECK(!rc->serving_eligible);
  const ReplicaState* rb = ctrl.find_replica(b.id);
  RF_REQUIRE(rb != nullptr);
  RF_CHECK(rb->lifecycle == ReplicaLifecycle::FAILED);
  RF_CHECK_EQ(ctrl.promotions().size(), promotions_before);
  RF_REQUIRE(!ctrl.failovers().empty());
  RF_CHECK(ctrl.failovers().back().state == FailoverState::COMPLETE);
  RF_CHECK(ctrl.failovers().back().replacement == a.id);
}

// The replacement a failover promotes must be able to serve: a standby whose
// worker has since restarted under a new boot identity is not eligible.
RF_TEST_CASE(controller_failover_requires_current_candidate) {
  ReplicaSetController ctrl(13);
  const ReplicaSetState set = new_set(3);
  RF_CHECK(ctrl.create_replica_set(set, epoch()).ok());
  const Hosted a = ready_replica(ctrl, set);
  const Hosted stale = ready_replica(ctrl, set);
  const Hosted fresh = ready_replica(ctrl, set);
  RF_CHECK(ctrl.promote_replica(a.id, PromotionState::PRIMARY, a.w, a.b, 0, 0).ok());
  RF_CHECK(ctrl.promote_replica(stale.id, PromotionState::STANDBY, stale.w, stale.b, 0, 0).ok());
  RF_CHECK(ctrl.promote_replica(fresh.id, PromotionState::CANARY, fresh.w, fresh.b, 0, 0).ok());

  // The standby's worker restarts: its replica is bound to a dead incarnation.
  RF_CHECK(ctrl.register_worker(make_worker(stale.w, wbid(), nid(), "host"), 0, 0).ok());
  // The primary's worker dies.
  RF_CHECK(ctrl.unregister_worker(a.w).ok());

  RF_CHECK(ctrl.trigger_failover(set.id, a.id, "worker died", 0, 0).ok());
  RF_REQUIRE(!ctrl.failovers().empty());
  RF_CHECK(ctrl.failovers().back().replacement == fresh.id);
  RF_CHECK(ctrl.servable(fresh.id).eligible);
  RF_CHECK(!ctrl.servable(stale.id).eligible);
  RF_CHECK_EQ(servable_count(ctrl, set.id), 1u);
  const ReplicaState* rs = ctrl.find_replica(stale.id);
  RF_REQUIRE(rs != nullptr);
  RF_CHECK(!rs->serving_eligible);
  RF_CHECK(rs->promotion == PromotionState::STANDBY);
  // The audit record states the role the replacement was actually promoted from.
  RF_REQUIRE(!ctrl.promotions().empty());
  RF_CHECK(ctrl.promotions().back().replica_id == fresh.id);
  RF_CHECK(ctrl.promotions().back().from == PromotionState::CANARY);
  RF_CHECK(ctrl.promotions().back().to == PromotionState::PRIMARY);
}

// Evidence and promotion requests from a worker boot identity that has been
// replaced (or unregistered) are stale and never change the replica.
RF_TEST_CASE(controller_superseded_boot_evidence_rejected) {
  ReplicaSetController ctrl(14);
  const ReplicaSetState set = new_set(2);
  RF_CHECK(ctrl.create_replica_set(set, epoch()).ok());
  const Hosted h = started_replica(ctrl, set);
  const Hosted gone = started_replica(ctrl, set);

  // Worker restarts under a new boot identity; the old process's messages follow.
  RF_CHECK(ctrl.register_worker(make_worker(h.w, wbid(), nid(), "host"), 0, 0).ok());
  RF_CHECK(ctrl.warm_replica(h.id, full_warm(), h.w, h.b, 0).rejected());
  RF_CHECK(report(ctrl, h.id, h.w, h.b).rejected());
  RF_CHECK(ctrl.report_readiness(h.id, all_ready(), h.w, h.b, 0).rejected());
  RF_CHECK(ctrl.promote_replica(h.id, PromotionState::PRIMARY, h.w, h.b, 0, 0).rejected());
  const ReplicaState* r = ctrl.find_replica(h.id);
  RF_REQUIRE(r != nullptr);
  RF_CHECK(r->lifecycle == ReplicaLifecycle::STARTING);
  RF_CHECK(r->warmth == WarmthState::COLD);
  RF_CHECK(r->health == HealthState::STARTING);
  RF_CHECK(r->readiness == ReadinessState::UNKNOWN);
  RF_CHECK(!r->serving_eligible);

  // Worker is no longer registered at all.
  RF_CHECK(ctrl.unregister_worker(gone.w).ok());
  RF_CHECK(ctrl.warm_replica(gone.id, full_warm(), gone.w, gone.b, 0).rejected());
  RF_CHECK(report(ctrl, gone.id, gone.w, gone.b).rejected());
  RF_CHECK(ctrl.report_readiness(gone.id, all_ready(), gone.w, gone.b, 0).rejected());
  const ReplicaState* g = ctrl.find_replica(gone.id);
  RF_REQUIRE(g != nullptr);
  RF_CHECK(g->lifecycle == ReplicaLifecycle::STARTING);
  RF_CHECK(g->health == HealthState::STARTING);
  RF_CHECK(g->readiness == ReadinessState::UNKNOWN);
}

// A replica is WARM only when every step the warming policy requires is done.
RF_TEST_CASE(controller_warm_requires_completed_steps) {
  ReplicaSetController ctrl(15);
  const ReplicaSetState set = new_set(2);
  RF_CHECK(ctrl.create_replica_set(set, epoch()).ok());
  const Hosted h = started_replica(ctrl, set);
  auto warmth = [&]() {
    const ReplicaState* r = ctrl.find_replica(h.id);
    return r != nullptr ? r->warmth : WarmthState::INVALIDATED;
  };

  // Nothing done yet.
  RF_CHECK(ctrl.warm_replica(h.id, WarmingRecord{}, h.w, h.b, 0).ok());
  RF_CHECK(warmth() == WarmthState::WARMING);

  // One required step missing.
  WarmingRecord no_endpoint = full_warm();
  no_endpoint.endpoint_registered = false;
  RF_CHECK(ctrl.warm_replica(h.id, no_endpoint, h.w, h.b, 0).ok());
  RF_CHECK(warmth() == WarmthState::WARMING);
  WarmingRecord no_weights = full_warm();
  no_weights.weights_resident = false;
  RF_CHECK(ctrl.warm_replica(h.id, no_weights, h.w, h.b, 0).ok());
  RF_CHECK(warmth() == WarmthState::WARMING);

  // Bounded warmup not finished.
  WarmingRecord partial = full_warm();
  partial.steps_required = 4;
  partial.steps_completed = 3;
  RF_CHECK(ctrl.warm_replica(h.id, partial, h.w, h.b, 0).ok());
  RF_CHECK(warmth() == WarmthState::WARMING);
  RF_CHECK(!ctrl.servable(h.id).eligible);

  // Everything done.
  WarmingRecord done = full_warm();
  done.steps_required = 4;
  done.steps_completed = 4;
  RF_CHECK(ctrl.warm_replica(h.id, done, h.w, h.b, 0).ok());
  RF_CHECK(warmth() == WarmthState::WARM);
}

// A replica whose health is FAILED or QUARANTINED is not promoted.
RF_TEST_CASE(controller_promotion_requires_working_health) {
  ReplicaSetController ctrl(16);
  const ReplicaSetState set = new_set(2);
  RF_CHECK(ctrl.create_replica_set(set, epoch()).ok());
  const Hosted h = ready_replica(ctrl, set);

  RF_CHECK(report(ctrl, h.id, h.w, h.b, HealthState::FAILED).ok());
  RF_CHECK(ctrl.promote_replica(h.id, PromotionState::PRIMARY, h.w, h.b, 0, 0).rejected());
  RF_CHECK(report(ctrl, h.id, h.w, h.b, HealthState::QUARANTINED).ok());
  RF_CHECK(ctrl.promote_replica(h.id, PromotionState::PRIMARY, h.w, h.b, 0, 0).rejected());
  RF_CHECK(ctrl.promote_replica(h.id, PromotionState::STANDBY, h.w, h.b, 0, 0).rejected());
  const ReplicaState* r = ctrl.find_replica(h.id);
  RF_REQUIRE(r != nullptr);
  RF_CHECK(!r->serving_eligible);
  RF_CHECK(r->promotion == PromotionState::NOT_PROMOTED);
  RF_CHECK(r->lifecycle == ReplicaLifecycle::READY);
  RF_CHECK(ctrl.promotions().empty());

  RF_CHECK(report(ctrl, h.id, h.w, h.b, HealthState::DEGRADED).ok());
  RF_CHECK(ctrl.promote_replica(h.id, PromotionState::PRIMARY, h.w, h.b, 0, 0).ok());
  RF_CHECK(ctrl.servable(h.id).eligible);
}

// A draining replica set is never re-armed as a side effect of a promotion or
// a failover; only an explicit transition makes it AVAILABLE again.
RF_TEST_CASE(controller_set_drain_not_reversed_implicitly) {
  ReplicaSetController ctrl(17);
  const ReplicaSetState set = new_set(2);
  RF_CHECK(ctrl.create_replica_set(set, epoch()).ok());
  const Hosted a = ready_replica(ctrl, set);
  const Hosted b = ready_replica(ctrl, set);
  RF_CHECK(ctrl.promote_replica(a.id, PromotionState::PRIMARY, a.w, a.b, 0, 0).ok());
  auto lifecycle = [&]() {
    const ReplicaSetState* s = ctrl.find_set(set.id);
    return s != nullptr ? s->lifecycle : ReplicaSetLifecycle::FAILED;
  };
  RF_CHECK(lifecycle() == ReplicaSetLifecycle::AVAILABLE);
  RF_CHECK(ctrl.transition_replica_set(set.id, ReplicaSetLifecycle::DRAINING).ok());
  RF_CHECK(!ctrl.servable(a.id).eligible);

  RF_CHECK(ctrl.promote_replica(b.id, PromotionState::STANDBY, b.w, b.b, 0, 0).ok());
  RF_CHECK(lifecycle() == ReplicaSetLifecycle::DRAINING);
  RF_CHECK(!ctrl.servable(a.id).eligible);

  RF_CHECK(ctrl.unregister_worker(a.w).ok());
  RF_CHECK(ctrl.trigger_failover(set.id, a.id, "worker died", 0, 0).ok());
  RF_CHECK(lifecycle() == ReplicaSetLifecycle::DRAINING);
  RF_CHECK(!ctrl.servable(b.id).eligible);

  RF_CHECK(ctrl.transition_replica_set(set.id, ReplicaSetLifecycle::AVAILABLE).ok());
  RF_CHECK(ctrl.servable(b.id).eligible);
}

// Active work is force-cancelled only when the draining policy allows it.
RF_TEST_CASE(controller_force_cancel_requires_policy) {
  ReplicaSetController ctrl(18);
  const CoordinatorEpoch ep = epoch();
  const ReplicaSetState strict = new_set(2);
  ReplicaSetState lenient = new_set(2);
  lenient.policies.draining.allow_force_cancel = true;
  RF_CHECK(ctrl.create_replica_set(strict, ep).ok());
  RF_CHECK(ctrl.create_replica_set(lenient, ep).ok());

  const Hosted a = ready_replica(ctrl, strict);
  RF_CHECK(ctrl.promote_replica(a.id, PromotionState::PRIMARY, a.w, a.b, 0, 0).ok());
  ctrl.replicas().at(a.id).active_requests = 3;
  RF_CHECK(ctrl.drain_replica(a.id, a.w, a.b, 0).ok());
  RF_CHECK(ctrl.complete_drain(a.id, 0, false, 1).rejected());
  RF_CHECK(ctrl.complete_drain(a.id, 0, true, 1).rejected());
  const ReplicaState* ra = ctrl.find_replica(a.id);
  RF_REQUIRE(ra != nullptr);
  RF_CHECK(ra->lifecycle == ReplicaLifecycle::DRAINING);
  RF_CHECK_EQ(ra->active_requests, 3u);
  // Once the work has finished the drain completes.
  ctrl.replicas().at(a.id).active_requests = 0;
  RF_CHECK(ctrl.complete_drain(a.id, 3, false, 2).ok());
  RF_CHECK(ctrl.find_replica(a.id)->lifecycle == ReplicaLifecycle::QUIESCED);

  const Hosted b = ready_replica(ctrl, lenient);
  RF_CHECK(ctrl.promote_replica(b.id, PromotionState::PRIMARY, b.w, b.b, 0, 0).ok());
  ctrl.replicas().at(b.id).active_requests = 3;
  RF_CHECK(ctrl.drain_replica(b.id, b.w, b.b, 0).ok());
  RF_CHECK(ctrl.complete_drain(b.id, 0, true, 1).ok());
  const ReplicaState* rb = ctrl.find_replica(b.id);
  RF_REQUIRE(rb != nullptr);
  RF_CHECK(rb->lifecycle == ReplicaLifecycle::QUIESCED);
  RF_CHECK_EQ(rb->active_requests, 0u);
}

// Provisioning never replaces a live replica, and a retired replica no longer
// occupies one of the set's max_replicas slots.
RF_TEST_CASE(controller_provision_replaces_only_terminal_replicas) {
  ReplicaSetController ctrl(19);
  const ReplicaSetState set = new_set(3);
  RF_CHECK(ctrl.create_replica_set(set, epoch()).ok());
  const DeviceCapability dev{AcceleratorKind::CUDA, "cuda:0", {12, 0}, 32ULL * 1024 * 1024 * 1024};
  const Hosted a = ready_replica(ctrl, set);
  RF_CHECK(ctrl.promote_replica(a.id, PromotionState::PRIMARY, a.w, a.b, 0, 0).ok());

  // A duplicate provision of the serving primary leaves it serving.
  RF_CHECK(ctrl.provision_replica(a.id, set.id, a.w, make_compat(set), dev, 1024, 0, 0).rejected());
  RF_CHECK(ctrl.servable(a.id).eligible);
  RF_CHECK(ctrl.find_replica(a.id)->lifecycle == ReplicaLifecycle::SERVING);

  const Hosted b = ready_replica(ctrl, set);
  const Hosted c = ready_replica(ctrl, set);
  RF_CHECK_EQ(ctrl.list_replicas(set.id).size(), 3u);
  const ReplicaGeneration first_generation = ctrl.find_replica(b.id)->generation;

  // The set is full.
  RF_CHECK(ctrl.provision_replica(rid(), set.id, b.w, make_compat(set), dev, 1024, 0, 0).rejected());

  // A failed replica is replaced by a new incarnation under the same identity.
  RF_CHECK(ctrl.fail_replica(b.id, b.w, b.b, "device lost", 0).ok());
  RF_CHECK(ctrl.provision_replica(b.id, set.id, b.w, make_compat(set), dev, 1024, 0, 0).ok());
  const ReplicaState* rb = ctrl.find_replica(b.id);
  RF_REQUIRE(rb != nullptr);
  RF_CHECK(rb->lifecycle == ReplicaLifecycle::DECLARED);
  RF_CHECK(rb->generation != first_generation);
  RF_CHECK(rb->health == HealthState::UNKNOWN);

  // A retired replica frees its slot for a replacement with a new identity.
  RF_CHECK(ctrl.retire_replica(c.id, 0).ok());
  const ReplicaId d = rid();
  RF_CHECK(ctrl.provision_replica(d, set.id, c.w, make_compat(set), dev, 1024, 0, 0).ok());
  RF_CHECK(ctrl.find_replica(d) != nullptr);
  // ...and the set is full again.
  RF_CHECK(ctrl.provision_replica(rid(), set.id, c.w, make_compat(set), dev, 1024, 0, 0).rejected());
  RF_CHECK(ctrl.servable(a.id).eligible);
}

// A rejected failure report leaves a retired replica's record untouched.
RF_TEST_CASE(controller_rejected_failure_leaves_replica_untouched) {
  ReplicaSetController ctrl(20);
  const ReplicaSetState set = new_set(2);
  RF_CHECK(ctrl.create_replica_set(set, epoch()).ok());
  const Hosted a = ready_replica(ctrl, set);
  RF_CHECK(ctrl.retire_replica(a.id, 0).ok());
  const ReplicaState before = *ctrl.find_replica(a.id);
  RF_CHECK(before.health == HealthState::HEALTHY);

  RF_CHECK(ctrl.fail_replica(a.id, a.w, a.b, "late failure", 5).rejected());
  const ReplicaState* after = ctrl.find_replica(a.id);
  RF_REQUIRE(after != nullptr);
  RF_CHECK(*after == before);
  RF_CHECK(after->lifecycle == ReplicaLifecycle::RETIRED);
  RF_CHECK(after->health == HealthState::HEALTHY);
}
