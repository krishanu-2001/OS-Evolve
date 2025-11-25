// Head-of-Line (HoL) blocking experiment for ghOSt CFS scheduler.
// This workload is a synchronized batch of ghOSt-managed threads where 
// exactly one “slow” worker hogs CPU time while the rest are lightweight, 
// letting you observe head-of-line blocking effects.

#include <atomic>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/substitute.h"
#include "lib/base.h"
#include "lib/ghost.h"
#include "schedulers/test/cfs_scheduler.h" // these are our Shinka's agents

ABSL_FLAG(bool, create_enclave_and_agent, false,
          "If true, spawns an enclave and a CFS agent for experiments. ");
ABSL_FLAG(std::string, ghost_cpus, "1-5",
          "List of cpu IDs to create an enclave with (effective when the "
          "enclave/agent is created by this binary).");

ABSL_FLAG(int, hol_threads, 16, "Total threads in HoL experiment");
ABSL_FLAG(int, hol_slow_index, 0, "Index of the slow thread [0..N-1]");
ABSL_FLAG(int, hol_slow_ms, 100, "Slow thread busy time in milliseconds");
ABSL_FLAG(int, hol_fast_ms, 5, "Fast thread busy time in milliseconds");
ABSL_FLAG(std::string, hol_metrics,
          "metrics/cfs_hol.csv",
          "CSV path to write HoL metrics");

namespace ghost {
namespace {

struct WorkloadSpec {
  int id;
  bool is_slow;
  absl::Duration work_duration;
};

struct Sample {
  int id;
  bool is_slow;
  int start_cpu;
  double start_ms;   // since release
  double finish_ms;  // since release
  double runtime_ms; // finish - start
};


std::vector<WorkloadSpec> BuildWorkloadSpecs(int num_threads, int slow_idx,
                                             absl::Duration slow_d,
                                             absl::Duration fast_d) {
  std::vector<WorkloadSpec> specs;
  specs.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    const bool is_slow = (i == slow_idx);
    specs.push_back({i, is_slow, is_slow ? slow_d : fast_d});
  }
  return specs;
}

// Busy-wait while attempting to discount preemption effects.
// The short loop around MonotonicNow() is copied from tests that try to
// avoid counting time when the thread was off-CPU, making runtime closer to
// actual on-CPU time.
static void SpinFor(absl::Duration d) {
  while (d > absl::ZeroDuration()) {
    absl::Time a = MonotonicNow();
    absl::Time b = a;
    for (int i = 0; i < 150; i++) b = MonotonicNow();
    absl::Duration t = b - a;
    if (t < absl::Microseconds(200)) d -= t;
  }
}

// PersistMetrics writes a CSV file with one row per thread. Columns:
//  id,is_slow,start_cpu,start_ms,finish_ms,runtime_ms
// Times are relative to the common release moment so they are comparable
// across threads.
void PersistMetrics(const std::string& path, const std::vector<Sample>& s) {
  // Ensure directory exists (best-effort).
#if __has_include(<filesystem>)
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::create_directories(fs::path(path).parent_path(), ec);
#endif
  std::ofstream out(path);
  out << "id,is_slow,start_cpu,start_ms,finish_ms,runtime_ms\n";
  for (const auto& x : s) {
    out << x.id << "," << (x.is_slow ? 1 : 0) << "," << x.start_cpu << ","
        << x.start_ms << "," << x.finish_ms << "," << x.runtime_ms << "\n";
  }
}

// Scheduling: run the specs under ghOSt CFS and collect metrics.
//
// Protocol
// - Each thread waits on a Notification; the main path flips a release flag
//   and notifies once all threads are ready. Using a shared release timestamp
//   lets us compute per-thread queueing delay (start_ms) relative to release.
// - We snapshot the CPU at start for observability (was placement skewed?).
// - Samples are pushed under a mutex; vectors are not thread-safe to append.
std::vector<Sample> RunHeadOfLine(const std::vector<WorkloadSpec>& specs) {
  std::atomic<bool> release{false};
  Notification ready;
  std::atomic<int> at_barrier{0};

  absl::Time release_time;
  std::vector<Sample> samples;
  samples.reserve(specs.size());

  std::vector<std::unique_ptr<GhostThread>> threads;
  threads.reserve(specs.size());

  for (const auto& w : specs) {
    threads.emplace_back(std::make_unique<GhostThread>(
        GhostThread::KernelScheduler::kGhost, [&release, &at_barrier, &ready,
                                               &release_time, &samples, w] {
          at_barrier.fetch_add(1, std::memory_order_relaxed);
          ready.WaitForNotification();

          // All threads observe a common release moment.
          while (!release.load(std::memory_order_acquire)) {}
          const absl::Time t0 = absl::Now();
          const int start_cpu = sched_getcpu();  // placement at start

          SpinFor(w.work_duration);
          const absl::Time t1 = absl::Now();

          Sample s;
          s.id = w.id;
          s.is_slow = w.is_slow;
          s.start_cpu = start_cpu;
          s.start_ms = absl::ToDoubleMilliseconds(t0 - release_time);
          s.finish_ms = absl::ToDoubleMilliseconds(t1 - release_time);
          s.runtime_ms = absl::ToDoubleMilliseconds(t1 - t0);

          // Serialize push into vector (it’s shared across threads).
          static absl::Mutex mu(absl::kConstInit);
          absl::MutexLock lock(&mu);
          samples.push_back(s);
        }));
  }

  // Wait for all threads to be ready in the barrier.
  while (at_barrier.load(std::memory_order_relaxed) <
         static_cast<int>(specs.size())) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  release_time = absl::Now();
  release.store(true, std::memory_order_release);
  ready.Notify();

  for (auto& t : threads) t->Join();

  return samples;
}

// PrintSummary logs a quick signal to STDOUT to detect HoL symptoms without
// opening the CSV: if fast start_ms p95 is high, fast tasks likely queued
// behind the slow one.
void PrintSummary(const std::vector<Sample>& s) {
  // Compute fast thread queueing delay percentiles.
  std::vector<double> fast_starts;
  fast_starts.reserve(s.size());
  for (const auto& x : s) if (!x.is_slow) fast_starts.push_back(x.start_ms);
  std::sort(fast_starts.begin(), fast_starts.end());
  auto pct = [&](double p) {
    if (fast_starts.empty()) return 0.0;
    size_t idx = static_cast<size_t>(p * (fast_starts.size() - 1));
    return fast_starts[idx];
  };
  double p50 = pct(0.50), p95 = pct(0.95);

  std::cout << absl::Substitute(
                   "HoL summary: fast start_ms p50=$0 p95=$1 (ms)", p50, p95)
            << std::endl;
}

// HeadOfLineCfs wires flags -> data -> run -> summary -> persistence.
// Keeping this orchestration small makes it easy to compose in other tests.
void HeadOfLineCfs() {
  const int N = absl::GetFlag(FLAGS_hol_threads);
  const int slow_idx = absl::GetFlag(FLAGS_hol_slow_index);
  const absl::Duration slow_d = absl::Milliseconds(absl::GetFlag(FLAGS_hol_slow_ms));
  const absl::Duration fast_d = absl::Milliseconds(absl::GetFlag(FLAGS_hol_fast_ms));
  const std::string csv = absl::GetFlag(FLAGS_hol_metrics);

  std::cout << absl::Substitute(
                   "Starting CFS HoL: N=$0 slow_idx=$1 slow_ms=$2 fast_ms=$3",
                   N, slow_idx, absl::ToInt64Milliseconds(slow_d),
                   absl::ToInt64Milliseconds(fast_d))
            << std::endl;

  auto specs = BuildWorkloadSpecs(N, slow_idx, slow_d, fast_d);
  auto samples = RunHeadOfLine(specs);
  PrintSummary(samples);
  PersistMetrics(csv, samples);
  std::cout << "Metrics written to " << csv << std::endl;
}

}  // namespace
}  // namespace ghost

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  std::unique_ptr<ghost::AgentProcess<ghost::FullCfsAgent<ghost::LocalEnclave>,
                                      ghost::CfsConfig>>
      ap;
  ghost::CpuList cpus = ghost::MachineTopology()->all_cpus();
  if (absl::GetFlag(FLAGS_create_enclave_and_agent)) {
    ghost::Topology* topology = ghost::MachineTopology();
    cpus = topology->ParseCpuStr(absl::GetFlag(FLAGS_ghost_cpus));
    ghost::CfsConfig config(topology, cpus);
    ap = std::make_unique<ghost::AgentProcess<
        ghost::FullCfsAgent<ghost::LocalEnclave>, ghost::CfsConfig>>(config);
  }

  ghost::HeadOfLineCfs();
  return 0;
}
