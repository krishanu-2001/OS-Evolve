// This workload runs the sysbench benchmark under ghOSt CFS scheduler
// and collects performance metrics.

#include <atomic>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdio>
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/substitute.h"
#include "lib/base.h"
#include "lib/ghost.h"
#include "schedulers/test/cfs_scheduler.h"

ABSL_FLAG(bool, create_enclave_and_agent, false,
          "If true, spawns an enclave and a CFS agent for experiments.");
ABSL_FLAG(std::string, ghost_cpus, "1-5",
          "List of cpu IDs to create an enclave with.");

// ---------------------------------------------------------
// Helper: Run a shell command and capture all stdout lines.
// ---------------------------------------------------------
std::string RunCommandCaptureOutput(const std::string& cmd) {
  std::array<char, 4096> buffer{};
  std::string result;

  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    perror("popen failed");
    return "";
  }

  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    result += buffer.data();
  }

  pclose(pipe);
  return result;
}

// ---------------------------------------------------------
// Helper: Parse sysbench output text into CSV fields.
// ---------------------------------------------------------
struct SysbenchMetrics {
  double total_time_sec = 0;
  double events_per_sec = 0;
  double total_events = 0;
  double min_latency_ms = 0;
  double avg_latency_ms = 0;
  double max_latency_ms = 0;
};

SysbenchMetrics ParseSysbenchOutput(const std::string& output) {
  SysbenchMetrics m;
  std::istringstream iss(output);
  std::string line;

  while (std::getline(iss, line)) {
    if (line.find("total time:") != std::string::npos) {
      sscanf(line.c_str(), "    total time: %lf", &m.total_time_sec);
    }
    if (line.find("events per second:") != std::string::npos) {
      sscanf(line.c_str(), "    events per second: %lf", &m.events_per_sec);
    }
    if (line.find("total number of events:") != std::string::npos) {
      sscanf(line.c_str(), "    total number of events: %lf", &m.total_events);
    }
    if (line.find("min:") == 0) {
      sscanf(line.c_str(), "min: %lf", &m.min_latency_ms);
    }
    if (line.find("avg:") == 0) {
      sscanf(line.c_str(), "avg: %lf", &m.avg_latency_ms);
    }
    if (line.find("max:") == 0) {
      sscanf(line.c_str(), "max: %lf", &m.max_latency_ms);
    }
  }

  return m;
}

// ---------------------------------------------------------
// 1. Run sysbench
// 2. Parse metrics
// 3. Save CSV → metrics/cfs_sysbench.csv
// ---------------------------------------------------------
void sysbenchCfs() {
  std::cout << "[sysbenchCfs] Starting sysbench under ghOSt...\n";
  const std::string ghost_cpus = absl::GetFlag(FLAGS_ghost_cpus);

  // Create metrics directory if missing
  system("mkdir -p metrics");

  // Full sysbench command
  const std::string cmd = absl::Substitute(
      "./tests/run_sysbench_ghost.sh 1 $0 2>&1", ghost_cpus);

  std::cout << "[sysbenchCfs] Running command: " << cmd << "\n";

  // Run sysbench and capture stdout
  std::string output = RunCommandCaptureOutput(cmd);

  // Save raw stdout for debugging
  {
    std::ofstream raw("metrics/cfs_sysbench_raw.txt");
    raw << output;
  }

  // Parse metrics from sysbench plaintext
  SysbenchMetrics metrics = ParseSysbenchOutput(output);

  // Open CSV file
  const std::string csv_path = "metrics/cfs_sysbench.csv";
  bool exists = (access(csv_path.c_str(), F_OK) == 0);

  std::ofstream csv(csv_path, std::ios::app);
  if (!csv.is_open()) {
    std::cerr << "Failed to open CSV file: " << csv_path << "\n";
    return;
  }

  // Write header only once
  if (!exists) {
    csv << "total_time_sec,events_per_sec,total_events,min_latency_ms,"
           "avg_latency_ms,max_latency_ms\n";
  }

  // Write metrics
  csv << metrics.total_time_sec << ","
      << metrics.events_per_sec << ","
      << metrics.total_events << ","
      << metrics.min_latency_ms << ","
      << metrics.avg_latency_ms << ","
      << metrics.max_latency_ms << "\n";

  std::cout << "[sysbenchCfs] Metrics saved to " << csv_path << "\n";
}

// ---------------------------------------------------------
int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  std::unique_ptr<
      ghost::AgentProcess<ghost::FullCfsAgent<ghost::LocalEnclave>,
                          ghost::CfsConfig>>
      ap;

  ghost::CpuList cpus = ghost::MachineTopology()->all_cpus();

  if (absl::GetFlag(FLAGS_create_enclave_and_agent)) {
    ghost::Topology* topology = ghost::MachineTopology();
    cpus = topology->ParseCpuStr(absl::GetFlag(FLAGS_ghost_cpus));
    ghost::CfsConfig config(topology, cpus);

    std::cout << "Creating enclave + CFS ghOSt agent on CPUs "
              << absl::GetFlag(FLAGS_ghost_cpus) << "...\n";

    ap = std::make_unique<
        ghost::AgentProcess<ghost::FullCfsAgent<ghost::LocalEnclave>,
                            ghost::CfsConfig>>(config);
  }

  sysbenchCfs();
  return 0;
}
