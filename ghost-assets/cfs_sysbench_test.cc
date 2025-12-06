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
#include <sys/syscall.h>
#include <sys/wait.h>
#include <dirent.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <cstdint>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "lib/base.h"
#include "lib/ghost.h"
#include "schedulers/test/cfs_scheduler.h"

ABSL_FLAG(bool, create_enclave_and_agent, false,
          "If true, spawns an enclave and a CFS agent for experiments.");
ABSL_FLAG(std::string, ghost_cpus, "1-5",
          "List of cpu IDs to create an enclave with.");

// ---------------------------------------------------------
// Helper: Move a process into ghOSt enclave
// Uses the GhostHelper API with a valid Enclave Directory FD
// ---------------------------------------------------------
int MoveProcessToGhost(pid_t pid, int enclave_fd) {
  // Use the official helper. 
  // enclave_fd must be the directory FD for /sys/fs/ghost/enclave_X
  return ghost::GhostHelper()->SchedTaskEnterGhost(pid, enclave_fd);
}

// ---------------------------------------------------------
// Helper: Get all thread PIDs for a process
// ---------------------------------------------------------
std::vector<pid_t> GetThreadPids(pid_t pid) {
  std::vector<pid_t> tids;
  std::string task_dir = absl::Substitute("/proc/$0/task", pid);
  
  DIR* dir = opendir(task_dir.c_str());
  if (!dir) {
    perror("opendir failed");
    return tids;
  }
  
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_name[0] == '.') continue;
    pid_t tid = atoi(entry->d_name);
    if (tid > 0) tids.push_back(tid);
  }
  closedir(dir);
  return tids;
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
// 1. Fork and exec sysbench
// 2. Move sysbench process and threads into ghOSt
// 3. Capture output and parse metrics
// 4. Save CSV → metrics/cfs_sysbench.csv
// ---------------------------------------------------------
void sysbenchCfs(int enclave_fd) {
  std::cout << "[sysbenchCfs] Starting sysbench under ghOSt...\n";
  const std::string ghost_cpus = absl::GetFlag(FLAGS_ghost_cpus);

  // Create metrics directory if missing
  // (void) cast prevents "ignoring return value" warning
  (void)system("mkdir -p metrics");

  // Wait a moment to ensure enclave is fully initialized
  absl::SleepFor(absl::Milliseconds(500));

  // Create a pipe to capture sysbench output
  int pipefd[2];
  if (pipe(pipefd) == -1) {
    perror("pipe failed");
    return;
  }

  // Fork and exec sysbench
  pid_t sysbench_pid = fork();
  if (sysbench_pid == 0) {
    // Child: redirect stdout and stderr to pipe, then exec sysbench
    close(pipefd[0]);  // Close read end
    dup2(pipefd[1], STDOUT_FILENO);
    dup2(pipefd[1], STDERR_FILENO);
    close(pipefd[1]);
    
    // NOTE: Increased time to 30s to ensure valid gathering period
    execlp("sysbench", "sysbench", "cpu",
           "--threads=1",
           "--cpu-max-prime=20000",
           "run", nullptr);
    perror("execlp failed");
    exit(1);
  } else if (sysbench_pid < 0) {
    std::cerr << "[sysbenchCfs] fork failed: " << strerror(errno) << std::endl;
    close(pipefd[0]);
    close(pipefd[1]);
    return;
  }

  // Parent: close write end of pipe
  close(pipefd[1]);

  // Wait a moment for sysbench to start and create threads
  absl::SleepFor(absl::Milliseconds(200));

  // Get all thread PIDs for sysbench process
  std::vector<pid_t> thread_pids = GetThreadPids(sysbench_pid);
  
  // Move main process and all threads into ghOSt
  std::cout << "[sysbenchCfs] Moving sysbench (pid=" << sysbench_pid 
            << ") and " << thread_pids.size() << " threads into ghOSt..." << std::endl;
  
  if (MoveProcessToGhost(sysbench_pid, enclave_fd) != 0) {
    std::cerr << "[sysbenchCfs] ERROR: Failed to move sysbench process to ghOSt: " 
              << strerror(errno) << std::endl;
  } else {
    std::cout << "[sysbenchCfs] SUCCESS: Moved main process." << std::endl;
  }
  
  int moved_count = 0;
  for (pid_t tid : thread_pids) {
    if (tid != sysbench_pid) {  // Don't move main process twice
      if (MoveProcessToGhost(tid, enclave_fd) != 0) {
        std::cerr << "[sysbenchCfs] Warning: Failed to move thread " << tid 
                  << " to ghOSt: " << strerror(errno) << std::endl;
      } else {
        moved_count++;
      }
    }
  }
  std::cout << "[sysbenchCfs] Moved " << moved_count << " threads to ghOSt" << std::endl;

  // Read sysbench output from pipe
  std::string output;
  char buffer[4096];
  ssize_t n;
  while ((n = read(pipefd[0], buffer, sizeof(buffer) - 1)) > 0) {
    buffer[n] = '\0';
    output += buffer;
  }
  close(pipefd[0]);

  // Wait for sysbench to complete
  int status;
  waitpid(sysbench_pid, &status, 0);
  
  if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
    std::cerr << "[sysbenchCfs] Warning: sysbench exited with status " 
              << WEXITSTATUS(status) << std::endl;
  }

  // Save raw output for debugging
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
    std::cerr << "[sysbenchCfs] Failed to open CSV file: " << csv_path << "\n";
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
  std::cout << "[sysbenchCfs] Events/sec: " << metrics.events_per_sec << std::endl;
}

// ---------------------------------------------------------
int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  std::unique_ptr<
      ghost::AgentProcess<ghost::FullCfsAgent<ghost::LocalEnclave>,
                          ghost::CfsConfig>>
      ap;

  // Initialize enclave_fd to -1 (invalid)
  int enclave_fd = -1;

  if (absl::GetFlag(FLAGS_create_enclave_and_agent)) {
    ghost::Topology* topology = ghost::MachineTopology();
    ghost::CpuList cpus = topology->ParseCpuStr(absl::GetFlag(FLAGS_ghost_cpus));
    ghost::CfsConfig config(topology, cpus);

    std::cout << "Creating enclave + CFS ghOSt agent on CPUs "
              << absl::GetFlag(FLAGS_ghost_cpus) << "...\n";

    ap = std::make_unique<
        ghost::AgentProcess<ghost::FullCfsAgent<ghost::LocalEnclave>,
                            ghost::CfsConfig>>(config);
    
    // FIX: Directly open the enclave directory instead of using the API
    // The AgentProcess above will create 'enclave_1' by default.
    // O_PATH is sufficient for reference, but O_RDONLY|O_DIRECTORY is standard.
    enclave_fd = open("/sys/fs/ghost/enclave_1", O_RDONLY | O_DIRECTORY);
    
    if (enclave_fd < 0) {
        // Fallback: In some race conditions, it might be enclave_2, though rare in tests.
        // We will stick to complaining if enclave_1 isn't there.
        std::cerr << "WARNING: Could not open /sys/fs/ghost/enclave_1. " 
                  << "Error: " << strerror(errno) << std::endl;
    }
  }

  // Check if we have a valid FD
  if (enclave_fd < 0) {
      std::cerr << "Error: No valid enclave file descriptor found.\n";
      return 1;
  }

  std::cout << "[main] Using enclave FD: " << enclave_fd << std::endl;

  // Pass the valid FD to your test function
  sysbenchCfs(enclave_fd);
  
  // Clean up FD
  if (enclave_fd >= 0) close(enclave_fd);

  return 0;
}
