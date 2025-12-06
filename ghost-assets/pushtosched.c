#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#define SCHED_GHOST 18

static const char *enclave_path = "/sys/fs/ghost/enclave_1/ctl";
static char *progname;

static void usage(int rc) {
  fprintf(stderr, "Usage: %s <policy>\n", progname);
  fprintf(stderr, "To push tasks into ghOSt:\n");
  fprintf(stderr, "    $ cat /dev/cgroup/cpu/mine/tasks | %s 18\n", progname);
  fprintf(stderr, "To push tasks into CFS\n");
  fprintf(stderr, "    $ cat /dev/cgroup/cpu/your/tasks | %s 0\n", progname);
  exit(rc);
}

// For various glibc reasons, this isn't available in glibc/grte.  Including
// uapi/sched.h or sched/types.h will run into conflicts on sched_param.
struct sched_attr {
  uint32_t size;
  uint32_t sched_policy;
  uint64_t sched_flags;
  int32_t sched_nice;
  uint32_t sched_priority;  // overloaded for is/is not an agent
  uint64_t sched_runtime;   // overloaded for enclave ctl fd
  uint64_t sched_deadline;
  uint64_t sched_period;
};
#define SCHED_FLAG_RESET_ON_FORK 0x01

int sched_enter_ghost(pid_t pid, int enclave_fd) {
  // Enter ghOSt sched class.
  struct sched_attr attr = {
      .size = sizeof(sched_attr),
      .sched_policy = SCHED_GHOST,
      .sched_priority = 0,  // GHOST_SCHED_TASK_PRIO
      .sched_runtime = static_cast<uint64_t>(enclave_fd),
  };
  return syscall(__NR_sched_setattr, pid, &attr, /*flags=*/0);
}

int sched_enter_other(pid_t pid, int policy) {
  struct sched_param param = { 0 };
  return sched_setscheduler(pid, policy, &param);
}

int main(int argc, char *argv[])
{
  pid_t pid;
  int policy, enclave_fd = -1;

  progname = basename(argv[0]);

  if (argc != 2)
    usage(1);

  policy = atoi(argv[1]);
  if (policy == SCHED_GHOST) {
    enclave_fd = open(enclave_path, O_RDWR);
    if (enclave_fd < 0) {
      fprintf(stderr, "open(%s): %s\n", enclave_path, strerror(errno));
      exit(1);
    }
  }

  fprintf(stderr, "Setting scheduling policy to %d\n", policy);

  while (fscanf(stdin, "%d\n", &pid) != EOF) {
    if (sched_getscheduler(pid) == policy)
      continue;

    int ret;
    fprintf(stderr, "pid: %d\n", pid);
    if (policy == SCHED_GHOST)
      ret = sched_enter_ghost(pid, enclave_fd);
    else
      ret = sched_enter_other(pid, policy);

    // Trust but verify.
    if (!ret) {
      int actual = sched_getscheduler(pid);
      if (actual != policy) {
        fprintf(stderr, "scheduling policy of %d: want %d, got %d: %s\n",
                pid, policy, actual, strerror(errno));
      }
    } else {
      fprintf(stderr, "setscheduler(%d) failed: %s\n", pid, strerror(errno));
      exit(1);
    }
  }

  exit(0);
}
