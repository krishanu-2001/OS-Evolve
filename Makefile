# -------------------------------
# Install ghost-userspace assets
# -------------------------------
GHOST_USERSPACE_DIR ?= ghost-userspace
GHOST_ENCLAVES_DIR ?= /sys/fs/ghost

install-ghost-userspace:
	bash ghost-assets/install-ghost-userspace.sh $(GHOST_USERSPACE_DIR)
	cp ghost-assets/BUILD $(GHOST_USERSPACE_DIR)/BUILD 
	echo "7.1.1" > $(GHOST_USERSPACE_DIR)/.bazelversion
	cp ghost-assets/.bazelversion $(GHOST_USERSPACE_DIR)/.bazelversion
	cp ghost-assets/.bazelrc $(GHOST_USERSPACE_DIR)/.bazelrc
	cp ghost-assets/cfs_hol_test.cc $(GHOST_USERSPACE_DIR)/tests/cfs_hol_test.cc
	cp ghost-assets/cfs_sysbench_test.cc $(GHOST_USERSPACE_DIR)/tests/cfs_sysbench_test.cc
	cp -r ghost-assets/test $(GHOST_USERSPACE_DIR)/schedulers/test
	cd $(GHOST_USERSPACE_DIR) && sudo bazel build -c opt agent_cfs_test
	cd $(GHOST_USERSPACE_DIR) && sudo bazel build -c opt cfs_hol_test
	cd $(GHOST_USERSPACE_DIR) && sudo ./bazel-bin/cfs_hol_test --create_enclave_and_agent --ghost_cpus=1-5

rocks-db-test:
	bash ghost-assets/install-ghost-userspace.sh $(GHOST_USERSPACE_DIR)
	cd $(GHOST_USERSPACE_DIR) && sudo bazel build -c opt rocksdb
	cd $(GHOST_USERSPACE_DIR) && sudo ./bazel-bin/rocksdb --rocksdb_db_path=/tmp/rocksdb_test \
		--experiment_duration=60s --throughput=2000.0 --range_query_ratio=0.5 \
		--discard_duration=5s --print_range=true --range_duration=1000us --num_workers=10 --worker_cpus=12-21 & \
	# TID_PID=$$!; \
	# sleep 10; \
	# echo "Process PID: $$TID_PID"; \
	# echo "Monitoring worker threads with perf..."; \
	# bash scripts/perf_by_role.sh $$TID_PID worker -e cycles,instructions,cache-misses,cache-references,LLC-loads,LLC-load-misses || true; \
	# wait $$TID_PID

other-test:
	bash ghost-assets/install-ghost-userspace.sh $(GHOST_USERSPACE_DIR)
	cd $(GHOST_USERSPACE_DIR) && sudo bazel build -c opt global_scalability
	cd $(GHOST_USERSPACE_DIR) && sudo ./bazel-bin/global_scalability none --enclave=1 --total_loops=1000 &> /tmp/other_test.log &


clean_enclaves:
	for i in $$(seq 1 10); do \
		echo destroy | sudo tee $(GHOST_ENCLAVES_DIR)/enclave_$${i}/ctl >/dev/null || true; \
	done
	