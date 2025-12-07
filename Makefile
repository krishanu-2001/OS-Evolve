# -------------------------------
# Install ghost-userspace assets
# -------------------------------
GHOST_USERSPACE_DIR ?= ghost-userspace
GHOST_USERSPACE_DIR ?= 

install-ghost-userspace:
	bash ghost-assets/install-ghost-userspace.sh $(GHOST_USERSPACE_DIR)
	cp ghost-assets/BUILD $(GHOST_USERSPACE_DIR)/BUILD 
	echo "7.1.1" > $(GHOST_USERSPACE_DIR)/.bazelversion
	cp ghost-assets/.bazelversion $(GHOST_USERSPACE_DIR)/.bazelversion
	cp ghost-assets/.bazelrc $(GHOST_USERSPACE_DIR)/.bazelrc
	cp ghost-assets/cfs_hol_test.cc $(GHOST_USERSPACE_DIR)/tests/cfs_hol_test.cc
	cp ghost-assets/cfs_sysbench_test.cc $(GHOST_USERSPACE_DIR)/tests/cfs_sysbench_test.cc
	cp ghost-assets/run_sysbench_ghost.sh $(GHOST_USERSPACE_DIR)/tests/run_sysbench_ghost.sh
	chmod +x $(GHOST_USERSPACE_DIR)/tests/run_sysbench_ghost.sh
	cp -r ghost-assets/test $(GHOST_USERSPACE_DIR)/schedulers/test
	cd $(GHOST_USERSPACE_DIR) && sudo bazel build -c opt agent_cfs_test
	cd $(GHOST_USERSPACE_DIR) && sudo bazel build -c opt cfs_hol_test
	cd $(GHOST_USERSPACE_DIR) && sudo ./bazel-bin/cfs_hol_test --create_enclave_and_agent --hol_threads=128 --hol_fast_ms=5 --hol_slow_index=5 --hol_slow_ms=150 --ghost_cpus=1-5 --hol_metrics='../results/make_run/cfs_hol.csv'

clean_enclaves:
	for i in $$(seq 1 10); do \
		echo destroy | sudo tee $(GHOST_ENCLAVES_DIR)/enclave_$${i}/ctl >/dev/null || true; \
	done
	