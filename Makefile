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
	cp -r ghost-assets/test $(GHOST_USERSPACE_DIR)/schedulers/test
	cd $(GHOST_USERSPACE_DIR) && sudo bazel build -c opt agent_cfs_test
	cd $(GHOST_USERSPACE_DIR) && sudo bazel build -c opt cfs_sysbench_test
	cd $(GHOST_USERSPACE_DIR) && sudo ./bazel-bin/cfs_sysbench_test --create_enclave_and_agent --ghost_cpus=1-5

clean_enclaves:
	for i in $$(seq 1 10); do \
		echo destroy | sudo tee $(GHOST_ENCLAVES_DIR)/enclave_$${i}/ctl >/dev/null || true; \
	done
	