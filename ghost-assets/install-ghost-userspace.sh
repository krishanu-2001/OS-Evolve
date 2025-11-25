#!/usr/bin/env bash
# Simple helper to clone the upstream ghOSt userspace repository.

set -euo pipefail

REPO_URL="https://github.com/google/ghost-userspace.git"
TARGET_DIR="${1:-ghost-userspace}"

if [[ -d "${TARGET_DIR}/.git" ]]; then
  echo "Repository already exists at ${TARGET_DIR}"
  exit 0
fi

echo "Cloning ${REPO_URL} into ${TARGET_DIR} ..."
git clone "${REPO_URL}" "${TARGET_DIR}"
echo "Done."
