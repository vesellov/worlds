#!/usr/bin/env bash

set -euo pipefail

INSTALLER_DIR="."
P4A_DIR="${INSTALLER_DIR}/cache/python-for-android"

if [[ -e "${P4A_DIR}" && ! -d "${P4A_DIR}/.git" ]]; then
    echo "Error: ${P4A_DIR} exists but is not a Git checkout." >&2
    exit 1
fi

if [[ ! -d "${P4A_DIR}/.git" ]]; then
    mkdir -p "${INSTALLER_DIR}/cache"
    git clone --branch master https://github.com/kivy/python-for-android.git "${P4A_DIR}"
    cd "${P4A_DIR}"
    git apply --verbose --check "../../src/python-for-android-local-changes.patch"
fi

echo "python-for-android is ready in ${P4A_DIR}"
