#!/usr/bin/env bash
# Fetches the O_o_Model package the test suites load and gives its folder to the next steps of the job as
# OOR_READER_DIST (wasm/test/helpers/runtime.mjs reads it).
#
# The package is the archive of a release of the O_o_Model repository, named by READER_REPOSITORY (owner/name),
# READER_TAG and READER_ARCHIVE. That repository is public: the archive is read without credentials.
set -euo pipefail

: "${READER_REPOSITORY:?the repository of the reader, as owner/name}"
: "${READER_TAG:?the tag of the release of the reader}"
: "${READER_ARCHIVE:?the name of the archive in that release}"

server="${GITHUB_SERVER_URL:-https://github.com}"
work="${RUNNER_TEMP:-$(mktemp -d)}"
archive="$work/$READER_ARCHIVE"
folder="$work/oo-model"

release_url="$server/$READER_REPOSITORY/releases/download/$READER_TAG"
curl --fail --silent --show-error --location --retry 3 --output "$archive" "$release_url/$READER_ARCHIVE"

rm -rf "$folder"
mkdir -p "$folder"
unzip -q "$archive" -d "$folder"
test -f "$folder/index.js"

echo "O_o_Model package: $(grep '"version"' "$folder/package.json" | tr -d ' ,') from $READER_REPOSITORY@$READER_TAG"
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "OOR_READER_DIST=$folder" >> "$GITHUB_ENV"
else
    echo "OOR_READER_DIST=$folder"
fi
