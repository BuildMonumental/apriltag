#!/usr/bin/env bash
# Profile-guided-optimization build: instrument, run the corpus to collect
# a profile, rebuild with it. Produces build-pgo/. Output is bit-identical
# to the plain build; only code layout and branch decisions change.
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR=${BUILD_DIR:-build-pgo}
IMAGE_DIR=${IMAGE_DIR:-vide_images}
PROFILE_DIR=${PROFILE_DIR:-$PWD/$BUILD_DIR/pgo-profile}

rm -rf "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"

cmake -S . -B "$BUILD_DIR" -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=native -fprofile-generate=$PROFILE_DIR" > /dev/null
cmake --build "$BUILD_DIR" -j "$(nproc)" > /dev/null

echo "collecting profile over $IMAGE_DIR..."
LD_LIBRARY_PATH="$BUILD_DIR" "$BUILD_DIR/apriltag_demo" -t 4 -i 1 -x 1.0 \
    -f tagStandard52h13 -q "$IMAGE_DIR"/*.jpg > /dev/null

cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_C_FLAGS="-march=native -fprofile-use=$PROFILE_DIR -fprofile-correction -Wno-missing-profile" > /dev/null
cmake --build "$BUILD_DIR" -j "$(nproc)" > /dev/null

echo "PGO build ready in $BUILD_DIR/"
