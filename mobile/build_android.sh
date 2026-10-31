#!/usr/bin/env bash
# Build the SLT JNI library for Android.
#
# Prerequisites:
#   - Download libtorch for Android from https://pytorch.org/get-started/locally/
#     (Select: Android → arm64-v8a → C++/Java)
#     Unzip to e.g. /opt/libtorch-android
#   - Set ANDROID_NDK to your NDK path (e.g. $ANDROID_HOME/ndk/26.1.10909125)
#   - Set TORCH_CMAKE_DIR to /opt/libtorch-android/share/cmake/Torch
#
# Usage:
#   TORCH_CMAKE_DIR=/opt/libtorch-android/share/cmake/Torch \
#   ANDROID_NDK=$ANDROID_HOME/ndk/26.1.10909125 \
#   ./build_android.sh

set -euo pipefail

: "${TORCH_CMAKE_DIR:?Set TORCH_CMAKE_DIR to libtorch/share/cmake/Torch}"
: "${ANDROID_NDK:?Set ANDROID_NDK to your NDK path}"

ABI="${1:-arm64-v8a}"
API_LEVEL=24
BUILD_DIR="build/android-${ABI}"

cmake -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ABI}" \
    -DANDROID_PLATFORM="android-${API_LEVEL}" \
    -DTORCH_CMAKE_DIR="${TORCH_CMAKE_DIR}" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" --config Release -j$(nproc)

echo "✅ Built: ${BUILD_DIR}/libslt_jni.so"
echo "Copy to: android/app/src/main/jniLibs/${ABI}/"
