#!/usr/bin/env bash
# Build and run the SLT test app on your Android phone.
#
# Prerequisites:
#   1. Android Studio installed with SDK 34 (compileSdk)
#   2. A phone connected via USB with USB debugging enabled
#   3. The project files are ready (models copied, test data prepared)
#
# Usage:
#   cd mobile/android_test
#   ./run_on_phone.sh
#
# To see results:
#   adb logcat -s SLTTest
#
# If you don't have the gradle wrapper (gradlew), generate it first:
#   cd mobile/android_test && gradle wrapper

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building SLT Test APK ==="

if [ ! -f gradlew ]; then
    echo "Error: gradlew not found."
    echo "Generate it with: cd $SCRIPT_DIR && gradle wrapper"
    echo "(Requires Gradle to be installed on your system)"
    exit 1
fi

chmod +x gradlew
./gradlew assembleDebug

echo ""
echo "=== Installing on phone ==="
adb install -r app/build/outputs/apk/debug/app-debug.apk 2>/dev/null || \
    adb install -r app/build/outputs/apk/debug/app-debug.apk

echo ""
echo "=== Launching ==="
adb shell am start -n org.slt/.TestActivity

echo ""
echo "=== Watching results (Ctrl+C to stop) ==="
echo "    Results will appear in ~10-60 seconds (57 samples)"
adb logcat -c
adb logcat -s SLTTest
