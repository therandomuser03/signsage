// JNI bridge for Android: loads SLTModel and exposes translate(byte[], int) → String
//
// Build with:
//   ANDROID_ABI=arm64-v8a
//   ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ \
//     -shared -fPIC -std=c++17 \
//     -I${ANDROID_NDK}/sources/third_party/vulkan/include \
//     -I/path/to/libtorch/include \
//     -I/path/to/libtorch/include/torch/csrc/api/include \
//     slt_jni_android.cpp slt_model.cpp \
//     -L/path/to/libtorch/lib \
//     -lc10 -ltorch -ltorch_cpu -ljni \
//     -o ../app/src/main/jniLibs/${ANDROID_ABI}/libslt_jni.so

#include <jni.h>
#include <string>
#include <vector>
#include "slt_model.h"

static SLTModel* g_model = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_org_slt_SLTModule_initModel(JNIEnv* env, jobject /*thiz*/, jstring model_dir_j) {
    if (g_model) {
        delete g_model;
        g_model = nullptr;
    }
    const char* dir_chars = env->GetStringUTFChars(model_dir_j, nullptr);
    std::string model_dir(dir_chars);
    env->ReleaseStringUTFChars(model_dir_j, dir_chars);

    try {
        g_model = new SLTModel(model_dir);
        return JNI_TRUE;
    } catch (const std::exception& e) {
        // Log error — in production, use __android_log_print
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_slt_SLTModule_translate(JNIEnv* env, jobject /*thiz*/,
                                  jfloatArray keypoints_ja, jint num_frames) {
    if (!g_model) {
        return env->NewStringUTF("");
    }

    jfloat* kp = env->GetFloatArrayElements(keypoints_ja, nullptr);
    int total_elems = env->GetArrayLength(keypoints_ja);

    // Expected: 4 * 94 * num_frames floats
    auto result_str = g_model->translate(kp, num_frames);

    env->ReleaseFloatArrayElements(keypoints_ja, kp, JNI_ABORT);
    return env->NewStringUTF(result_str.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_org_slt_SLTModule_destroyModel(JNIEnv* /*env*/, jobject /*thiz*/) {
    if (g_model) {
        delete g_model;
        g_model = nullptr;
    }
}
