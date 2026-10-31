cmake_minimum_required(VERSION 3.18)
project(slt_jni CXX)

set(CMAKE_CXX_STANDARD 17)

# Find libtorch (set TORCH_CMAKE_DIR to point to libtorch/share/cmake/Torch)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
find_package(Torch REQUIRED)

# Android JNI
find_library(ANDROID_JNI_LIBRARY jni)

add_library(slt_jni SHARED
    slt_model.cpp
    slt_jni_android.cpp
)

target_include_directories(slt_jni PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(slt_jni
    ${TORCH_LIBRARIES}
    ${ANDROID_JNI_LIBRARY}
)
