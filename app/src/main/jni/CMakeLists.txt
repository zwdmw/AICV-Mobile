cmake_minimum_required(VERSION 3.10.2)
project("gyq_yolov8")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Werror -Wformat-security -O3 -ffast-math -ftree-vectorize")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -ffast-math -ftree-vectorize")

# Removed include directories for missing files
# include_directories(ocsort/include)
# include_directories(eigen)

# 为ARM CPU启用NEON - 仅针对ARMv7，ARMv8 (arm64-v8a)默认已启用
if(ANDROID_ABI STREQUAL "armeabi-v7a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon")
endif()

# 优化链接器
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

# 设置OpenMP标志
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.6.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc video)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20230223-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# YOLOx support
include_directories(${CMAKE_SOURCE_DIR}/yolox/jni)

# 主要的YOLO v8库
add_library(yolov8ncnn SHARED yolov8ncnn.cpp yolo.cpp ndkcamera.cpp)
target_link_libraries(yolov8ncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)

# 只保留YOLOv8，先注释掉YOLOx库
#add_library(ncnnyolox SHARED 
#    ${CMAKE_SOURCE_DIR}/yolox/jni/yoloxncnn.cpp 
#    ${CMAKE_SOURCE_DIR}/yolox/jni/yolox.cpp 
#    ${CMAKE_SOURCE_DIR}/yolox/jni/ndkcamera.cpp)
#target_link_libraries(ncnnyolox ncnn ${OpenCV_LIBS} camera2ndk mediandk)
