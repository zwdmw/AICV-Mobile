#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <set>
#include <atomic>
#include <unistd.h>  // For usleep

#include <platform.h>
#include <benchmark.h>

#include "yolo.h"
#include "Object.h"
#include "ndkcamera.h"
#include "net.h"
#include "TrackingParams.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

// Add a global variable to track the current detection count
static int g_detection_count = 0;

// Add a set to store enabled class IDs (using a set for fast lookups)
static std::set<int> g_enabled_class_ids;

// Flag to indicate if class filtering is enabled
static bool g_class_filtering_enabled = false;

// 定义检测阈值变量，默认值为0.4
static float g_detection_threshold = 0.4f;

// 添加掩码阈值变量，默认值为0.4
static float g_mask_threshold = 0.4f;

// Add global atomic flag to indicate camera state at the top of the file, after other globals
std::atomic<bool> g_camera_active(false);

// Add a global variable to store the current FPS
static int g_current_fps = 0;

static int draw_unsupported(cv::Mat &rgb) {
    const char text[] = "unsupported";
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);
    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;
    cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);
    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));
    return 0;
}

static int draw_fps(cv::Mat &rgb) {
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};
        double t1 = ncnn::get_current_time();
        if (t0 == 0.f) {
            t0 = t1;
            return 0;
        }
        float fps = 1000.f / (t1 - t0);
        t0 = t1;
        for (int i = 9; i >= 1; i--) {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;
        if (fps_history[9] == 0.f) {
            return 0;
        }
        for (int i = 0; i < 10; i++) {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }
    
    // Store the calculated FPS in a global variable for access from Java
    g_current_fps = (int)avg_fps;
    
    // Don't draw FPS directly on the frame since we'll handle it in the UI
    return 0;
}

/*声明用于做目标检测的yolo*/
static Yolo *g_yolo = nullptr;

static ncnn::Mutex Lock;

// note: 必须需要继承官方提供的 NdkCameraWindow，重写 on_image_render。
class MyNdkCamera : public NdkCameraWindow {
public:
    virtual void on_image_render(cv::Mat &rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat &rgb) const {
    // Check if camera is still active before processing
    if (!g_camera_active.load()) {
        __android_log_print(ANDROID_LOG_WARN, "ncnn", "Skipping image processing as camera is inactive");
        return;
    }

    // 下面是整个检测流程
    {
        ncnn::MutexLockGuard g(Lock);
        if (g_yolo) {
            std::vector<Object> objects;
            // 使用全局阈值变量来进行检测
            g_yolo->detect(rgb, objects, g_detection_threshold);
            
            // Filter objects based on enabled class IDs if filtering is enabled
            if (g_class_filtering_enabled) {
                std::vector<Object> filtered_objects;
                for (const auto& obj : objects) {
                    if (g_enabled_class_ids.find(obj.label) != g_enabled_class_ids.end()) {
                        filtered_objects.push_back(obj);
                    }
                }
                objects = filtered_objects;
            }
            
            // Update the global detection count with the number of detected objects
            g_detection_count = objects.size();
            // 追踪 Object
            // ocsort的追踪函数就写在这个 draw 函数的实现里面。
            g_yolo->draw(rgb, objects);
        } else {
            draw_unsupported(rgb);
        }
    }
    draw_fps(rgb);
}

/*************************************************************/
/*********************下面都是用来跟Java交互的 *******************/
/*****note:JNI方法的命名要求：Java_<包名>_<类名>_<方法名> **********/
/*************************************************************/

static MyNdkCamera *g_camera = nullptr;

extern "C" {
JNIEXPORT jint
JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "JNI_OnLoad开始初始化");

    // 确保g_camera有效
    if (g_camera == nullptr) {
        g_camera = new MyNdkCamera;
        if (g_camera == nullptr) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to create camera object in JNI_OnLoad!");
            return JNI_ERR;
        }
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Camera object created: %p", g_camera);
    } else {
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Using existing camera object: %p", g_camera);
    }

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "JNI_OnUnload开始清理资源");

    // 首先关闭相机
    if (g_camera) {
        g_camera->close();
    }

    {
        ncnn::MutexLockGuard g(Lock);

        delete g_yolo;
        g_yolo = 0;
    }

    // 安全删除相机对象
    if (g_camera) {
        delete g_camera;
        g_camera = 0;
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Camera object deleted");
    }
}

// public native boolean loadModel(AssetManager mgr, String modelName, int cpugpu);
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_loadModel(JNIEnv *env, jobject thiz, jobject assetManager,
                                           jstring modelName, jint cpugpu) {
    if (cpugpu < 0 || cpugpu > 1) {
        return JNI_FALSE;
    }

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    // 将jstring转换为C++字符串
    const char* modelNameCStr = env->GetStringUTFChars(modelName, nullptr);
    if (modelNameCStr == nullptr) {
        // JNI错误，内存不足
        return JNI_FALSE;
    }
    
    std::string modelNameStr(modelNameCStr);
    
    // 释放JNI字符串资源
    env->ReleaseStringUTFChars(modelName, modelNameCStr);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p, modelName=%s", mgr, modelNameStr.c_str());

    // 检查模型类型，不再排除YOLOx模型
    // 判断是否是YOLOx模型
    bool is_yolox_model = (modelNameStr.find("yolox") != std::string::npos);
    if (is_yolox_model) {
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "正在加载YOLOx模型: %s - 请使用专用的YOLOx处理接口", modelNameStr.c_str());
        // YOLOx模型需要通过专门的接口处理，这里仍然允许加载，但提示应该使用专用接口
    }

    // 检查模型名是否包含"seg"来确定是否为分割模型
    bool is_segmentation_model = (modelNameStr.find("seg") != std::string::npos);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Model %s is segmentation model: %d", 
                        modelNameStr.c_str(), is_segmentation_model);

    // 确定目标尺寸，优化为当前设备性能最佳的值
    int target_size = 320;
    
    // 设置标准的预处理参数（适用于所有YOLO模型）
    float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    
    bool use_gpu = (int) cpugpu == 1;

    // 加载模型
    {
        ncnn::MutexLockGuard g(Lock);

        if (use_gpu && ncnn::get_gpu_count() == 0) {
            // 没有可用的GPU
            delete g_yolo;
            g_yolo = nullptr;
            return JNI_FALSE;
        } else {
            if (!g_yolo)
                g_yolo = new Yolo;
                
            // 传递模型名而非模型类型（不要添加yolov8前缀）
            int ret = g_yolo->load(mgr, modelNameStr.c_str(), target_size, mean_vals, norm_vals, use_gpu);
            
            if (ret != 0) {
                __android_log_print(ANDROID_LOG_ERROR, "ncnn", "yolo->load failed %d", ret);
                return JNI_FALSE;
            }
            
            __android_log_print(ANDROID_LOG_INFO, "ncnn", "yolo->load success!");
            return JNI_TRUE;
        }
    }
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_openCamera(JNIEnv *env, jobject thiz, jint facing) {
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_INFO, "ncnn", "openCamera %d", facing);
    
    // Set the active flag to true before opening camera
    g_camera_active.store(true);
    
    // 检查相机对象
    if (g_camera == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Camera object is null, creating new camera");
        g_camera = new MyNdkCamera;
        if (g_camera == nullptr) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to create camera object!");
            return JNI_FALSE;
        }
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Opening camera, g_camera=%p", g_camera);
    
    // 先尝试关闭一次，确保上一次的资源被正确释放
    g_camera->close();
    
    // 打开相机并返回结果
    int ret = g_camera->open((int) facing);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Camera open result: %d", ret);

    return ret == 0 ? JNI_TRUE : JNI_FALSE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL 
Java_com_gyq_yolov8_Yolov8Ncnn_closeCamera(JNIEnv *env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");
    
    // First set the active flag to false so image processing threads know to stop
    g_camera_active.store(false);
    
    // Sleep a short time to give processing threads time to check the flag
    #ifdef _WIN32
        Sleep(50); // Windows sleep in milliseconds
    #else
        usleep(50000); // POSIX sleep in microseconds (50ms)
    #endif
    
    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setOutputWindow(JNIEnv *env, jobject thiz, jobject surface) {
    if (surface == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Input surface is null!");
        return JNI_FALSE;
    }

    ANativeWindow *win = ANativeWindow_fromSurface(env, surface);
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "setOutputWindow %p", win);
    
    if (win == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Surface window is null!");
        return JNI_FALSE;
    }
    
    // 检查窗口有效性
    int width = ANativeWindow_getWidth(win);
    int height = ANativeWindow_getHeight(win);
    int format = ANativeWindow_getFormat(win);
    
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "Window dimensions: %dx%d, format: %d", width, height, format);
    
    if (width <= 0 || height <= 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Invalid window dimensions: %dx%d", width, height);
        ANativeWindow_release(win);
        return JNI_FALSE;
    }
    
    if (g_camera == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Camera object is null when setting window!");
        ANativeWindow_release(win);
        return JNI_FALSE;
    }

    // 设置窗口
    g_camera->set_window(win);
    
    // 释放本地引用，因为set_window中已经执行了acquire
    ANativeWindow_release(win);

    return JNI_TRUE;
}

// public native int getDetectionCount();
JNIEXPORT jint JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_getDetectionCount(JNIEnv *env, jobject thiz) {
    // Return the current detection count
    return g_detection_count;
}

// public native int getFilteredDetectionCount();
JNIEXPORT jint JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_getFilteredDetectionCount(JNIEnv *env, jobject thiz) {
    // For now, this is the same as getDetectionCount since filtering
    // is already applied in the detection process
    return g_detection_count;
}

// public native boolean setEnabledClassIds(int[] classIds);
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setEnabledClassIds(JNIEnv *env, jobject thiz, jintArray classIds) {
    // Clear existing filter if null is passed
    if (classIds == nullptr) {
        ncnn::MutexLockGuard g(Lock);
        if (g_yolo) {
            g_yolo->clearClassFilter();
            return JNI_TRUE;
        }
        return JNI_FALSE;
    }
    
    // Get array elements
    jint* classIdsElements = env->GetIntArrayElements(classIds, nullptr);
    if (classIdsElements == nullptr) {
        return JNI_FALSE;
    }
    
    jsize length = env->GetArrayLength(classIds);
    
    // Create a set of enabled class IDs
    std::set<int> enabledClassIds;
    for (jsize i = 0; i < length; i++) {
        enabledClassIds.insert(classIdsElements[i]);
    }
    
    // Release the array
    env->ReleaseIntArrayElements(classIds, classIdsElements, JNI_ABORT);
    
    // Set the enabled class IDs in Yolo
    ncnn::MutexLockGuard g(Lock);
    if (g_yolo) {
        g_yolo->setEnabledClassIds(enabledClassIds);
        return JNI_TRUE;
    }
    
    return JNI_FALSE;
}

// public native boolean setDetectionThreshold(float threshold);
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setDetectionThreshold(JNIEnv *env, jobject thiz, jfloat threshold) {
    if (threshold < 0.0f || threshold > 1.0f) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Invalid threshold value: %f (should be between 0.0 and 1.0)", threshold);
        return JNI_FALSE;
    }
    
    // 设置全局阈值变量
    g_detection_threshold = threshold;
    
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Set detection threshold to %f", threshold);
    
    return JNI_TRUE;
}

// public native boolean setMaskThreshold(float threshold);
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setMaskThreshold(JNIEnv *env, jobject thiz, jfloat threshold) {
    if (threshold < 0.0f || threshold > 1.0f) {
        __android_log_print(ANDROID_LOG_WARN, "ncnn", "Mask threshold %.2f out of range (0-1)", threshold);
        return JNI_FALSE;
    }
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setMaskThreshold %.2f", threshold);
    g_mask_threshold = threshold;

    // Propagate the mask threshold to the Yolo instance
    {
        ncnn::MutexLockGuard g(Lock);
        if (g_yolo) {
            g_yolo->setMaskThreshold(threshold);
            return JNI_TRUE;
        }
    }
    return JNI_FALSE;
}

// 添加控制跟踪功能的JNI方法
// public native boolean setEnableTracking(boolean enable);
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setEnableTracking(JNIEnv *env, jobject thiz, jboolean enable) {
    ncnn::MutexLockGuard g(Lock);
    if (g_yolo) {
        g_yolo->setTrackingEnabled(enable);
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Tracking %s via JNI", enable ? "enabled" : "disabled");
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// New JNI function to set all tracking parameters
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setTrackingParams(JNIEnv *env, jobject thiz,
                                                 jfloat iou_threshold, jint max_age, jint min_hits,
                                                 jfloat kalman_process_noise, jfloat kalman_measurement_noise,
                                                 jboolean enable_mask_tracking, jboolean enable_trajectory_viz,
                                                 jint trajectory_length, jfloat trajectory_thickness, 
                                                 jint trajectory_color_mode) {
    ncnn::MutexLockGuard g(Lock);
    if (g_yolo) {
        TrackingParams params;
        params.iou_threshold = iou_threshold;
        params.max_age = max_age;
        params.min_hits = min_hits;
        params.kalman_process_noise = kalman_process_noise;
        params.kalman_measurement_noise = kalman_measurement_noise;
        params.enable_mask_tracking = enable_mask_tracking;
        // Set trajectory visualization parameters
        params.enable_trajectory_viz = enable_trajectory_viz;
        params.trajectory_length = trajectory_length;
        params.trajectory_thickness = trajectory_thickness;
        params.trajectory_color_mode = trajectory_color_mode;

        g_yolo->setTrackingParams(params); // Call the C++ Yolo method

        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Tracking parameters set via JNI: IoU=%.2f, MaxAge=%d, MinHits=%d, KF_P=%.2e, KF_M=%.2e, MaskTrack=%d, TrajViz=%d, TrajLen=%d, TrajThick=%.1f, TrajColor=%d",
                            params.iou_threshold, params.max_age, params.min_hits,
                            params.kalman_process_noise, params.kalman_measurement_noise,
                            params.enable_mask_tracking, params.enable_trajectory_viz,
                            params.trajectory_length, params.trajectory_thickness,
                            params.trajectory_color_mode);
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// New JNI function for continuous tracking parameters
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setTrackingParamsWithContinuous(JNIEnv *env, jobject thiz,
                                                 jfloat iou_threshold, jint max_age, jint min_hits,
                                                 jfloat kalman_process_noise, jfloat kalman_measurement_noise,
                                                 jboolean enable_mask_tracking, jboolean enable_trajectory_viz,
                                                 jint trajectory_length, jfloat trajectory_thickness, 
                                                 jint trajectory_color_mode, jboolean enable_continuous_tracking, 
                                                 jfloat prediction_probability_threshold, jint max_prediction_frames, 
                                                 jint prediction_line_type) {
    ncnn::MutexLockGuard g(Lock);
    if (g_yolo) {
        TrackingParams params;
        params.iou_threshold = iou_threshold;
        params.max_age = max_age;
        params.min_hits = min_hits;
        params.kalman_process_noise = kalman_process_noise;
        params.kalman_measurement_noise = kalman_measurement_noise;
        params.enable_mask_tracking = enable_mask_tracking;
        // Set trajectory visualization parameters
        params.enable_trajectory_viz = enable_trajectory_viz;
        params.trajectory_length = trajectory_length;
        params.trajectory_thickness = trajectory_thickness;
        params.trajectory_color_mode = trajectory_color_mode;
        // 设置连续跟踪模式参数
        params.enable_continuous_tracking = enable_continuous_tracking;
        params.prediction_probability_threshold = prediction_probability_threshold;
        params.max_prediction_frames = max_prediction_frames;
        params.prediction_line_type = prediction_line_type;

        g_yolo->setTrackingParams(params); // Call the C++ Yolo method

        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Tracking parameters set via JNI: IoU=%.2f, MaxAge=%d, MinHits=%d, KF_P=%.2e, KF_M=%.2e, MaskTrack=%d, TrajViz=%d, TrajLen=%d, TrajThick=%.1f, TrajColor=%d, PredLine=%d",
                            params.iou_threshold, params.max_age, params.min_hits,
                            params.kalman_process_noise, params.kalman_measurement_noise,
                            params.enable_mask_tracking, params.enable_trajectory_viz,
                            params.trajectory_length, params.trajectory_thickness,
                            params.trajectory_color_mode, params.prediction_line_type);
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// New JNI function for tracking parameters with mode
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setTrackingParamsWithMode(JNIEnv *env, jobject thiz,
                                                 jfloat iou_threshold, jint max_age, jint min_hits,
                                                 jfloat kalman_process_noise, jfloat kalman_measurement_noise,
                                                 jboolean enable_mask_tracking, jboolean enable_trajectory_viz,
                                                 jint trajectory_length, jfloat trajectory_thickness, 
                                                 jint trajectory_color_mode, jboolean enable_continuous_tracking, 
                                                 jfloat prediction_probability_threshold, jint max_prediction_frames, 
                                                 jint prediction_line_type, jint tracking_mode,
                                                 jint motion_submode) {
    ncnn::MutexLockGuard g(Lock);
    if (g_yolo) {
        TrackingParams params;
        // 基本参数
        params.iou_threshold = iou_threshold;
        params.max_age = max_age;
        params.min_hits = min_hits;
        params.kalman_process_noise = kalman_process_noise;
        params.kalman_measurement_noise = kalman_measurement_noise;
        params.enable_mask_tracking = enable_mask_tracking;
        // 轨迹可视化参数
        params.enable_trajectory_viz = enable_trajectory_viz;
        params.trajectory_length = trajectory_length;
        params.trajectory_thickness = trajectory_thickness;
        params.trajectory_color_mode = trajectory_color_mode;
        // 连续跟踪模式参数
        params.enable_continuous_tracking = enable_continuous_tracking;
        params.prediction_probability_threshold = prediction_probability_threshold;
        params.max_prediction_frames = max_prediction_frames;
        params.prediction_line_type = prediction_line_type;
        // 跟踪模式
        params.tracking_mode = tracking_mode;
        // 运动子模式
        params.motion_submode = motion_submode;
        
        // 根据模式设置额外参数
        if (tracking_mode == 1) { // 手持模式
            params.enable_motion_compensation = true;
            params.acceleration_weight = 0.2f;
            params.velocity_history_length = 5;
        } else {
            params.enable_motion_compensation = false;
            params.acceleration_weight = 0.0f;
            params.velocity_history_length = 3;
        }
        
        g_yolo->setTrackingParams(params); // Call the C++ Yolo method

        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Tracking parameters set with mode %d, submode %d: IoU=%.2f...",
                            tracking_mode, motion_submode, params.iou_threshold);
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// New JNI function for tracking parameters with modes - including spatial distribution
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setTrackingParamsWithModes(JNIEnv *env, jobject thiz,
                                                 jfloat iou_threshold, jint max_age, jint min_hits,
                                                 jfloat kalman_process_noise, jfloat kalman_measurement_noise,
                                                 jboolean enable_mask_tracking, jboolean enable_trajectory_viz,
                                                 jint trajectory_length, jfloat trajectory_thickness, 
                                                 jint trajectory_color_mode, jboolean enable_continuous_tracking, 
                                                 jfloat prediction_probability_threshold, jint max_prediction_frames, 
                                                 jint prediction_line_type, jint tracking_mode,
                                                 jint motion_submode, jint spatial_distribution)
{
    ncnn::MutexLockGuard g(Lock);
    if (g_yolo) {
        TrackingParams params;
        // 基本参数
        params.iou_threshold = iou_threshold;
        params.max_age = max_age;
        params.min_hits = min_hits;
        params.kalman_process_noise = kalman_process_noise;
        params.kalman_measurement_noise = kalman_measurement_noise;
        params.enable_mask_tracking = enable_mask_tracking;
        // 轨迹可视化参数
        params.enable_trajectory_viz = enable_trajectory_viz;
        params.trajectory_length = trajectory_length;
        params.trajectory_thickness = trajectory_thickness;
        params.trajectory_color_mode = trajectory_color_mode;
        // 连续跟踪模式参数
        params.enable_continuous_tracking = enable_continuous_tracking;
        params.prediction_probability_threshold = prediction_probability_threshold;
        params.max_prediction_frames = max_prediction_frames;
        params.prediction_line_type = prediction_line_type;
        // 跟踪模式
        params.tracking_mode = tracking_mode;
        // 运动子模式
        params.motion_submode = motion_submode;
        // 空间分布模式
        params.spatial_distribution = spatial_distribution;
        
        // 根据模式设置额外参数
        if (tracking_mode == 1) { // 手持模式
            params.enable_motion_compensation = true;
            params.acceleration_weight = 0.2f;
            params.velocity_history_length = 5;
        } else {
            params.enable_motion_compensation = false;
            params.acceleration_weight = 0.0f;
            params.velocity_history_length = 3;
        }
        
        // 根据空间分布模式自动启用特殊功能
        if (spatial_distribution == SPATIAL_VERY_DENSE) {
            // 超密集场景启用ReID特征和群组管理
            params.use_reid_features = true;
            params.use_group_management = true;
            params.use_occlusion_handling = true;
        } else if (spatial_distribution == SPATIAL_DENSE) {
            // 密集场景只启用遮挡处理
            params.use_occlusion_handling = true;
            params.use_reid_features = false;
            params.use_group_management = false;
        } else if (spatial_distribution == SPATIAL_SPARSE) {
            // 稀疏场景不需要特殊处理
            params.use_reid_features = false;
            params.use_group_management = false;
            params.use_occlusion_handling = false;
        } else { // SPATIAL_NORMAL 或其他值
            // 正常密度场景，默认设置
            params.use_reid_features = false;
            params.use_group_management = false;
            params.use_occlusion_handling = false;
        }
        
        g_yolo->setTrackingParams(params); // Call the C++ Yolo method

        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Tracking parameters set with mode %d, submode %d, spatial %d: IoU=%.2f, special=%d,%d,%d",
                            tracking_mode, motion_submode, spatial_distribution, params.iou_threshold,
                            params.use_reid_features, params.use_group_management, params.use_occlusion_handling);
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// JNI function to set the drawing style -> Renamed to setDetectionStyle
// extern "C" JNIEXPORT void JNICALL Java_com_gyq_yolov8_Yolov8Ncnn_setDrawStyle(JNIEnv* env, jobject thiz, jint style)
extern "C" JNIEXPORT void JNICALL Java_com_gyq_yolov8_Yolov8Ncnn_setDetectionStyle(JNIEnv* env, jobject thiz, jint style)
{
    if (g_yolo)
    {
        // Ensure style is within expected bounds (0 to 5) - This check is now inside Yolo::setDetectionStyle
        // int draw_style = (style >= 0 && style <= 5) ? style : 0; // Default to 0 if out of bounds
        // g_yolo->setDrawStyle(draw_style);
        g_yolo->setDetectionStyle(style); // Call the renamed method
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Set detection style to %d", style);
    }
}

// 添加控制分割掩码跟踪功能的JNI方法
// public native boolean setEnableMaskTracking(boolean enable);
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setEnableMaskTracking(JNIEnv *env, jobject thiz, jboolean enable) {
    ncnn::MutexLockGuard g(Lock);
    if (g_yolo) {
        // This now internally calls setTrackingParams
        g_yolo->setEnableMaskTracking(enable); 
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Mask Tracking %s via JNI", enable ? "enabled" : "disabled");
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// public native int getDatasetType();
JNIEXPORT jint JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_getDatasetType(JNIEnv *env, jobject thiz) {
    ncnn::MutexLockGuard g(Lock);

    if (!g_yolo) {
        return DATASET_COCO; // Default to COCO
    }

    return g_yolo->getDatasetType();
}

// 实现获取OIV类别名称的JNI方法
JNIEXPORT jobjectArray JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_getOIVClassNames(JNIEnv *env, jobject thiz) {
    // 获取完整的OIV类别名称列表
    ncnn::MutexLockGuard g(Lock);
    
    // Count OIV class names - 直接从yolo.h中定义的全局数组中获取
    int count = sizeof(oiv_class_names) / sizeof(oiv_class_names[0]);
    
    // 创建Java的String数组
    jclass stringClass = env->FindClass("java/lang/String");
    jobjectArray result = env->NewObjectArray(count, stringClass, nullptr);
    
    // 填充数组
    for (int i = 0; i < count; i++) {
        jstring str = env->NewStringUTF(oiv_class_names[i]);
        env->SetObjectArrayElement(result, i, str);
        env->DeleteLocalRef(str);
    }
    
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "Returning %d OIV class names to Java", count);
    
    return result;
}

// New JNI method: loadClassesFromFile
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_loadClassesFromFile(JNIEnv *env, jobject thiz, jobject assetManager, jstring modelName) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr) {
        return JNI_FALSE;
    }
    
    // Convert jstring to C++ string
    const char* modelNameCStr = env->GetStringUTFChars(modelName, nullptr);
    if (modelNameCStr == nullptr) {
        return JNI_FALSE;
    }
    
    bool success = false;
    {
        ncnn::MutexLockGuard g(Lock);
        if (g_yolo) {
            int result = g_yolo->loadClassesFromFile(mgr, modelNameCStr);
            success = (result > 0);
            __android_log_print(ANDROID_LOG_INFO, "YoloClasses", "已加载 %d 个类别", result);
        }
    }
    
    // Release JNI string resources
    env->ReleaseStringUTFChars(modelName, modelNameCStr);
    return success ? JNI_TRUE : JNI_FALSE;
}

// New JNI method: clearClassFilter
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_clearClassFilter(JNIEnv *env, jobject thiz) {
    ncnn::MutexLockGuard g(Lock);
    if (g_yolo) {
        g_yolo->clearClassFilter();
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// New JNI method: isClassFilteringEnabled
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_isClassFilteringEnabled(JNIEnv *env, jobject thiz) {
    ncnn::MutexLockGuard g(Lock);
    if (g_yolo) {
        return g_yolo->isClassFilteringEnabled() ? JNI_TRUE : JNI_FALSE;
    }
    return JNI_FALSE;
}

// public native int getCurrentFPS();
JNIEXPORT jint JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_getCurrentFPS(JNIEnv *env, jobject thiz) {
    // Return the current FPS value
    return g_current_fps;
}

// 新增：设置中文类别名称的JNI方法
JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setChineseClassNames(JNIEnv *env, jobject thiz, jobjectArray namesArray) {
    if (!namesArray) {
        __android_log_print(ANDROID_LOG_ERROR, "YoloClasses", "setChineseClassNames: 传入的名称数组为空");
        return JNI_FALSE;
    }
    
    jsize count = env->GetArrayLength(namesArray);
    if (count <= 0) {
        __android_log_print(ANDROID_LOG_ERROR, "YoloClasses", "setChineseClassNames: 传入的名称数组长度为0");
        return JNI_FALSE;
    }
    
    // 将Java字符串数组转换为C++字符串数组
    const char** names = new const char*[count];
    for (int i = 0; i < count; i++) {
        jstring jstr = (jstring)env->GetObjectArrayElement(namesArray, i);
        if (jstr) {
            names[i] = env->GetStringUTFChars(jstr, NULL);
        } else {
            names[i] = NULL;
        }
    }
    
    // 调用Yolo类的方法设置中文名称
    bool success = false;
    {
        ncnn::MutexLockGuard g(Lock);
        if (g_yolo) {
            g_yolo->setChineseClassNames(names, count);
            success = true;
        }
    }
    
    // 释放Java字符串资源
    for (int i = 0; i < count; i++) {
        jstring jstr = (jstring)env->GetObjectArrayElement(namesArray, i);
        if (jstr && names[i]) {
            env->ReleaseStringUTFChars(jstr, names[i]);
            env->DeleteLocalRef(jstr);
        }
    }
    
    // 释放C++数组
    delete[] names;
    
    return success ? JNI_TRUE : JNI_FALSE;
}

// 新增：更新的设置样式参数的JNI方法（包含文字风格和背景覆盖参数）
extern "C" JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setStyleParameters(JNIEnv *env, jobject thiz, 
                                                 jint styleId, jint lineThickness, jint lineType,
                                                 jfloat boxAlpha, jint boxColor, jfloat fontSize,
                                                 jint fontType, jint textColor, jint textStyle,
                                                 jboolean fullTextBackground,
                                                 jfloat maskAlpha, jfloat maskContrast) {
    ncnn::MutexLockGuard g(Lock);
    if (g_yolo) {
        // Convert box color to individual RGB components for better logging
        int r = (boxColor >> 16) & 0xFF;
        int g = (boxColor >> 8) & 0xFF;
        int b = boxColor & 0xFF;
        int a = (boxColor >> 24) & 0xFF;
        
        // Convert text color to individual RGB components
        int textR = (textColor >> 16) & 0xFF;
        int textG = (textColor >> 8) & 0xFF;
        int textB = textColor & 0xFF;
        int textA = (textColor >> 24) & 0xFF;
        
        // Log detailed information about the style parameters
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                           "Style ID: %d - Applying parameters:", styleId);
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                           "  Line: thickness=%d, type=%d", lineThickness, lineType);
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                           "  Box: alpha=%.2f, color=0x%08X (RGBA: %d,%d,%d,%d)", 
                           boxAlpha, boxColor, r, g, b, a);
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                           "  Font: size=%.2f, type=%d, color=0x%08X (RGBA: %d,%d,%d,%d)", 
                           fontSize, fontType, textColor, textR, textG, textB, textA);
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                           "  Text Style: %d, Full background: %s",
                           textStyle, fullTextBackground ? "true" : "false");
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                           "  Mask: alpha=%.2f, contrast=%.2f", 
                           maskAlpha, maskContrast);
        
        // Call the Yolo class implementation of setStyleParameters with the additional parameters
        // Note: This requires updating the Yolo class to accept these new parameters
        return g_yolo->setStyleParameters(styleId, lineThickness, lineType, boxAlpha, boxColor,
                                        fontSize, fontType, textColor, textStyle, fullTextBackground,
                                        maskAlpha, maskContrast,
                                        1, 0, 0xFFFFFFFF); // Add default mask edge params
    }
    return JNI_FALSE;
}

// 旧版本的setStyleParameters方法，我们保留它以保持向后兼容性
extern "C" JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setStyleParameters__IIFIFIIFF(JNIEnv *env, jobject thiz, 
                                                 jint styleId, jint lineThickness, jint lineType,
                                                 jfloat boxAlpha, jint boxColor, jfloat fontSize,
                                                 jint fontType, jint textColor,
                                                 jfloat maskAlpha, jfloat maskContrast) {
    ncnn::MutexLockGuard g(Lock);
    if (g_yolo) {
        // Log the deprecated method call
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                           "Using deprecated setStyleParameters method for Style ID: %d", styleId);
        
        // Call the new method with default values for new parameters
        return g_yolo->setStyleParameters(styleId, lineThickness, lineType, boxAlpha, boxColor,
                                        fontSize, fontType, textColor, 0, true,
                                        maskAlpha, maskContrast,
                                        1, 0, 0xFFFFFFFF); // Add default mask edge params
    }
    return JNI_FALSE;
}

// 新增：获取特定风格的样式参数的JNI方法
extern "C" JNIEXPORT jobject JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_getStyleParameters(JNIEnv *env, jobject thiz, jint styleId) {
    ncnn::MutexLockGuard g(Lock);
    
    // Create the StyleParameters class object
    jclass styleParamsClass = env->FindClass("com/gyq/yolov8/StyleParameters");
    if (styleParamsClass == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "YoloStyle", "Cannot find StyleParameters class");
        return nullptr;
    }
    
    // Get constructor ID
    jmethodID constructor = env->GetMethodID(styleParamsClass, "<init>", "(I)V");
    if (constructor == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "YoloStyle", "Cannot find StyleParameters constructor");
        return nullptr;
    }
    
    // Create a new StyleParameters object
    jobject styleParams = env->NewObject(styleParamsClass, constructor, styleId);
    
    if (g_yolo) {
        // Variables to store parameters
        int lineThickness, lineType, fontType;
        float boxAlpha, fontSize, maskAlpha, maskContrast;
        int boxColor, textColor;
        // Add mask edge parameter variables
        int maskEdgeThickness, maskEdgeType, maskEdgeColor;
        
        // Get parameters from Yolo
        bool success = g_yolo->getStyleParameters(styleId, &lineThickness, &lineType,
                                                 &boxAlpha, &boxColor, &fontSize,
                                                 &fontType, &textColor,
                                                 &maskAlpha, &maskContrast,
                                                 &maskEdgeThickness, &maskEdgeType, &maskEdgeColor);
        
        if (success) {
            // 详细记录获取到的参数，以便调试颜色问题
            __android_log_print(ANDROID_LOG_DEBUG, "YoloStyle", 
                               "JNI: Retrieved parameters for Style ID: %d", styleId);
            
            // 记录获取到的颜色值
            int r_box = (boxColor >> 16) & 0xFF;
            int g_box = (boxColor >> 8) & 0xFF;
            int b_box = boxColor & 0xFF;
            int a_box = (boxColor >> 24) & 0xFF;
            
            int r_text = (textColor >> 16) & 0xFF;
            int g_text = (textColor >> 8) & 0xFF;
            int b_text = textColor & 0xFF;
            int a_text = (textColor >> 24) & 0xFF;
            
            __android_log_print(ANDROID_LOG_DEBUG, "YoloStyle", 
                               "JNI Box Color: 0x%08X (RGBA: %d,%d,%d,%d)", 
                               boxColor, r_box, g_box, b_box, a_box);
                               
            __android_log_print(ANDROID_LOG_DEBUG, "YoloStyle", 
                               "JNI Text Color: 0x%08X (RGBA: %d,%d,%d,%d)",
                               textColor, r_text, g_text, b_text, a_text);
            
            // Get field IDs
            jfieldID lineThicknessField = env->GetFieldID(styleParamsClass, "lineThickness", "I");
            jfieldID lineTypeField = env->GetFieldID(styleParamsClass, "lineType", "I");
            jfieldID boxAlphaField = env->GetFieldID(styleParamsClass, "boxAlpha", "F");
            jfieldID boxColorField = env->GetFieldID(styleParamsClass, "boxColor", "I");
            jfieldID fontSizeField = env->GetFieldID(styleParamsClass, "fontSize", "F");
            jfieldID fontTypeField = env->GetFieldID(styleParamsClass, "fontType", "I");
            jfieldID textColorField = env->GetFieldID(styleParamsClass, "textColor", "I");
            jfieldID maskAlphaField = env->GetFieldID(styleParamsClass, "maskAlpha", "F");
            jfieldID maskContrastField = env->GetFieldID(styleParamsClass, "maskContrast", "F");
            // Get mask edge field IDs
            jfieldID maskEdgeThicknessField = env->GetFieldID(styleParamsClass, "maskEdgeThickness", "I");
            jfieldID maskEdgeTypeField = env->GetFieldID(styleParamsClass, "maskEdgeType", "I");
            jfieldID maskEdgeColorField = env->GetFieldID(styleParamsClass, "maskEdgeColor", "I");
            
            // Set fields
            env->SetIntField(styleParams, lineThicknessField, lineThickness);
            env->SetIntField(styleParams, lineTypeField, lineType);
            env->SetFloatField(styleParams, boxAlphaField, boxAlpha);
            env->SetIntField(styleParams, boxColorField, boxColor);
            env->SetFloatField(styleParams, fontSizeField, fontSize);
            env->SetIntField(styleParams, fontTypeField, fontType);
            env->SetIntField(styleParams, textColorField, textColor);
            env->SetFloatField(styleParams, maskAlphaField, maskAlpha);
            env->SetFloatField(styleParams, maskContrastField, maskContrast);
            // Set mask edge fields
            env->SetIntField(styleParams, maskEdgeThicknessField, maskEdgeThickness);
            env->SetIntField(styleParams, maskEdgeTypeField, maskEdgeType);
            env->SetIntField(styleParams, maskEdgeColorField, maskEdgeColor);
            
            // Log the retrieved parameters
            __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                               "Retrieved parameters for Style ID: %d", styleId);
            __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                               "  Line: thickness=%d, type=%d", lineThickness, lineType);
            __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                               "  Box: alpha=%.2f, color=0x%08X", boxAlpha, boxColor);
            __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                               "  Font: size=%.2f, type=%d, color=0x%08X", fontSize, fontType, textColor);
            __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                               "  Mask: alpha=%.2f, contrast=%.2f", maskAlpha, maskContrast);
            // Log the mask edge parameters
            __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                               "  Mask Edge: thickness=%d, type=%d, color=0x%08X", 
                               maskEdgeThickness, maskEdgeType, maskEdgeColor);
        }
    }
    
    return styleParams;
}

// New JNI method: resetStyleParameters
extern "C" JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_resetStyleParameters(JNIEnv *env, jobject thiz, jint styleId) {
    ncnn::MutexLockGuard g(Lock);
    if (g_yolo) {
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                         "重置风格参数 - ID: %d", styleId);
        
        // 调用Yolo类的resetStyleParameters方法
        return g_yolo->resetStyleParameters(styleId);
    }
    return JNI_FALSE;
}

// 新增：实现 setStyleParametersWithMaskEdge JNI 方法
extern "C" JNIEXPORT jboolean JNICALL
Java_com_gyq_yolov8_Yolov8Ncnn_setStyleParametersWithMaskEdge(
        JNIEnv *env, jobject thiz,
        jint styleId, jint lineThickness, jint lineType,
        jfloat boxAlpha, jint boxColor, jfloat fontSize,
        jint fontType, jint textColor, jint textStyle,
        jboolean fullTextBackground,
        jfloat maskAlpha, jfloat maskContrast,
        jint maskEdgeThickness, jint maskEdgeType, jint maskEdgeColor) {
    ncnn::MutexLockGuard g(Lock);
    if (g_yolo) {
        // Log the received parameters for debugging
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle",
                           "JNI setStyleParametersWithMaskEdge called for Style ID: %d", styleId);
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle",
                           "  Mask Edge: thickness=%d, type=%d, color=0x%08X",
                           maskEdgeThickness, maskEdgeType, maskEdgeColor);

        // Call the Yolo class implementation, passing all parameters
        return g_yolo->setStyleParameters(styleId, lineThickness, lineType, boxAlpha, boxColor,
                                        fontSize, fontType, textColor, textStyle, fullTextBackground,
                                        maskAlpha, maskContrast,
                                        maskEdgeThickness, maskEdgeType, maskEdgeColor);
    }
    return JNI_FALSE;
}
}
