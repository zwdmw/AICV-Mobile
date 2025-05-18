#include "ndkcamera.h"

#include <string>

#include <android/log.h>

#include <opencv2/core/core.hpp>

#include "mat.h"

// Import the camera active flag from yolov8ncnn.cpp
extern std::atomic<bool> g_camera_active;

static void onDisconnected(void* context, ACameraDevice* device)
{
    __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "onDisconnected %p", device);
}

static void onError(void* context, ACameraDevice* device, int error)
{
    __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "onError %p %d", device, error);
}

static void onImageAvailable(void* context, AImageReader* reader)
{
    // Add debug logging - 添加调试日志
    __android_log_print(ANDROID_LOG_DEBUG, "NdkCamera", "onImageAvailable %p", reader);

    // Check if camera is still active
    if (!g_camera_active.load()) {
        __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "Camera inactive, skipping image processing");
        return;
    }
    
    AImage* image = 0;
    media_status_t status = AImageReader_acquireLatestImage(reader, &image);

    if (status != AMEDIA_OK)
    {
        // error
        __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "AImageReader_acquireLatestImage failed with status %d", status);
        return;
    }

    if (image == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Acquired image is null");
        return;
    }

    int32_t format;
    AImage_getFormat(image, &format);
    
    __android_log_print(ANDROID_LOG_DEBUG, "NdkCamera", "Image format: %d (YUV_420_888=%d)", 
                        format, AIMAGE_FORMAT_YUV_420_888);

    // assert format == AIMAGE_FORMAT_YUV_420_888
    if (format != AIMAGE_FORMAT_YUV_420_888) {
        __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Unexpected image format: %d", format);
        AImage_delete(image);
        return;
    }

    int32_t width = 0;
    int32_t height = 0;
    AImage_getWidth(image, &width);
    AImage_getHeight(image, &height);
    
    __android_log_print(ANDROID_LOG_DEBUG, "NdkCamera", "Image dimensions: %dx%d", width, height);

    int32_t y_pixelStride = 0;
    int32_t u_pixelStride = 0;
    int32_t v_pixelStride = 0;
    AImage_getPlanePixelStride(image, 0, &y_pixelStride);
    AImage_getPlanePixelStride(image, 1, &u_pixelStride);
    AImage_getPlanePixelStride(image, 2, &v_pixelStride);

    int32_t y_rowStride = 0;
    int32_t u_rowStride = 0;
    int32_t v_rowStride = 0;
    AImage_getPlaneRowStride(image, 0, &y_rowStride);
    AImage_getPlaneRowStride(image, 1, &u_rowStride);
    AImage_getPlaneRowStride(image, 2, &v_rowStride);

    uint8_t* y_data = 0;
    uint8_t* u_data = 0;
    uint8_t* v_data = 0;
    int y_len = 0;
    int u_len = 0;
    int v_len = 0;
    AImage_getPlaneData(image, 0, &y_data, &y_len);
    AImage_getPlaneData(image, 1, &u_data, &u_len);
    AImage_getPlaneData(image, 2, &v_data, &v_len);

    if (u_data == v_data + 1 && v_data == y_data + width * height && y_pixelStride == 1 && u_pixelStride == 2 && v_pixelStride == 2 && y_rowStride == width && u_rowStride == width && v_rowStride == width)
    {
        // already nv21  :)
        ((NdkCamera*)context)->on_image((unsigned char*)y_data, (int)width, (int)height);
    }
    else
    {
        // construct nv21
        unsigned char* nv21 = new unsigned char[width * height + width * height / 2];
        {
            // Y
            unsigned char* yptr = nv21;
            for (int y=0; y<height; y++)
            {
                const unsigned char* y_data_ptr = y_data + y_rowStride * y;
                for (int x=0; x<width; x++)
                {
                    yptr[0] = y_data_ptr[0];
                    yptr++;
                    y_data_ptr += y_pixelStride;
                }
            }

            // UV
            unsigned char* uvptr = nv21 + width * height;
            for (int y=0; y<height/2; y++)
            {
                const unsigned char* v_data_ptr = v_data + v_rowStride * y;
                const unsigned char* u_data_ptr = u_data + u_rowStride * y;
                for (int x=0; x<width/2; x++)
                {
                    uvptr[0] = v_data_ptr[0];
                    uvptr[1] = u_data_ptr[0];
                    uvptr += 2;
                    v_data_ptr += v_pixelStride;
                    u_data_ptr += u_pixelStride;
                }
            }
        }

        ((NdkCamera*)context)->on_image((unsigned char*)nv21, (int)width, (int)height);

        delete[] nv21;
    }

    AImage_delete(image);
}

static void onSessionActive(void* context, ACameraCaptureSession *session)
{
    __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "onSessionActive %p", session);
}

static void onSessionReady(void* context, ACameraCaptureSession *session)
{
    __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "onSessionReady %p", session);
}

static void onSessionClosed(void* context, ACameraCaptureSession *session)
{
    __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "onSessionClosed %p", session);
}

void onCaptureFailed(void* context, ACameraCaptureSession* session, ACaptureRequest* request, ACameraCaptureFailure* failure)
{
    __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "onCaptureFailed %p %p %p", session, request, failure);
}

void onCaptureSequenceCompleted(void* context, ACameraCaptureSession* session, int sequenceId, int64_t frameNumber)
{
    __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "onCaptureSequenceCompleted %p %d %ld", session, sequenceId, frameNumber);
}

void onCaptureSequenceAborted(void* context, ACameraCaptureSession* session, int sequenceId)
{
    __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "onCaptureSequenceAborted %p %d", session, sequenceId);
}

void onCaptureCompleted(void* context, ACameraCaptureSession* session, ACaptureRequest* request, const ACameraMetadata* result)
{
//     __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "onCaptureCompleted %p %p %p", session, request, result);
}

NdkCamera::NdkCamera()
{
    camera_facing = 0;
    camera_orientation = 0;

    camera_manager = 0;
    camera_device = 0;
    image_reader = 0;
    image_reader_surface = 0;
    image_reader_target = 0;
    capture_request = 0;
    capture_session_output_container = 0;
    capture_session_output = 0;
    capture_session = 0;


    // setup imagereader and its surface
    {
        // Changed from fixed 640x480 to dynamic resolution that works better on most devices
        int width = 1280;
        int height = 720;
        
        __android_log_print(ANDROID_LOG_INFO, "NdkCamera", "Creating image reader with size %dx%d", width, height);
        
        media_status_t status = AImageReader_new(width, height, AIMAGE_FORMAT_YUV_420_888, /*maxImages*/2, &image_reader);
        
        if (status != AMEDIA_OK) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to create image reader, status=%d", status);
            return;
        }

        if (!image_reader) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Image reader is null after creation");
            return;
        }

        AImageReader_ImageListener listener;
        listener.context = this;
        listener.onImageAvailable = onImageAvailable;

        status = AImageReader_setImageListener(image_reader, &listener);
        if (status != AMEDIA_OK) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to set image listener, status=%d", status);
            return;
        }

        // 获取AImageReader窗口，并立即检查结果
        status = AImageReader_getWindow(image_reader, &image_reader_surface);
        if (status != AMEDIA_OK) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to get image reader window, status=%d", status);
            return;
        }

        if (!image_reader_surface) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "image_reader_surface is null after creation");
            return;
        }

        __android_log_print(ANDROID_LOG_INFO, "NdkCamera", "Successfully created image_reader_surface: %p", image_reader_surface);
        
        // 先释放旧的，再重新获取（防止潜在的资源泄漏）
        ANativeWindow_release(image_reader_surface);        
        ANativeWindow_acquire(image_reader_surface);
    }
}

NdkCamera::~NdkCamera()
{
    close();

    if (image_reader)
    {
        AImageReader_delete(image_reader);
        image_reader = 0;
    }

    if (image_reader_surface)
    {
        ANativeWindow_release(image_reader_surface);
        image_reader_surface = 0;
    }

    if (capture_session_output)
    {
        ACaptureSessionOutput_free(capture_session_output);
        capture_session_output = 0;
    }
    
    if (camera_manager)
    {
        // Only release camera_manager in destructor, not in close
        // as it's a shared resource that might be reused
        // Just log that we're keeping it
        __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "Keeping camera_manager alive for reuse");
    }
}

int NdkCamera::open(int _camera_facing)
{
    __android_log_print(ANDROID_LOG_INFO, "NdkCamera", "open - starting camera open process");

    camera_facing = _camera_facing;

    // 如果image_reader为null，需要重新初始化
    if (image_reader == 0) {
        __android_log_print(ANDROID_LOG_INFO, "NdkCamera", "image_reader is null, recreating it");
        
        // 重新创建image_reader和surface
        int width = 1280;
        int height = 720;
        
        __android_log_print(ANDROID_LOG_INFO, "NdkCamera", "Creating image reader with size %dx%d", width, height);
        
        media_status_t status = AImageReader_new(width, height, AIMAGE_FORMAT_YUV_420_888, /*maxImages*/2, &image_reader);
        
        if (status != AMEDIA_OK || !image_reader) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to create image reader in open(), status=%d", status);
            return -1;
        }

        AImageReader_ImageListener listener;
        listener.context = this;
        listener.onImageAvailable = onImageAvailable;

        status = AImageReader_setImageListener(image_reader, &listener);
        if (status != AMEDIA_OK) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to set image listener in open(), status=%d", status);
            return -1;
        }

        status = AImageReader_getWindow(image_reader, &image_reader_surface);
        if (status != AMEDIA_OK || !image_reader_surface) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to get image reader window in open(), status=%d", status);
            return -1;
        }

        ANativeWindow_acquire(image_reader_surface);
        __android_log_print(ANDROID_LOG_INFO, "NdkCamera", "Successfully recreated image_reader and surface in open()");
    }
    
    // 确保image_reader_surface是有效的
    if (!image_reader_surface) {
        __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "image_reader_surface is null, trying to recreate");
        
        // 如果image_reader是有效的，尝试重新获取surface
        if (image_reader) {
            media_status_t status = AImageReader_getWindow(image_reader, &image_reader_surface);
            if (status != AMEDIA_OK || !image_reader_surface) {
                __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to recreate image_reader_surface, status=%d", status);
                return -1;
            }
            ANativeWindow_acquire(image_reader_surface);
            __android_log_print(ANDROID_LOG_INFO, "NdkCamera", "Successfully recreated image_reader_surface: %p", image_reader_surface);
        } else {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "image_reader is null, cannot recreate surface");
            return -1;
        }
    } else {
        __android_log_print(ANDROID_LOG_INFO, "NdkCamera", "Using existing image_reader_surface: %p", image_reader_surface);
    }

    camera_manager = ACameraManager_create();
    if (!camera_manager) {
        __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to create camera manager");
        return -1;
    }

    // find front camera
    std::string camera_id;
    {
        ACameraIdList* camera_id_list = 0;
        camera_status_t status = ACameraManager_getCameraIdList(camera_manager, &camera_id_list);
        if (status != ACAMERA_OK) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to get camera ID list, status=%d", status);
            return -1;
        }

        __android_log_print(ANDROID_LOG_INFO, "NdkCamera", "Found %d cameras", camera_id_list->numCameras);
        
        if (camera_id_list->numCameras <= 0) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "No cameras found");
            ACameraManager_deleteCameraIdList(camera_id_list);
            return -1;
        }

        for (int i = 0; i < camera_id_list->numCameras; ++i)
        {
            const char* id = camera_id_list->cameraIds[i];
            ACameraMetadata* camera_metadata = 0;
            ACameraManager_getCameraCharacteristics(camera_manager, id, &camera_metadata);

            // query faceing
            acamera_metadata_enum_android_lens_facing_t facing = ACAMERA_LENS_FACING_FRONT;
            {
                ACameraMetadata_const_entry e = { 0 };
                ACameraMetadata_getConstEntry(camera_metadata, ACAMERA_LENS_FACING, &e);
                facing = (acamera_metadata_enum_android_lens_facing_t)e.data.u8[0];
            }

            if (camera_facing == 0 && facing != ACAMERA_LENS_FACING_FRONT)
            {
                ACameraMetadata_free(camera_metadata);
                continue;
            }

            if (camera_facing == 1 && facing != ACAMERA_LENS_FACING_BACK)
            {
                ACameraMetadata_free(camera_metadata);
                continue;
            }

            camera_id = id;

            // query orientation
            int orientation = 0;
            {
                ACameraMetadata_const_entry e = { 0 };
                ACameraMetadata_getConstEntry(camera_metadata, ACAMERA_SENSOR_ORIENTATION, &e);

                orientation = (int)e.data.i32[0];
            }

            camera_orientation = orientation;

            ACameraMetadata_free(camera_metadata);

            break;
        }

        ACameraManager_deleteCameraIdList(camera_id_list);
    }

    if (camera_id.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to find camera with facing %d", camera_facing);
        return -1;
    }

    __android_log_print(ANDROID_LOG_INFO, "NdkCamera", "Using camera %s with orientation %d", camera_id.c_str(), camera_orientation);

    // open camera
    {
        ACameraDevice_StateCallbacks camera_device_state_callbacks;
        camera_device_state_callbacks.context = this;
        camera_device_state_callbacks.onDisconnected = onDisconnected;
        camera_device_state_callbacks.onError = onError;

        ACameraManager_openCamera(camera_manager, camera_id.c_str(), &camera_device_state_callbacks, &camera_device);
    }

    // 在创建捕获请求前再次确认surface是有效的
    if (!image_reader_surface) {
        __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "image_reader_surface became null before creating capture request");
        return -1;
    }

    // capture request
    {
        ACameraDevice_createCaptureRequest(camera_device, TEMPLATE_PREVIEW, &capture_request);

        // 创建输出目标前输出surface的指针值，便于调试
        __android_log_print(ANDROID_LOG_INFO, "NdkCamera", "Creating output target with surface: %p", image_reader_surface);
        
        // 创建目标前确认surface有效
        camera_status_t status = ACameraOutputTarget_create(image_reader_surface, &image_reader_target);
        if (status != ACAMERA_OK || !image_reader_target) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to create camera output target, status=%d", status);
            return -1;
        }
        
        // 添加目标前确认目标有效
        status = ACaptureRequest_addTarget(capture_request, image_reader_target);
        if (status != ACAMERA_OK) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to add target to capture request, status=%d", status);
            return -1;
        }
    }

    // capture session - 添加错误检查
    {
        ACameraCaptureSession_stateCallbacks camera_capture_session_state_callbacks;
        camera_capture_session_state_callbacks.context = this;
        camera_capture_session_state_callbacks.onActive = onSessionActive;
        camera_capture_session_state_callbacks.onReady = onSessionReady;
        camera_capture_session_state_callbacks.onClosed = onSessionClosed;

        camera_status_t status = ACaptureSessionOutputContainer_create(&capture_session_output_container);
        if (status != ACAMERA_OK) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to create capture session output container, status=%d", status);
            return -1;
        }

        status = ACaptureSessionOutput_create(image_reader_surface, &capture_session_output);
        if (status != ACAMERA_OK) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to create capture session output, status=%d", status);
            return -1;
        }

        status = ACaptureSessionOutputContainer_add(capture_session_output_container, capture_session_output);
        if (status != ACAMERA_OK) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to add output to container, status=%d", status);
            return -1;
        }

        status = ACameraDevice_createCaptureSession(camera_device, capture_session_output_container, &camera_capture_session_state_callbacks, &capture_session);
        if (status != ACAMERA_OK) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to create capture session, status=%d", status);
            return -1;
        }

        ACameraCaptureSession_captureCallbacks camera_capture_session_capture_callbacks;
        camera_capture_session_capture_callbacks.context = this;
        camera_capture_session_capture_callbacks.onCaptureStarted = 0;
        camera_capture_session_capture_callbacks.onCaptureProgressed = 0;
        camera_capture_session_capture_callbacks.onCaptureCompleted = onCaptureCompleted;
        camera_capture_session_capture_callbacks.onCaptureFailed = onCaptureFailed;
        camera_capture_session_capture_callbacks.onCaptureSequenceCompleted = onCaptureSequenceCompleted;
        camera_capture_session_capture_callbacks.onCaptureSequenceAborted = onCaptureSequenceAborted;
        camera_capture_session_capture_callbacks.onCaptureBufferLost = 0;

        status = ACameraCaptureSession_setRepeatingRequest(capture_session, &camera_capture_session_capture_callbacks, 1, &capture_request, nullptr);
        if (status != ACAMERA_OK) {
            __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Failed to set repeating request, status=%d", status);
            return -1;
        }
    }

    __android_log_print(ANDROID_LOG_INFO, "NdkCamera", "Camera successfully opened and configured");
    return 0;
}

void NdkCamera::close()
{
    __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "close");

    if (capture_session)
    {
        ACameraCaptureSession_stopRepeating(capture_session);
        ACameraCaptureSession_close(capture_session);
        capture_session = 0;
    }

    if (camera_device)
    {
        ACameraDevice_close(camera_device);
        camera_device = 0;
    }

    if (image_reader)
    {
        AImageReader_delete(image_reader);
        image_reader = 0;
    }

    if (image_reader_surface)
    {
        ANativeWindow_release(image_reader_surface);
        image_reader_surface = 0;
    }

    if (image_reader_target)
    {
        ACameraOutputTarget_free(image_reader_target);
        image_reader_target = 0;
    }

    if (capture_request)
    {
        ACaptureRequest_free(capture_request);
        capture_request = 0;
    }

    if (capture_session_output_container)
    {
        ACaptureSessionOutputContainer_free(capture_session_output_container);
        capture_session_output_container = 0;
    }

    if (capture_session_output)
    {
        ACaptureSessionOutput_free(capture_session_output);
        capture_session_output = 0;
    }
}

void NdkCamera::on_image(const cv::Mat& rgb) const
{
}

void NdkCamera::on_image(const unsigned char* nv21, int nv21_width, int nv21_height) const
{
    // rotate nv21
    int w = 0;
    int h = 0;
    int rotate_type = 0;
    {
        if (camera_orientation == 0)
        {
            w = nv21_width;
            h = nv21_height;
            rotate_type = camera_facing == 0 ? 2 : 1;
        }
        if (camera_orientation == 90)
        {
            w = nv21_height;
            h = nv21_width;
            rotate_type = camera_facing == 0 ? 5 : 6;
        }
        if (camera_orientation == 180)
        {
            w = nv21_width;
            h = nv21_height;
            rotate_type = camera_facing == 0 ? 4 : 3;
        }
        if (camera_orientation == 270)
        {
            w = nv21_height;
            h = nv21_width;
            rotate_type = camera_facing == 0 ? 7 : 8;
        }
    }

    cv::Mat nv21_rotated(h + h / 2, w, CV_8UC1);
    ncnn::kanna_rotate_yuv420sp(nv21, nv21_width, nv21_height, nv21_rotated.data, w, h, rotate_type);

    // nv21_rotated to rgb
    cv::Mat rgb(h, w, CV_8UC3);
    ncnn::yuv420sp2rgb(nv21_rotated.data, w, h, rgb.data);

    on_image(rgb);
}

static const int NDKCAMERAWINDOW_ID = 233;

NdkCameraWindow::NdkCameraWindow() : NdkCamera()
{
    sensor_manager = 0;
    sensor_event_queue = 0;
    accelerometer_sensor = 0;
    win = 0;

    accelerometer_orientation = 0;

    // sensor
    sensor_manager = ASensorManager_getInstance();

    accelerometer_sensor = ASensorManager_getDefaultSensor(sensor_manager, ASENSOR_TYPE_ACCELEROMETER);
}

NdkCameraWindow::~NdkCameraWindow()
{
    if (accelerometer_sensor)
    {
        ASensorEventQueue_disableSensor(sensor_event_queue, accelerometer_sensor);
        accelerometer_sensor = 0;
    }

    if (sensor_event_queue)
    {
        ASensorManager_destroyEventQueue(sensor_manager, sensor_event_queue);
        sensor_event_queue = 0;
    }

    if (win)
    {
        ANativeWindow_release(win);
    }
}

void NdkCameraWindow::set_window(ANativeWindow* _win)
{
    if (!_win) {
        __android_log_print(ANDROID_LOG_ERROR, "NdkCameraWindow", "Trying to set null window");
        return;
    }

    __android_log_print(ANDROID_LOG_INFO, "NdkCameraWindow", "Setting window from %p to %p", win, _win);
    
    if (win)
    {
        ANativeWindow_release(win);
        win = nullptr;
    }

    win = _win;
    ANativeWindow_acquire(win);
    
    // 检查窗口大小和格式，确保窗口有效
    int32_t width = ANativeWindow_getWidth(win);
    int32_t height = ANativeWindow_getHeight(win);
    int32_t format = ANativeWindow_getFormat(win);
    
    __android_log_print(ANDROID_LOG_INFO, "NdkCameraWindow", "New window info - width: %d, height: %d, format: %d", 
                        width, height, format);
    
    if (width <= 0 || height <= 0) {
        __android_log_print(ANDROID_LOG_ERROR, "NdkCameraWindow", "Invalid window dimensions: %dx%d", width, height);
        // 不返回错误，继续尝试使用
    }
}

void NdkCameraWindow::on_image_render(cv::Mat& rgb) const
{
}

void NdkCameraWindow::on_image(const unsigned char* nv21, int nv21_width, int nv21_height) const
{
    // Check if camera is still active
    if (!g_camera_active.load()) {
        __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "Camera inactive, skipping on_image processing");
        return;
    }

    // Make sure window is valid before proceeding
    if (!win) {
        __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "Output window is null, cannot render");
        return;
    }

    // Log window dimensions for debugging
    int win_width = ANativeWindow_getWidth(win);
    int win_height = ANativeWindow_getHeight(win);
    __android_log_print(ANDROID_LOG_DEBUG, "NdkCamera", "Window dimensions: %dx%d", win_width, win_height);
    
    // Log camera orientation info
    __android_log_print(ANDROID_LOG_DEBUG, "NdkCamera", "Camera info: facing=%d, orientation=%d, accelerometer_orientation=%d", 
                      camera_facing, camera_orientation, accelerometer_orientation);

    // resolve orientation from camera_orientation and accelerometer_sensor
    {
        if (!sensor_event_queue)
        {
            sensor_event_queue = ASensorManager_createEventQueue(sensor_manager, ALooper_prepare(ALOOPER_PREPARE_ALLOW_NON_CALLBACKS), NDKCAMERAWINDOW_ID, 0, 0);

            ASensorEventQueue_enableSensor(sensor_event_queue, accelerometer_sensor);
        }

        int id = ALooper_pollAll(0, 0, 0, 0);
        if (id == NDKCAMERAWINDOW_ID)
        {
            ASensorEvent e[8];
            ssize_t num_event = 0;
            while (ASensorEventQueue_hasEvents(sensor_event_queue) == 1)
            {
                num_event = ASensorEventQueue_getEvents(sensor_event_queue, e, 8);
                if (num_event < 0)
                    break;
            }

            if (num_event > 0)
            {
                float acceleration_x = e[num_event - 1].acceleration.x;
                float acceleration_y = e[num_event - 1].acceleration.y;
                float acceleration_z = e[num_event - 1].acceleration.z;
//                 __android_log_print(ANDROID_LOG_WARN, "NdkCameraWindow", "x = %f, y = %f, z = %f", x, y, z);

                if (acceleration_y > 7)
                {
                    accelerometer_orientation = 0;
                }
                if (acceleration_x < -7)
                {
                    accelerometer_orientation = 90;
                }
                if (acceleration_y < -7)
                {
                    accelerometer_orientation = 180;
                }
                if (acceleration_x > 7)
                {
                    accelerometer_orientation = 270;
                }
            }
        }
    }

    // roi crop and rotate nv21
    int nv21_roi_x = 0;
    int nv21_roi_y = 0;
    int nv21_roi_w = 0;
    int nv21_roi_h = 0;
    int roi_x = 0;
    int roi_y = 0;
    int roi_w = 0;
    int roi_h = 0;
    int rotate_type = 0;
    int render_w = 0;
    int render_h = 0;
    int render_rotate_type = 0;
    {
        int win_w = ANativeWindow_getWidth(win);
        int win_h = ANativeWindow_getHeight(win);

        if (accelerometer_orientation == 90 || accelerometer_orientation == 270)
        {
            std::swap(win_w, win_h);
        }

        const int final_orientation = (camera_orientation + accelerometer_orientation) % 360;

        if (final_orientation == 0 || final_orientation == 180)
        {
            if (win_w * nv21_height > win_h * nv21_width)
            {
                roi_w = nv21_width;
                roi_h = (nv21_width * win_h / win_w) / 2 * 2;
                roi_x = 0;
                roi_y = ((nv21_height - roi_h) / 2) / 2 * 2;
            }
            else
            {
                roi_h = nv21_height;
                roi_w = (nv21_height * win_w / win_h) / 2 * 2;
                roi_x = ((nv21_width - roi_w) / 2) / 2 * 2;
                roi_y = 0;
            }

            nv21_roi_x = roi_x;
            nv21_roi_y = roi_y;
            nv21_roi_w = roi_w;
            nv21_roi_h = roi_h;
        }
        if (final_orientation == 90 || final_orientation == 270)
        {
            if (win_w * nv21_width > win_h * nv21_height)
            {
                roi_w = nv21_height;
                roi_h = (nv21_height * win_h / win_w) / 2 * 2;
                roi_x = 0;
                roi_y = ((nv21_width - roi_h) / 2) / 2 * 2;
            }
            else
            {
                roi_h = nv21_width;
                roi_w = (nv21_width * win_w / win_h) / 2 * 2;
                roi_x = ((nv21_height - roi_w) / 2) / 2 * 2;
                roi_y = 0;
            }

            nv21_roi_x = roi_y;
            nv21_roi_y = roi_x;
            nv21_roi_w = roi_h;
            nv21_roi_h = roi_w;
        }

        if (camera_facing == 0)
        {
            if (camera_orientation == 0 && accelerometer_orientation == 0)
            {
                rotate_type = 2;
            }
            if (camera_orientation == 0 && accelerometer_orientation == 90)
            {
                rotate_type = 7;
            }
            if (camera_orientation == 0 && accelerometer_orientation == 180)
            {
                rotate_type = 4;
            }
            if (camera_orientation == 0 && accelerometer_orientation == 270)
            {
                rotate_type = 5;
            }
            if (camera_orientation == 90 && accelerometer_orientation == 0)
            {
                rotate_type = 5;
            }
            if (camera_orientation == 90 && accelerometer_orientation == 90)
            {
                rotate_type = 2;
            }
            if (camera_orientation == 90 && accelerometer_orientation == 180)
            {
                rotate_type = 7;
            }
            if (camera_orientation == 90 && accelerometer_orientation == 270)
            {
                rotate_type = 4;
            }
            if (camera_orientation == 180 && accelerometer_orientation == 0)
            {
                rotate_type = 4;
            }
            if (camera_orientation == 180 && accelerometer_orientation == 90)
            {
                rotate_type = 5;
            }
            if (camera_orientation == 180 && accelerometer_orientation == 180)
            {
                rotate_type = 2;
            }
            if (camera_orientation == 180 && accelerometer_orientation == 270)
            {
                rotate_type = 7;
            }
            if (camera_orientation == 270 && accelerometer_orientation == 0)
            {
                rotate_type = 7;
            }
            if (camera_orientation == 270 && accelerometer_orientation == 90)
            {
                rotate_type = 4;
            }
            if (camera_orientation == 270 && accelerometer_orientation == 180)
            {
                rotate_type = 5;
            }
            if (camera_orientation == 270 && accelerometer_orientation == 270)
            {
                rotate_type = 2;
            }
        }
        else
        {
            if (final_orientation == 0)
            {
                rotate_type = 1;
            }
            if (final_orientation == 90)
            {
                rotate_type = 6;
            }
            if (final_orientation == 180)
            {
                rotate_type = 3;
            }
            if (final_orientation == 270)
            {
                rotate_type = 8;
            }
        }

        if (accelerometer_orientation == 0)
        {
            render_w = roi_w;
            render_h = roi_h;
            render_rotate_type = 1;
        }
        if (accelerometer_orientation == 90)
        {
            render_w = roi_h;
            render_h = roi_w;
            render_rotate_type = 8;
        }
        if (accelerometer_orientation == 180)
        {
            render_w = roi_w;
            render_h = roi_h;
            render_rotate_type = 3;
        }
        if (accelerometer_orientation == 270)
        {
            render_w = roi_h;
            render_h = roi_w;
            render_rotate_type = 6;
        }
    }

    // crop and rotate nv21
    cv::Mat nv21_croprotated(roi_h + roi_h / 2, roi_w, CV_8UC1);
    {
        const unsigned char* srcY = nv21 + nv21_roi_y * nv21_width + nv21_roi_x;
        unsigned char* dstY = nv21_croprotated.data;
        ncnn::kanna_rotate_c1(srcY, nv21_roi_w, nv21_roi_h, nv21_width, dstY, roi_w, roi_h, roi_w, rotate_type);

        const unsigned char* srcUV = nv21 + nv21_width * nv21_height + nv21_roi_y * nv21_width / 2 + nv21_roi_x;
        unsigned char* dstUV = nv21_croprotated.data + roi_w * roi_h;
        ncnn::kanna_rotate_c2(srcUV, nv21_roi_w / 2, nv21_roi_h / 2, nv21_width, dstUV, roi_w / 2, roi_h / 2, roi_w, rotate_type);
    }

    // nv21_croprotated to rgb
    cv::Mat rgb(roi_h, roi_w, CV_8UC3);
    ncnn::yuv420sp2rgb(nv21_croprotated.data, roi_w, roi_h, rgb.data);

    on_image_render(rgb);

    // rotate to native window orientation
    cv::Mat rgb_render(render_h, render_w, CV_8UC3);
    ncnn::kanna_rotate_c3(rgb.data, roi_w, roi_h, rgb_render.data, render_w, render_h, render_rotate_type);

    // Make sure window is valid before proceeding
    if (!win) {
        __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "Output window is null, skipping render");
        return;
    }

    // Check if camera is still active
    if (!g_camera_active.load()) {
        __android_log_print(ANDROID_LOG_WARN, "NdkCamera", "Camera inactive, skipping window render");
        return;
    }

    ANativeWindow_setBuffersGeometry(win, render_w, render_h, AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM);

    ANativeWindow_Buffer buf;
    int res = ANativeWindow_lock(win, &buf, NULL);
    if (res != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "ANativeWindow_lock failed with error %d", res);
        return;
    }

    // Check buffer format
    __android_log_print(ANDROID_LOG_DEBUG, "NdkCamera", "ANativeWindow buffer format: %d", buf.format);

    // scale to target size
    if (buf.format == AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM || buf.format == AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM)
    {
        for (int y = 0; y < render_h; y++)
        {
            const unsigned char* ptr = rgb_render.ptr<const unsigned char>(y);
            unsigned char* outptr = (unsigned char*)buf.bits + buf.stride * 4 * y;

            int x = 0;
#if __ARM_NEON
            for (; x + 7 < render_w; x += 8)
            {
                uint8x8x3_t _rgb = vld3_u8(ptr);
                uint8x8x4_t _rgba;
                _rgba.val[0] = _rgb.val[0];
                _rgba.val[1] = _rgb.val[1];
                _rgba.val[2] = _rgb.val[2];
                _rgba.val[3] = vdup_n_u8(255);
                vst4_u8(outptr, _rgba);

                ptr += 24;
                outptr += 32;
            }
#endif // __ARM_NEON
            for (; x < render_w; x++)
            {
                outptr[0] = ptr[0];
                outptr[1] = ptr[1];
                outptr[2] = ptr[2];
                outptr[3] = 255;

                ptr += 3;
                outptr += 4;
            }
        }
    }

    res = ANativeWindow_unlockAndPost(win);
    if (res != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "NdkCamera", "ANativeWindow_unlockAndPost failed with error %d", res);
    }
}
