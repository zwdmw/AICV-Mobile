/*
 * note: 这里只是声明接口，用于与 C++ 层通过 JNI (Java Native Interface) 进行交互
 * 支持多种YOLO模型，包括YOLOv5、YOLOv7、YOLOv8等（不支持YOLOx系列）
 */
package com.gyq.yolov8;

import android.content.res.AssetManager;
import android.view.Surface;
import android.util.Log;

public class Yolov8Ncnn {
    // 数据集类型常量
    public static final int DATASET_COCO = 0;
    public static final int DATASET_OIV = 1;
    
    // mgr 是 AssetManager 用来访问应用的资源文件, modelName 是模型的名称，cpugpu 指定使用 CPU 还是 GPU
    public native boolean loadModel(AssetManager mgr, String modelName, int cpugpu);
    // facing 指定打开摄像头的方向
    public native boolean openCamera(int facing);
    // 关闭摄像头
    public native boolean closeCamera();
    // 数 surface 是输出图像的显示界面，是一个 SurfaceView 控件
    public native boolean setOutputWindow(Surface surface);
    // 返回当前帧中检测到的物体数量
    public native int getDetectionCount();
    // 返回应用过滤后的物体数量
    public native int getFilteredDetectionCount();
    // 设置启用的类别ID列表，传递null表示启用所有类别
    public native boolean setEnabledClassIds(int[] classIds);
    // 设置检测阈值
    public native boolean setDetectionThreshold(float threshold);
    // 添加设置掩码阈值的本地方法
    public native boolean setMaskThreshold(float threshold);
    // 添加设置绘制样式的本地方法
    public native void setDetectionStyle(int style);
    // 添加控制跟踪功能的方法
    public native boolean setEnableTracking(boolean enable);
    // 添加控制分割掩码跟踪功能的方法
    public native boolean setEnableMaskTracking(boolean enable);
    // 获取当前模型使用的数据集类型
    public native int getDatasetType();

    // 获取完整的OIV类别名称列表
    public native String[] getOIVClassNames();

    // Updated native method to set all tracking parameters including trajectory visualization
    public native boolean setTrackingParams(float iouThreshold, int maxAge, int minHits,
                                          float kalmanProcessNoise, float kalmanMeasurementNoise,
                                          boolean enableMaskTracking, boolean enableTrajectoryViz,
                                          int trajectoryLength, float trajectoryThickness, 
                                          int trajectoryColorMode);
                                          
    // 更新后的设置轨迹参数方法，包含连续跟踪模式
    public native boolean setTrackingParamsWithContinuous(float iouThreshold, int maxAge, int minHits,
                                          float kalmanProcessNoise, float kalmanMeasurementNoise,
                                          boolean enableMaskTracking, boolean enableTrajectoryViz,
                                          int trajectoryLength, float trajectoryThickness, 
                                          int trajectoryColorMode, boolean enableContinuousTracking,
                                          float predictionThreshold, int maxPredictionFrames,
                                          int predictionLineType);

    // 更新后的设置轨迹参数方法，包含跟踪模式
    public native boolean setTrackingParamsWithMode(float iouThreshold, int maxAge, int minHits,
                                          float kalmanProcessNoise, float kalmanMeasurementNoise,
                                          boolean enableMaskTracking, boolean enableTrajectoryViz,
                                          int trajectoryLength, float trajectoryThickness, 
                                          int trajectoryColorMode, boolean enableContinuousTracking,
                                          float predictionThreshold, int maxPredictionFrames,
                                          int predictionLineType, int trackingMode,
                                          int motionSubMode);

    // 带空间分布模式的跟踪参数设置
    public native boolean setTrackingParamsWithModes(float iouThreshold, int maxAge, int minHits,
                                          float kalmanProcessNoise, float kalmanMeasurementNoise,
                                          boolean enableMaskTracking, boolean enableTrajectoryViz,
                                          int trajectoryLength, float trajectoryThickness, 
                                          int trajectoryColorMode, boolean enableContinuousTracking,
                                          float predictionThreshold, int maxPredictionFrames,
                                          int predictionLineType, int trackingMode,
                                          int motionSubMode, int spatialDistribution);

    // 新增：加载类别文件（从assets目录）
    public native boolean loadClassesFromFile(AssetManager mgr, String modelName);
    
    // 新增：清除类别过滤
    public native boolean clearClassFilter();
    
    // 新增：检查类别过滤是否启用
    public native boolean isClassFilteringEnabled();

    // Get the current FPS value
    public native int getCurrentFPS();
    
    // 新增：设置中文类别名称
    public native boolean setChineseClassNames(String[] chineseNames);
    
    // 新增：设置样式参数
    public native boolean setStyleParameters(int styleId, int lineThickness, int lineType, 
                                            float boxAlpha, int boxColor, float fontSize, 
                                            int fontType, int textColor, int textStyle,
                                            boolean fullTextBackground,
                                            float maskAlpha, float maskContrast);

    // 新增：设置样式参数（包含掩码边缘设置）
    public native boolean setStyleParametersWithMaskEdge(int styleId, int lineThickness, int lineType, 
                                            float boxAlpha, int boxColor, float fontSize, 
                                            int fontType, int textColor, int textStyle,
                                            boolean fullTextBackground,
                                            float maskAlpha, float maskContrast,
                                            int maskEdgeThickness, int maskEdgeType, int maskEdgeColor);

    // Get style parameters method
    public native StyleParameters getStyleParameters(int styleId);
    
    // Reset style parameters to default values
    public native boolean resetStyleParameters(int styleId);

    // 通过JNI，这些声明成 public native 的方法可以用C++方法实现
    static {
        System.loadLibrary("yolov8ncnn");
    }
}
