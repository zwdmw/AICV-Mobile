#ifndef TRACKING_PARAMS_H
#define TRACKING_PARAMS_H

// 空间分布模式常量
static const int SPATIAL_SPARSE = 0;      // 稀疏模式
static const int SPATIAL_NORMAL = 1;      // 正常模式
static const int SPATIAL_DENSE = 2;       // 密集模式
static const int SPATIAL_VERY_DENSE = 3;  // 超密集模式

struct TrackingParams {
    // Association
    float iou_threshold = 0.3f;

    // Track lifecycle
    int max_age = 30;
    int min_hits = 3;

    // Kalman Filter (adjust defaults as needed based on KF tuning)
    float kalman_process_noise = 1e-2f;
    float kalman_measurement_noise = 1e-1f;

    // Mask Tracking
    bool enable_mask_tracking = true;
    
    // Trajectory Visualization
    bool enable_trajectory_viz = false;
    int trajectory_length = 20;     // Maximum length of trajectory to visualize
    float trajectory_thickness = 2.0f;  // Line thickness
    int trajectory_color_mode = 0;   // 0=single color, 1=fade out, 2=rainbow
    
    // Continuous Tracking Mode
    bool enable_continuous_tracking = false;  // 启用连续跟踪模式
    float prediction_probability_threshold = 0.6f;  // 预测概率阈值，低于此值停止显示
    int max_prediction_frames = 30;  // 最大预测帧数，超过此值停止预测
    int prediction_line_type = 1;  // 预测轨迹线型: 0=实线, 1=虚线
    
    // 跟踪模式
    int tracking_mode = 0;  // 0=稳定模式, 1=手持模式, 2=自定义模式
    
    // 手持模式特有参数
    bool enable_motion_compensation = false;  // 运动补偿
    float acceleration_weight = 0.2f;         // 加速度权重因子，用于手持场景
    int velocity_history_length = 5;          // 速度历史长度
    
    // 运动子模式
    int motion_submode = 4; // 0=Static, 1=CV, 2=CA, 3=Variable, 4=General
    
    // 空间分布模式
    int spatial_distribution = SPATIAL_NORMAL;
    
    // 特殊策略标志
    bool use_reid_features = false;       // 使用ReID特征
    bool use_group_management = false;    // 使用群组管理
    bool use_social_force_model = false;  // 使用社交力模型
    bool use_occlusion_handling = false;  // 使用遮挡处理

    // 构造函数：初始化默认参数
    TrackingParams() 
        : iou_threshold(0.30f),
          max_age(15),
          min_hits(2),
          kalman_process_noise(1e-2),
          kalman_measurement_noise(1e-1),
          enable_mask_tracking(false),
          enable_trajectory_viz(false),
          trajectory_length(5),
          trajectory_thickness(2),
          trajectory_color_mode(0),
          enable_continuous_tracking(false),
          prediction_probability_threshold(0.7f),
          max_prediction_frames(10),
          prediction_line_type(0),
          tracking_mode(0),
          motion_submode(0),
          spatial_distribution(0) {
    }
};

#endif // TRACKING_PARAMS_H 