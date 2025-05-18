#ifndef ENHANCED_TRACKER_H
#define ENHANCED_TRACKER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <deque>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
#include <android/log.h>
#include "Object.h"
#include "TrackingParams.h"

#define TAG "EnhancedTrackerKF"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// 带有卡尔曼滤波的跟踪器类
class EnhancedTracking {
public:
    // 跟踪对象结构体
    struct TrackingObject {
        int id;                         // 跟踪ID
        cv::Mat mask;                   // 当前对象的分割掩码
        std::deque<cv::Mat> mask_history; // 历史掩码
        std::vector<float> mask_coeffs;  // 掩码系数
        bool has_valid_mask;            // 是否有有效的分割掩码
        cv::Rect rect;                  // 当前边界框（由卡尔曼滤波更新）
        std::deque<cv::Rect> history;   // 历史位置 (观测值)
        int missing_count;              // 多少帧未被检测到
        int hit_count;                  // 连续多少帧被检测到
        int total_hits;                 // 总共被检测到的次数
        int age;                        // 年龄(帧数)
        int label;                      // 类别标签
        float confidence;               // 置信度 (最后一次观测)
        
        // 新增: 卡尔曼滤波器
        cv::KalmanFilter kalman;
        
        // 新增: 用于轨迹可视化的颜色
        cv::Scalar color;
        
        // 连续跟踪相关字段
        float prediction_probability;   // 预测位置的可信度概率
        bool is_prediction_only;        // 是否仅为预测状态(未观测到)
        
        // 手持模式相关变量
        bool enable_motion_compensation;  // 是否启用运动补偿
        float acceleration_weight;        // 加速度权重
        
        // 新增: 对外部跟踪参数的引用
        const TrackingParams& params_;
        
        // 构造函数
        TrackingObject(const cv::Rect& bbox, int track_id, int obj_label, float conf, const TrackingParams& current_params) 
            : id(track_id),
              missing_count(0),
              hit_count(1),
              total_hits(1),
              age(1),
              label(obj_label),
              confidence(conf),
              has_valid_mask(false),
              prediction_probability(1.0f),
              is_prediction_only(false),
              enable_motion_compensation(false),
              acceleration_weight(0.2f),
              params_(current_params) {
            
            history.push_back(bbox);
            rect = bbox;
            
            // 初始化颜色 (使用固定的颜色表)
            static const cv::Scalar colors[] = {
                cv::Scalar(0, 0, 255),     // 红色
                cv::Scalar(0, 255, 0),     // 绿色
                cv::Scalar(255, 0, 0),     // 蓝色
                cv::Scalar(0, 255, 255),   // 黄色
                cv::Scalar(255, 0, 255),   // 紫色
                cv::Scalar(255, 255, 0),   // 青色
                cv::Scalar(128, 0, 255),   // 粉色
                cv::Scalar(0, 128, 255),   // 橙色
                cv::Scalar(255, 128, 0),   // 浅蓝色
                cv::Scalar(128, 255, 0)    // 浅绿色
            };
            const int num_colors = sizeof(colors) / sizeof(colors[0]);
            color = colors[id % num_colors];
            
            // 初始化卡尔曼滤波器
            // 状态向量: [x, y, width, height, vx, vy, vw, vh]
            kalman.init(8, 4, 0);
            
            // 设置转移矩阵 (F) - 匀速模型
            cv::setIdentity(kalman.transitionMatrix);
            kalman.transitionMatrix.at<float>(0, 4) = 1.0f; // x = x + vx
            kalman.transitionMatrix.at<float>(1, 5) = 1.0f; // y = y + vy
            kalman.transitionMatrix.at<float>(2, 6) = 1.0f; // w = w + vw
            kalman.transitionMatrix.at<float>(3, 7) = 1.0f; // h = h + vh
            
            // 设置测量矩阵 (H) - 只测量位置和尺寸
            cv::setIdentity(kalman.measurementMatrix);
            kalman.measurementMatrix.at<float>(0, 4) = 0.0f; // 测量矩阵只关注前4个状态
            kalman.measurementMatrix.at<float>(0, 5) = 0.0f;
            kalman.measurementMatrix.at<float>(0, 6) = 0.0f;
            kalman.measurementMatrix.at<float>(0, 7) = 0.0f;
            kalman.measurementMatrix.at<float>(1, 4) = 0.0f;
            kalman.measurementMatrix.at<float>(1, 5) = 0.0f;
            kalman.measurementMatrix.at<float>(1, 6) = 0.0f;
            kalman.measurementMatrix.at<float>(1, 7) = 0.0f;
            kalman.measurementMatrix.at<float>(2, 4) = 0.0f;
            kalman.measurementMatrix.at<float>(2, 5) = 0.0f;
            kalman.measurementMatrix.at<float>(2, 6) = 0.0f;
            kalman.measurementMatrix.at<float>(2, 7) = 0.0f;
            kalman.measurementMatrix.at<float>(3, 4) = 0.0f;
            kalman.measurementMatrix.at<float>(3, 5) = 0.0f;
            kalman.measurementMatrix.at<float>(3, 6) = 0.0f;
            kalman.measurementMatrix.at<float>(3, 7) = 0.0f;

            // 设置过程噪声协方差矩阵 (Q)
            // 速度变化的不确定性设为较大值，位置和尺寸变化不确定性较小
            cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-2)); // 基础不确定性
            kalman.processNoiseCov.at<float>(4,4) = 1e-1; // vx 不确定性
            kalman.processNoiseCov.at<float>(5,5) = 1e-1; // vy 不确定性
            kalman.processNoiseCov.at<float>(6,6) = 1e-3; // vw 不确定性
            kalman.processNoiseCov.at<float>(7,7) = 1e-3; // vh 不确定性
            
            // 设置测量噪声协方差矩阵 (R)
            cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(1e-1)); // 测量不确定性
            
            // 设置后验错误协方差矩阵 (P)
            cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(1)); // 初始误差
            
            // 设置状态向量初始值 (x, y, width, height, 0, 0, 0, 0)
            kalman.statePost.at<float>(0) = bbox.x + bbox.width / 2.0f;  // 使用中心点 x
            kalman.statePost.at<float>(1) = bbox.y + bbox.height / 2.0f; // 使用中心点 y
            kalman.statePost.at<float>(2) = bbox.width;
            kalman.statePost.at<float>(3) = bbox.height;
            kalman.statePost.at<float>(4) = 0;  // vx
            kalman.statePost.at<float>(5) = 0;  // vy
            kalman.statePost.at<float>(6) = 0;  // vw
            kalman.statePost.at<float>(7) = 0;  // vh
            
            // 同时初始化rect为初始bbox
            rect = bbox;
        }
        
        // 计算质心 (基于卡尔曼滤波后的状态)
        cv::Point2f getCentroid() const {
            return cv::Point2f(kalman.statePost.at<float>(0), kalman.statePost.at<float>(1));
        }
        
        // 从矩形获取中心点
        cv::Point2f getCentroidFromRect(const cv::Rect& r) const {
            return cv::Point2f(r.x + r.width/2.0f, r.y + r.height/2.0f);
        }
        
        // 基于卡尔曼滤波器预测下一个位置
        cv::Rect predictNextPosition() {
            // 常规卡尔曼预测
            cv::Mat prediction = kalman.predict();
            
            // 从预测状态中提取位置和尺寸 (中心点x, y, w, h)
            float cx = prediction.at<float>(0);
            float cy = prediction.at<float>(1);
            float w = prediction.at<float>(2);
            float h = prediction.at<float>(3);
            
            // 手持模式特殊处理 - 加入运动补偿
            if (enable_motion_compensation && history.size() >= 3) {
                // 计算加速度（二阶差分）
                cv::Point2f p1 = getCentroidFromRect(history[history.size()-3]);
                cv::Point2f p2 = getCentroidFromRect(history[history.size()-2]);
                cv::Point2f p3 = getCentroidFromRect(history[history.size()-1]);
                
                cv::Point2f v1 = p2 - p1;
                cv::Point2f v2 = p3 - p2;
                cv::Point2f acc = v2 - v1;
                
                // 将加速度加权应用到预测中
                // 这里hardcode了加速度权重为0.2，实际应该使用params_中的值
                cx += acc.x * acceleration_weight;
                cy += acc.y * acceleration_weight;
            }
            
            // 根据运动子模式调整预测 (示例)
            if (params_.motion_submode == 0) { // Static (assuming 0 is static from Java)
                 // 对于确认的静态目标，可以强制预测速度为0，或者显著降低预测步长
                 // Example: Force position to be very close to previous state
                 cx = kalman.statePost.at<float>(0);
                 cy = kalman.statePost.at<float>(1);
                 // Or simply trust prediction less if it moved significantly? Needs tuning.
            }
            
            // 根据空间分布模式调整预测
            switch (params_.spatial_distribution) {
                case 0: // SPATIAL_SPARSE
                    // 稀疏场景中，可以允许更大的预测步长，更容易捕获快速移动对象
                    // cx, cy 和 w, h 保持卡尔曼预测值
                    break;
                    
                case 2: // SPATIAL_DENSE
                    // 密集场景中，需要更加保守的预测，减小步长以避免ID切换
                    {
                        // 速度分量缩减 30%
                        float vx = kalman.statePost.at<float>(4) * 0.7f;
                        float vy = kalman.statePost.at<float>(5) * 0.7f;
                        
                        // 使用缩减后的速度重新计算位置
                        cx = kalman.statePost.at<float>(0) + vx;
                        cy = kalman.statePost.at<float>(1) + vy;
                    }
                    break;
                    
                case 3: // SPATIAL_VERY_DENSE
                    // 超密集场景，极其保守的预测，几乎只依赖观测而非预测
                    {
                        // 速度分量缩减 50%
                        float vx = kalman.statePost.at<float>(4) * 0.5f;
                        float vy = kalman.statePost.at<float>(5) * 0.5f;
                        
                        // 使用缩减后的速度重新计算位置
                        cx = kalman.statePost.at<float>(0) + vx;
                        cy = kalman.statePost.at<float>(1) + vy;
                        
                        // 如果有历史记录，增加对最近观测的信任
                        if (!history.empty()) {
                            cv::Point2f recent = getCentroidFromRect(history.back());
                            // 预测位置向最近观测位置偏移 20%
                            cx = cx * 0.8f + recent.x * 0.2f;
                            cy = cy * 0.8f + recent.y * 0.2f;
                        }
                    }
                    break;
                    
                case 1: // SPATIAL_NORMAL (默认)
                default:
                    // 使用原始卡尔曼预测
                    break;
            }
            
            // 确保尺寸为正
            w = std::max(5.0f, w);
            h = std::max(5.0f, h);
            
            // 转换回左上角坐标 (x, y)
            float x = cx - w / 2.0f;
            float y = cy - h / 2.0f;
            
            // 更新内部rect（预测值）
            rect = cv::Rect(round(x), round(y), round(w), round(h));
            return rect;
        }
        
        // 计算运动方向向量 (基于速度状态)
        cv::Point2f getMotionDirection() const {
            float vx = kalman.statePost.at<float>(4);
            float vy = kalman.statePost.at<float>(5);
            cv::Point2f dir(vx, vy);
            
            // 归一化
            float norm = std::sqrt(dir.x*dir.x + dir.y*dir.y);
            if (norm > 1e-5) {
                dir.x /= norm;
                dir.y /= norm;
            }
            return dir;
        }
        
        // 更新跟踪器状态 (使用卡尔曼滤波器校正)
        void update(const cv::Rect& bbox, float conf) {
            // 添加到历史 (观测值)
            if (history.size() >= 10) { 
                history.pop_front();
            }
            history.push_back(bbox);
            
            // 准备测量值 (中心点x, y, w, h)
            cv::Mat measurement = cv::Mat::zeros(4, 1, CV_32F);
            measurement.at<float>(0) = bbox.x + bbox.width / 2.0f;
            measurement.at<float>(1) = bbox.y + bbox.height / 2.0f;
            measurement.at<float>(2) = bbox.width;
            measurement.at<float>(3) = bbox.height;
            
            // 卡尔曼校正
            cv::Mat estimated = kalman.correct(measurement);
            
            // 更新内部rect为校正后的状态
            float cx = estimated.at<float>(0);
            float cy = estimated.at<float>(1);
            float w = estimated.at<float>(2);
            float h = estimated.at<float>(3);
            w = std::max(5.0f, w);
            h = std::max(5.0f, h);
            rect.x = round(cx - w / 2.0f);
            rect.y = round(cy - h / 2.0f);
            rect.width = round(w);
            rect.height = round(h);
            
            confidence = conf;
            
            // 更新统计信息
            missing_count = 0;
            hit_count++;
            total_hits++;
            age++;
            
            // 重置连续跟踪状态
            prediction_probability = 1.0f;
            is_prediction_only = false;
        }
        
        // 标记为未检测到 (只进行预测，不校正)
        void markMissing() {
            missing_count++;
            age++;
            hit_count = 0;
            
            // 卡尔曼预测
            kalman.predict();
            
            // 更新内部rect为预测值
            rect = cv::Rect(round(kalman.statePre.at<float>(0) - kalman.statePre.at<float>(2)/2.0f),
                           round(kalman.statePre.at<float>(1) - kalman.statePre.at<float>(3)/2.0f),
                           round(std::max(5.0f, kalman.statePre.at<float>(2))),
                           round(std::max(5.0f, kalman.statePre.at<float>(3))));
            
            // 更新连续跟踪状态
            is_prediction_only = true;
            
            // 计算预测概率 - 随时间指数衰减
            // 运动越稳定，衰减越慢；运动越不稳定，衰减越快
            float stability_factor = calculateMotionStability();
            float decay_rate = 0.2f * (1.0f - stability_factor); // 稳定运动衰减更慢
            prediction_probability *= (1.0f - decay_rate);
        }
        
        // 计算运动稳定性因子 (0-1)，用于预测概率衰减
        float calculateMotionStability() const {
            // 如果历史记录太少，返回默认值
            if (history.size() < 3) return 0.5f;
            
            // 计算最近几次移动的方向一致性
            std::vector<cv::Point2f> directions;
            for (size_t i = 1; i < history.size(); i++) {
                cv::Point2f p1(history[i-1].x + history[i-1].width/2.0f, 
                              history[i-1].y + history[i-1].height/2.0f);
                cv::Point2f p2(history[i].x + history[i].width/2.0f, 
                              history[i].y + history[i].height/2.0f);
                cv::Point2f dir = p2 - p1;
                float mag = std::sqrt(dir.x*dir.x + dir.y*dir.y);
                if (mag > 1e-5) {
                    dir.x /= mag;
                    dir.y /= mag;
                    directions.push_back(dir);
                }
            }
            
            // 计算方向一致性
            if (directions.size() < 2) return 0.5f;
            
            float avg_cos_sim = 0.0f;
            for (size_t i = 1; i < directions.size(); i++) {
                float dot = directions[i].x * directions[i-1].x + directions[i].y * directions[i-1].y;
                avg_cos_sim += std::max(0.0f, dot); // 只考虑正相关
            }
            avg_cos_sim /= (directions.size() - 1);
            
            return avg_cos_sim; // 0=不稳定, 1=完全稳定
        }
        
        // 获取平滑掩码 - 时间域平滑，减少抖动
        cv::Mat getSmoothedMask() const {
            if (mask_history.empty()) {
                return mask.empty() ? cv::Mat() : mask.clone();
            }
            if (mask_history.size() == 1) {
                return mask_history.front().empty() ? cv::Mat() : mask_history.front().clone();
            }
            
            cv::Mat recent_mask;
            for (auto it = mask_history.rbegin(); it != mask_history.rend(); ++it) {
                if (!it->empty()) {
                    recent_mask = it->clone();
                    break;
                }
            }
            
            if (recent_mask.empty()) return cv::Mat();
            if (mask_history.size() < 3) return recent_mask;
            
            cv::Mat current = mask_history.back();
            if (current.empty()) return recent_mask;
            
            cv::Mat prev, prev2;
            int valid_count = 0;
            for (int i = mask_history.size() - 2; i >= 0 && valid_count < 2; i--) {
                if (!mask_history[i].empty()) {
                    if (valid_count == 0) prev = mask_history[i].clone();
                    else prev2 = mask_history[i].clone();
                    valid_count++;
                }
            }
            
            if (valid_count < 2 || current.size() != prev.size() || (valid_count > 1 && current.size() != prev2.size())) {
                return recent_mask;
            }
            
            cv::Mat smoothed_mask;
            double alpha = 0.6, beta = 0.3, gamma = 0.1;
            cv::addWeighted(current, alpha, prev, beta, 0.0, smoothed_mask);
            if (valid_count > 1) {
                cv::Mat temp = smoothed_mask.clone();
                cv::addWeighted(temp, 1.0, prev2, gamma, 0.0, smoothed_mask);
            }
            
            cv::threshold(smoothed_mask, smoothed_mask, 127, 255, cv::THRESH_BINARY);
            return smoothed_mask;
        }
        
        // 更新掩码 - 添加新掩码，维护历史
        void updateMask(const cv::Mat& new_mask, const std::vector<float>& coeffs) {
            if (new_mask.empty() || new_mask.cols <= 0 || new_mask.rows <= 0) {
                return;
            }
            
            if (mask_history.size() >= 5) {
                mask_history.pop_front();
            }
            
            cv::Mat mask_copy = new_mask.clone();
            if (!mask_copy.empty() && mask_copy.data != nullptr) {
                mask_history.push_back(mask_copy);
                mask = mask_copy.clone();
                if (!coeffs.empty()) {
                    mask_coeffs = coeffs;
                }
                has_valid_mask = true;
            }
        }
        
        // 预测掩码 - 基于卡尔曼预测的框变换掩码
        cv::Mat predictMask() const {
            if (!has_valid_mask || mask.empty() || mask_history.empty()) {
                return cv::Mat();
            }
            
            cv::Mat latest_mask = getSmoothedMask();
            if (latest_mask.empty()) {
                return cv::Mat();
            }
            
            // 获取卡尔曼预测的框
            cv::Rect predicted_box = cv::Rect(round(kalman.statePre.at<float>(0) - kalman.statePre.at<float>(2)/2.0f),
                                             round(kalman.statePre.at<float>(1) - kalman.statePre.at<float>(3)/2.0f),
                                             round(std::max(5.0f, kalman.statePre.at<float>(2))),
                                             round(std::max(5.0f, kalman.statePre.at<float>(3))));
            
            // 获取当前状态的框 (校正后或上一次预测)
            cv::Rect current_box = rect; 

            if (predicted_box.width <= 0 || predicted_box.height <= 0 || 
                current_box.width <= 0 || current_box.height <= 0) {
                return latest_mask.clone();
            }
            
            float scale_x = (float)predicted_box.width / current_box.width;
            float scale_y = (float)predicted_box.height / current_box.height;
            float dx = predicted_box.x - current_box.x;
            float dy = predicted_box.y - current_box.y;
            
            cv::Mat transform_matrix = (cv::Mat_<double>(2,3) << scale_x, 0, dx, 0, scale_y, dy);
            cv::Mat transformed_mask;
            cv::Size target_size(predicted_box.width, predicted_box.height);
            
            if (target_size.width > 0 && target_size.height > 0) {
                cv::warpAffine(latest_mask, transformed_mask, transform_matrix, 
                            target_size, cv::INTER_NEAREST);
                if (transformed_mask.empty() || transformed_mask.cols <= 0 || transformed_mask.rows <= 0) {
                    return latest_mask.clone();
                }
            } else {
                return latest_mask.clone();
            }
            
            return transformed_mask;
        }
    };
    
    // 计算两个矩形的IoU
    float calculateIoU(const cv::Rect& r1, const cv::Rect& r2) const {
        int x1 = std::max(r1.x, r2.x);
        int y1 = std::max(r1.y, r2.y);
        int x2 = std::min(r1.x + r1.width, r2.x + r2.width);
        int y2 = std::min(r1.y + r1.height, r2.y + r2.height);
        if (x2 < x1 || y2 < y1) return 0.0f;
        float intersection = (x2 - x1) * (y2 - y1);
        float area1 = r1.width * r1.height;
        float area2 = r2.width * r2.height;
        return intersection / (area1 + area2 - intersection + 1e-6); // Add epsilon
    }
    
    // 计算两个质心之间的距离
    float calculateCentroidDistance(const cv::Rect& r1, const cv::Rect& r2) const {
        float c1x = r1.x + r1.width/2.0f;
        float c1y = r1.y + r1.height/2.0f;
        float c2x = r2.x + r2.width/2.0f;
        float c2y = r2.y + r2.height/2.0f;
        float dx = c1x - c2x;
        float dy = c1y - c2y;
        return std::sqrt(dx*dx + dy*dy);
    }
    
    // 计算运动方向相似度
    float calculateDirectionSimilarity(const TrackingObject& track1, const TrackingObject& track2) const {
        cv::Point2f dir1 = track1.getMotionDirection();
        cv::Point2f dir2 = track2.getMotionDirection();
        if ((std::abs(dir1.x) < 1e-5 && std::abs(dir1.y) < 1e-5) || 
            (std::abs(dir2.x) < 1e-5 && std::abs(dir2.y) < 1e-5)) {
            return 0.0f;
        }
        return dir1.x * dir2.x + dir1.y * dir2.y;
    }
    
    // 计算关联代价 (使用卡尔曼预测的框)
    float associationCost(const TrackingObject& track, const cv::Rect& detection) const {
        cv::Rect predicted = track.rect; // predictNextPosition() 已在update函数开始时调用并更新了track.rect
        float iou = calculateIoU(predicted, detection);
        float iou_cost = 1.0f - iou;
        float dist = calculateCentroidDistance(predicted, detection);
        float normalized_dist = std::min(1.0f, (float)(dist / (std::max(predicted.width, predicted.height) * 2.0f + 1e-6f))); 
        
        // 常量定义
        const int MOTION_SUBMODE_STATIC = 0;    // 静态
        const int MOTION_SUBMODE_VARIABLE = 3;  // 变速
        const int SPATIAL_SPARSE = 0;           // 稀疏
        const int SPATIAL_NORMAL = 1;           // 正常
        const int SPATIAL_DENSE = 2;            // 密集
        const int SPATIAL_VERY_DENSE = 3;       // 超密集
        
        // 基于运动子模式和空间分布模式的权重调整
        float iou_weight = 0.7f;
        float dist_weight = 0.3f;
        
        // 先根据运动子模式调整基础权重
        if (params_.motion_submode == MOTION_SUBMODE_VARIABLE) { // Variable Motion
            // 机动模式下降低预测位置的权重，更依赖IoU
            iou_weight = 0.9f;
            dist_weight = 0.1f;
        } else if (params_.motion_submode == MOTION_SUBMODE_STATIC) { // Static
            // 静态模式下可以更信任预测位置
            iou_weight = 0.5f;
            dist_weight = 0.5f;
        }
        
        // 再根据空间分布模式进一步调整权重
        switch (params_.spatial_distribution) {
            case SPATIAL_SPARSE: // 稀疏场景
                // 稀疏场景中，目标分散，可以更信任IoU
                iou_weight += 0.1f;
                dist_weight -= 0.1f;
                break;
                
            case SPATIAL_DENSE: // 密集场景
                // 密集场景中，IoU不太可靠，应增加距离权重
                iou_weight -= 0.1f;
                dist_weight += 0.1f;
                break;
                
            case SPATIAL_VERY_DENSE: // 超密集场景
                // 超密集场景中，目标重叠严重，距离权重更重要
                iou_weight -= 0.2f;
                dist_weight += 0.2f;
                
                // 在特殊标志启用时应用额外的关联逻辑
                if (params_.use_reid_features && track.has_valid_mask && track.age > 3) {
                    // 此处是占位逻辑，如果有ReID特征，实际应加入特征相似度计算
                    float feature_sim = 0.8f; // 模拟特征相似度，实际应基于特征计算
                    return 0.5f * iou_cost + 0.2f * normalized_dist + 0.3f * (1.0f - feature_sim);
                }
                
                // 群组管理逻辑，仅占位示例
                if (params_.use_group_management && !track.is_prediction_only) {
                    // 如果启用了群组管理，可以适当调整关联权重
                    iou_weight -= 0.05f;
                    dist_weight += 0.05f;
                }
                break;
                
            case SPATIAL_NORMAL: // 正常场景 (默认)
            default:
                // 保持默认权重
                break;
        }
        
        // 确保权重总和为1
        float sum = iou_weight + dist_weight;
        iou_weight /= sum;
        dist_weight /= sum;
        
        // 计算最终代价
        return iou_weight * iou_cost + dist_weight * normalized_dist;
    }
    
    // 贪婪关联算法 (保持不变)
    void greedyAssignment(
        const std::vector<std::vector<float>>& cost_matrix,
        float cost_threshold,
        std::vector<std::pair<int, int>>& matches,
        std::vector<int>& unmatched_detections,
        std::vector<int>& unmatched_tracks) {
        int num_tracks = cost_matrix.size();
        int num_detections = num_tracks > 0 ? cost_matrix[0].size() : 0;
        unmatched_tracks.clear();
        for (int i = 0; i < num_tracks; i++) unmatched_tracks.push_back(i);
        unmatched_detections.clear();
        for (int i = 0; i < num_detections; i++) unmatched_detections.push_back(i);
        if (num_tracks == 0 || num_detections == 0) return;
        std::vector<std::vector<float>> cost_matrix_copy = cost_matrix;
        while (!unmatched_tracks.empty() && !unmatched_detections.empty()) {
            float min_cost = FLT_MAX;
            int min_track_idx_in_list = -1;
            int min_detection_idx_in_list = -1;
            for (size_t i = 0; i < unmatched_tracks.size(); i++) {
                int track_idx = unmatched_tracks[i];
                for (size_t j = 0; j < unmatched_detections.size(); j++) {
                    int detection_idx = unmatched_detections[j];
                    float cost = cost_matrix_copy[track_idx][detection_idx];
                    if (cost < min_cost) {
                        min_cost = cost;
                        min_track_idx_in_list = i;
                        min_detection_idx_in_list = j;
                    }
                }
            }
            if (min_cost > cost_threshold || min_track_idx_in_list < 0) break;
            matches.push_back({unmatched_tracks[min_track_idx_in_list], unmatched_detections[min_detection_idx_in_list]});
            unmatched_tracks.erase(unmatched_tracks.begin() + min_track_idx_in_list);
            unmatched_detections.erase(unmatched_detections.begin() + min_detection_idx_in_list);
        }
    }
    
    // 处理未匹配轨迹 (标记丢失)
    void processUnmatchedTracks(const std::vector<int>& unmatched_tracks) {
        for (int idx : unmatched_tracks) {
            tracks[idx].markMissing();
        }
    }
    
    // 检查是否是同一个目标重现 (保持不变)
    bool checkReappearance(const TrackingObject& track, const cv::Rect& detection) const {
        float size_change = std::abs(1.0f - (float)(detection.width * detection.height) / (track.rect.width * track.rect.height + 1e-6));
        float dist = calculateCentroidDistance(track.rect, detection);
        return size_change < 0.5f && dist < std::max(detection.width, detection.height) * 2;
    }
    
    std::vector<TrackingObject> tracks;
    int next_id = 0;
    std::mt19937 rng;
    TrackingParams params_; // 存储当前使用的参数
    
    // 根据跟踪模式更新策略
    void updateTrackingStrategy() {
        switch (params_.tracking_mode) {
            case 0: // 稳定模式
                // 稳定模式使用常规卡尔曼更新策略
                params_.enable_motion_compensation = false;
                break;
                
            case 1: // 手持模式
                // 启用运动补偿
                params_.enable_motion_compensation = true;
                break;
                
            case 2: // 自定义模式
                // 使用用户设定的参数
                break;
        }
    }

public:
    // 构造函数，接收参数
    EnhancedTracking(const TrackingParams& params = TrackingParams())
        : rng(std::random_device()()), params_(params) {
        // 预分配内存以避免实时跟踪时的频繁内存分配
        tracks.reserve(100);
        next_id = 0;
    }

    // 默认构造函数（如果需要）
    // EnhancedTracking() : EnhancedTracking(TrackingParams()) {}

    // Add a method to get the internal tracks for visualization
    const std::vector<TrackingObject>& getTracks() const {
        return tracks;
    }

    // 更新函数
    std::vector<Object> update(const std::vector<Object>& detections) {
        // 根据当前跟踪模式更新策略
        updateTrackingStrategy();
        
        // 预测所有跟踪对象的下一个位置 (卡尔曼滤波)
        for (auto& track : tracks) {
            track.predictNextPosition();
        }

        std::vector<std::pair<int, int>> matches;
        std::vector<int> unmatched_tracks_indices;
        std::vector<int> unmatched_detections_indices;
        
        float confirm_threshold = 1.0f - params_.iou_threshold; // 使用参数

        if (!detections.empty() && !tracks.empty()) {
            std::vector<std::vector<float>> cost_matrix(tracks.size(), std::vector<float>(detections.size()));
            for (size_t i = 0; i < tracks.size(); i++) {
                for (size_t j = 0; j < detections.size(); j++) {
                    if (tracks[i].label != detections[j].label) {
                        cost_matrix[i][j] = 1.0f;
                    } else {
                        cost_matrix[i][j] = associationCost(tracks[i], detections[j].rect);
                    }
                }
            }
            // 使用参数中的阈值
            greedyAssignment(cost_matrix, confirm_threshold, matches, unmatched_detections_indices, unmatched_tracks_indices);
        } else {
            for (size_t i = 0; i < tracks.size(); i++) unmatched_tracks_indices.push_back(i);
            for (size_t i = 0; i < detections.size(); i++) unmatched_detections_indices.push_back(i);
        }
        
        // 重现检查 (逻辑保持不变)
        for (auto it_track = unmatched_tracks_indices.begin(); it_track != unmatched_tracks_indices.end();) {
            bool found_match = false;
            if (tracks[*it_track].missing_count <= 5) {
                for (auto it_det = unmatched_detections_indices.begin(); it_det != unmatched_detections_indices.end();) {
                    if (tracks[*it_track].label == detections[*it_det].label && 
                        checkReappearance(tracks[*it_track], detections[*it_det].rect)) {
                        matches.push_back({*it_track, *it_det});
                        it_track = unmatched_tracks_indices.erase(it_track);
                        it_det = unmatched_detections_indices.erase(it_det);
                        found_match = true;
                        break;
                    } else {
                        ++it_det;
                    }
                }
            }
            if (!found_match) ++it_track;
        }
        
        // 处理未匹配轨迹
        processUnmatchedTracks(unmatched_tracks_indices);
        
        // 用新检测更新匹配的跟踪器
        for (const auto& match : matches) {
            int track_idx = match.first;
            int det_idx = match.second;
            tracks[track_idx].update(detections[det_idx].rect, detections[det_idx].prob);
            if (!detections[det_idx].mask.empty() && params_.enable_mask_tracking) {
                tracks[track_idx].updateMask(detections[det_idx].mask, detections[det_idx].mask_coeffs);
            }
        }
        
        // 为未匹配的检测创建新的跟踪器
        for (int idx : unmatched_detections_indices) {
            TrackingObject new_track(detections[idx].rect, next_id++, detections[idx].label, detections[idx].prob, params_);
            // Set Kalman parameters based on params_
            cv::setIdentity(new_track.kalman.processNoiseCov, cv::Scalar::all(params_.kalman_process_noise)); 
            // Adjust specific dimensions if needed based on params_
            cv::setIdentity(new_track.kalman.measurementNoiseCov, cv::Scalar::all(params_.kalman_measurement_noise));
            
            // 设置手持模式相关参数
            new_track.enable_motion_compensation = params_.enable_motion_compensation;
            new_track.acceleration_weight = params_.acceleration_weight;
            
            if (!detections[idx].mask.empty() && params_.enable_mask_tracking) {
                new_track.updateMask(detections[idx].mask, detections[idx].mask_coeffs);
            }
            tracks.push_back(new_track);
        }
        
        // --- 删除长时间未检测到的轨迹 (修改逻辑避免拷贝/移动) ---
        std::vector<TrackingObject> kept_tracks;
        kept_tracks.reserve(tracks.size()); // 预分配空间
        for (auto& track : tracks) {
            if (track.missing_count <= params_.max_age) { // 使用参数
                // 对于需要保留的track，直接移动到新向量中
                // 注意: 这要求TrackingObject有移动构造函数 (默认生成即可，因为没有指针成员)
                kept_tracks.push_back(std::move(track)); 
            }
        }
        // 用保留的轨迹替换旧的轨迹列表
        tracks = std::move(kept_tracks);
        // --- 删除逻辑修改结束 ---
        
        // 生成输出
        std::vector<Object> results;
        for (const auto& track : tracks) {
            if (track.total_hits >= params_.min_hits && track.missing_count <= 2) { 
                Object obj;
                obj.rect = track.rect;
                obj.label = track.label;
                obj.prob = track.confidence;
                obj.track_id = track.id;
                
                if (params_.enable_mask_tracking && track.has_valid_mask) {
                    if (track.missing_count > 0) {
                        cv::Mat predicted = track.predictMask();
                        if (!predicted.empty() && predicted.cols > 0 && predicted.rows > 0) {
                            obj.mask = predicted;
                        }
                    } else {
                        cv::Mat smoothed = track.getSmoothedMask();
                        if (!smoothed.empty() && smoothed.cols > 0 && smoothed.rows > 0) {
                            obj.mask = smoothed;
                        }
                    }
                    
                    if (!obj.mask.empty() && (obj.mask.cols != track.rect.width || obj.mask.rows != track.rect.height)) {
                        if (track.rect.width > 0 && track.rect.height > 0) {
                            cv::Mat resized;
                            cv::resize(obj.mask, resized, cv::Size(track.rect.width, track.rect.height), 0, 0, cv::INTER_NEAREST);
                            if (!resized.empty() && resized.size() == cv::Size(track.rect.width, track.rect.height)) {
                                obj.mask = resized;
                            } else {
                                obj.mask = cv::Mat();
                            }
                        }
                    }
                    if (!track.mask_coeffs.empty()) {
                        obj.mask_coeffs = track.mask_coeffs;
                    }
                    if (obj.mask.empty() || obj.mask.cols <= 0 || obj.mask.rows <= 0) {
                        obj.mask = cv::Mat();
                    }
                }
                results.push_back(obj);
            }
        }
        return results;
    }

    // Add this function to draw trajectory paths
    void drawTrajectoryPaths(cv::Mat& img, const std::vector<TrackingObject>& tracks, const TrackingParams& params) {
        if (!params.enable_trajectory_viz) return;

        for (const auto& track : tracks) {
            // 基本检查：只处理有足够命中的轨迹
            if (track.total_hits < params.min_hits) continue;
            
            // 连续跟踪模式下的特殊处理
            if (params.enable_continuous_tracking && track.is_prediction_only) {
                // 检查是否满足预测显示条件
                if (track.prediction_probability < params.prediction_probability_threshold || 
                    track.missing_count > params.max_prediction_frames) {
                    continue; // 预测概率太低或消失太久，不显示此轨迹
                }
            }

            // 获取轨迹点 (未来预测)
            std::vector<cv::Point2f> futurePoints;
            cv::Rect currentRect = track.rect;
            cv::KalmanFilter tempKalman = track.kalman; // 创建副本以不干扰实际跟踪器
            
            // 预测未来轨迹点
            for (int i = 0; i < params.trajectory_length; i++) {
                cv::Mat prediction = tempKalman.predict();
                float cx = prediction.at<float>(0);
                float cy = prediction.at<float>(1);
                float w = prediction.at<float>(2);
                float h = prediction.at<float>(3);
                
                futurePoints.push_back(cv::Point2f(cx, cy));
            }
            
            if (futurePoints.size() < 2) continue;
            
            // 确定线型 - 对预测状态使用虚线
            int lineType = cv::LINE_AA; // 默认抗锯齿
            int lineStyle = track.is_prediction_only ? params.prediction_line_type : 0;
            
            // 绘制轨迹
            for (size_t i = 0; i < futurePoints.size() - 1; i++) {
                cv::Scalar color;
                
                // 基于选定模式获取颜色
                switch (params.trajectory_color_mode) {
                    case 0: // 单色 - 使用轨迹颜色
                        color = track.color;
                        break;
                    case 1: // 渐变 - 从全色到透明
                        {
                            float alpha = 1.0f - (float)i / futurePoints.size();
                            // 对预测状态，整体调暗
                            if (track.is_prediction_only) {
                                alpha *= track.prediction_probability; // 根据预测概率调整透明度
                            }
                            color = cv::Scalar(
                                track.color[0] * alpha,
                                track.color[1] * alpha,
                                track.color[2] * alpha
                            );
                        }
                        break;
                    case 2: // 彩虹效果
                        {
                            // 彩虹效果：色调从0到240 (红到蓝)
                            float hue = 240.0f * i / futurePoints.size();
                            cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
                            cv::Mat rgb;
                            cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
                            color = cv::Scalar(rgb.at<cv::Vec3b>(0, 0)[0], 
                                              rgb.at<cv::Vec3b>(0, 0)[1], 
                                              rgb.at<cv::Vec3b>(0, 0)[2]);
                            
                            // 预测状态时调整亮度
                            if (track.is_prediction_only) {
                                float bright = track.prediction_probability;
                                color = cv::Scalar(
                                    color[0] * bright,
                                    color[1] * bright,
                                    color[2] * bright
                                );
                            }
                        }
                        break;
                }
                
                // 确定绘制样式
                float thickness = params.trajectory_thickness;
                
                // 对预测轨迹可以使用虚线效果
                if (lineStyle == 1) { // 虚线
                    // 虚线效果实现
                    cv::Point2f p1 = futurePoints[i];
                    cv::Point2f p2 = futurePoints[i+1];
                    cv::Point2f dir = p2 - p1;
                    float dist = sqrt(dir.x*dir.x + dir.y*dir.y);
                    
                    if (dist > 1.0f) {
                        dir.x /= dist;
                        dir.y /= dist;
                        
                        const float dash_len = 5.0f;
                        const float gap_len = 3.0f;
                        
                        for (float d = 0; d < dist; d += dash_len + gap_len) {
                            float start = d;
                            float end = std::min(d + dash_len, dist);
                            
                            cv::Point2f dash_start = p1 + dir * start;
                            cv::Point2f dash_end = p1 + dir * end;
                            
                            cv::line(img, dash_start, dash_end, color, thickness, lineType);
                        }
                    }
                } else { // 实线
                    cv::line(img, futurePoints[i], futurePoints[i+1], color, thickness, lineType);
                }
                
                // 在每隔几个轨迹点绘制小圆点
                if (i % 3 == 0) { // 每隔3个点画一个圆，避免太密集
                    cv::circle(img, futurePoints[i], thickness + 1, color, -1, lineType);
                }
            }
            
            // 绘制终点
            cv::Scalar endColor = track.color;
            // 对预测状态调整终点颜色
            if (track.is_prediction_only) {
                float bright = track.prediction_probability;
                endColor = cv::Scalar(
                    track.color[0] * bright,
                    track.color[1] * bright,
                    track.color[2] * bright
                );
            }
            cv::circle(img, futurePoints.back(), params.trajectory_thickness + 2, endColor, -1, lineType);
        }
    }

    // Update drawDetections to call the trajectory visualization
    void drawDetections(cv::Mat& img, const std::vector<Object>& detections, 
                        const std::vector<TrackingObject>& tracks, 
                        const std::vector<const char*>& class_names,
                        int detection_style, const TrackingParams& params) {
        // Draw trajectory paths if enabled
        drawTrajectoryPaths(img, tracks, params);
        
        // Draw detection boxes - keep existing code
        // ... existing code ...
    }
};

#endif // ENHANCED_TRACKER_H 