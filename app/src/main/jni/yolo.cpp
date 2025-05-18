#include "yolo.h"
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "EnhancedTracker.h" // 包含跟踪器头文件
#include "cpu.h"
#include <iostream>
#include <string>
#include <sstream>
#include <android/log.h>
#include <map>
#include <deque>
#include <vector>
#include <cmath>
#include <mutex> // 添加互斥锁头文件
#include "benchmark.h"

// 添加互斥锁和掩码启用标志
static std::mutex draw_mutex;
static bool is_mask_enabled = true;

// Declare g_tracker as a static pointer
static EnhancedTracking* g_tracker = nullptr; 

static ncnn::Mutex draw_lock;

// Wrap helper functions in an anonymous namespace
namespace {

float fast_exp(float x) {
    // 对常见的x值范围优化
    if (x < -5.0f) return 0.0f;
    if (x > 5.0f) return exp(x);
    
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

float intersection_area(const Object &a, const Object &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }
    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void qsort_descent_inplace(std::vector<Object> &faceobjects) {
    if (faceobjects.empty())
        return;
    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked,
                              float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();
    if (n == 0) return;

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    // 预先分配内存，避免频繁分配
    picked.reserve(n);

    for (int i = 0; i < n; i++) {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = faceobjects[picked[j]];

            // 使用快速交集计算
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (union_area <= 0) continue;
            
            // 超过IoU阈值则抑制
            if (inter_area / union_area > nms_threshold) {
                keep = 0;
                break; // 提前终止循环
            }
        }
        if (keep)
            picked.push_back(i);
    }
}

void
generate_grids_and_stride(const int target_w, const int target_h, std::vector<int> &strides,
                          std::vector<GridAndStride> &grid_strides) {
    for (int i = 0; i < (int) strides.size(); i++) {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++) {
            for (int g0 = 0; g0 < num_grid_w; g0++) {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

// Modified generate_proposals to handle segmentation coefficients
void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat &pred,
                               float prob_threshold, std::vector<Object> &objects,
                               bool is_segmentation, int num_coeffs, int num_class = 80) {
    const int num_points = grid_strides.size(); // Should match pred.h
    const int reg_max_1 = 16;
    // Calculate the number of channels for box+class and mask coeffs
    const int box_class_channels = 4 * reg_max_1 + num_class;
    
    // 输出张量形状检查与自适应
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "生成预测: 张量形状=[%d,%d,%d], 网格点数=%d, 通道数=%d", 
                       pred.dims, pred.w, pred.h, num_points, box_class_channels);
    
    // 自适应处理预测张量：处理可能的形状不匹配
    ncnn::Mat working_pred;
    bool using_working_copy = false;
    
    if (pred.w < box_class_channels && pred.h >= box_class_channels) {
        // 特殊情况：宽度小于所需通道数，但高度足够 - 可能需要转置
        working_pred = pred.reshape(pred.h, pred.w);
        using_working_copy = true;
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "自动转置预测张量: %dx%d -> %dx%d", 
                           pred.w, pred.h, working_pred.w, working_pred.h);
    } else if (pred.w < box_class_channels && pred.h < box_class_channels && pred.dims == 3) {
        // 3D张量的特殊处理
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "3D预测张量: [%d,%d,%d]", 
                           pred.c, pred.h, pred.w);
        if (pred.c * pred.w >= box_class_channels) {
            // 可能可以重新解释为2D张量
            working_pred = ncnn::Mat(pred.c * pred.w, pred.h, pred.cstep);
            using_working_copy = true;
            __android_log_print(ANDROID_LOG_INFO, "ncnn", "重新解释3D张量为: %dx%d", 
                               working_pred.w, working_pred.h);
        }
    } else {
        // 使用原始预测张量
        working_pred = pred;
    }
    
    // 最终安全检查
    if (working_pred.w < box_class_channels) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "无效的预测矩阵: 宽度 (%d) < 所需通道数 (%d)，可能需要其他格式转换", 
                            working_pred.w, box_class_channels);
        return;
    }
    
    // Special case for YOLOv8n-seg: In some converted models, the mask coefficients
    // are not part of the detection output tensor, but in a separate tensor.
    // In this case, we don't need to check for the additional mask_coeffs size
    bool separate_mask_coeffs = false;
    
    // For yolov8s-seg, determine if we have coefficients in the prediction tensor
    bool has_embedded_coeffs = false;
    int coeff_offset = box_class_channels;
    
    // Check if prediction tensor has additional width for mask coefficients
    if (is_segmentation && num_coeffs > 0) {
        if (working_pred.w >= box_class_channels + num_coeffs) {
            has_embedded_coeffs = true;
            __android_log_print(ANDROID_LOG_INFO, "ncnn", "检测到预测矩阵中的嵌入掩码系数: 偏移=%d", coeff_offset);
        } else {
            __android_log_print(ANDROID_LOG_WARN, "ncnn", "预测矩阵宽度 (%d) 不足以包含嵌入掩码系数 (需要 %d)", 
                              working_pred.w, box_class_channels + num_coeffs);
        }
    }
    
    // Verify we have enough rows in our prediction
    int effective_num_points = std::min(num_points, working_pred.h);
    if (effective_num_points < num_points) {
        __android_log_print(ANDROID_LOG_WARN, "ncnn", "预测行数不足: 仅有 %d 行，需要 %d 行", 
                           effective_num_points, num_points);
    }
    
    // 处理每个预测点
    for (int i = 0; i < effective_num_points; i++) {
        // Pointer to the start of box+class scores for the i-th proposal
        const float *box_class_scores = working_pred.row(i);

        // --- Box Decoding and Class Score --- (same as before)
        const float *scores = box_class_scores + 4 * reg_max_1;
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++) {
            float confidence = scores[k];
            if (confidence > score) {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);

        if (box_prob >= prob_threshold) {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void *) box_class_scores);
            // Apply softmax to decode box distribution (DFL)
            {
                ncnn::Layer *softmax = ncnn::create_layer("Softmax");
                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);
                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;
                softmax->create_pipeline(opt);
                softmax->forward_inplace(bbox_pred, opt);
                softmax->destroy_pipeline(opt);
                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++) {
                float dis = 0.f;
                const float *dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++) {
                    dis += l * dis_after_sm[l];
                }
                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;
            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];
            // --- End Box Decoding ---

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;
            obj.gindex = i; // Store grid index for mask coefficient lookup

            // --- Extract Mask Coefficients --- (if segmentation model)
            if (is_segmentation && num_coeffs > 0) {
                obj.mask_coeffs.clear();
                obj.mask_coeffs.reserve(num_coeffs);
                
                if (has_embedded_coeffs) {
                    // Get pointer to the mask coefficients in the prediction tensor
                    const float* mask_ptr = box_class_scores + coeff_offset;
                    
                    // Sanity check for accessing coefficients
                    if (coeff_offset + num_coeffs <= working_pred.w) {
                        // Copy all coefficients
                        for (int c = 0; c < num_coeffs; c++) {
                            float coeff_val = mask_ptr[c];
                            obj.mask_coeffs.push_back(coeff_val);
                        }
                    } else {
                        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "系数访问越界: 偏移=%d, 需要=%d, 有=%d", 
                                          coeff_offset, num_coeffs, working_pred.w - coeff_offset);
                    }
                }
            }

            objects.push_back(obj);
        }
    }
    
    // 记录处理结果
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "proposal生成完成，共%zu个有效检测", objects.size());
}

// Helper function to apply sigmoid to a cv::Mat
void mat_sigmoid(cv::Mat& mat) {
    cv::exp(-mat, mat);
    mat = 1.0 / (1.0 + mat);
}

// Helper function for drawing corner brackets
void draw_corner_brackets(cv::Mat& img, const cv::Rect& rect, const cv::Scalar& color, int thickness, int corner_length_percent = 15) {
    int corner_len_w = rect.width * corner_length_percent / 100;
    int corner_len_h = rect.height * corner_length_percent / 100;
    corner_len_w = std::max(1, corner_len_w);
    corner_len_h = std::max(1, corner_len_h);

    // Top-left
    cv::line(img, rect.tl(), cv::Point(rect.x + corner_len_w, rect.y), color, thickness);
    cv::line(img, rect.tl(), cv::Point(rect.x, rect.y + corner_len_h), color, thickness);
    // Top-right
    cv::line(img, cv::Point(rect.x + rect.width, rect.y), cv::Point(rect.x + rect.width - corner_len_w, rect.y), color, thickness);
    cv::line(img, cv::Point(rect.x + rect.width, rect.y), cv::Point(rect.x + rect.width, rect.y + corner_len_h), color, thickness);
    // Bottom-left
    cv::line(img, cv::Point(rect.x, rect.y + rect.height), cv::Point(rect.x + corner_len_w, rect.y + rect.height), color, thickness);
    cv::line(img, cv::Point(rect.x, rect.y + rect.height), cv::Point(rect.x, rect.y + rect.height - corner_len_h), color, thickness);
    // Bottom-right
    cv::line(img, cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Point(rect.x + rect.width - corner_len_w, rect.y + rect.height), color, thickness);
    cv::line(img, cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Point(rect.x + rect.width, rect.y + rect.height - corner_len_h), color, thickness);
}

// Helper function for drawing gradient filled rect
void draw_gradient_rect(cv::Mat& img, const cv::Rect& rect, const cv::Scalar& color1, const cv::Scalar& color2, bool horizontal = true) {
    cv::Mat roi = img(rect);
    for (int i = 0; i < (horizontal ? rect.width : rect.height); ++i) {
        float ratio = (float)i / (horizontal ? rect.width : rect.height);
        cv::Scalar mixed_color = color1 * (1.0 - ratio) + color2 * ratio;
        if (horizontal) {
            roi.col(i).setTo(mixed_color);
        } else {
            roi.row(i).setTo(mixed_color);
        }
    }
}

// Helper for dashed rectangle
void draw_dashed_rectangle(cv::Mat& img, const cv::Rect& rect, const cv::Scalar& color, int thickness, int dash_length = 10) {
    cv::Point p1 = rect.tl();
    cv::Point p2 = cv::Point(rect.x + rect.width, rect.y + rect.height);
    int x1 = p1.x, y1 = p1.y;
    int x2 = p2.x, y2 = p2.y;
    int current_length = 0;
    bool draw = true;
    // Top
    for (int x = x1; x < x2; ++x) {
        if (draw) cv::line(img, {x, y1}, {x+1, y1}, color, thickness);
        current_length++;
        if (current_length >= dash_length) { draw = !draw; current_length = 0; }
    }
    // Bottom
    current_length = 0; draw = true;
    for (int x = x1; x < x2; ++x) {
        if (draw) cv::line(img, {x, y2}, {x+1, y2}, color, thickness);
        current_length++;
        if (current_length >= dash_length) { draw = !draw; current_length = 0; }
    }
    // Left
    current_length = 0; draw = true;
    for (int y = y1; y < y2; ++y) {
        if (draw) cv::line(img, {x1, y}, {x1, y+1}, color, thickness);
        current_length++;
        if (current_length >= dash_length) { draw = !draw; current_length = 0; }
    }
    // Right
    current_length = 0; draw = true;
    for (int y = y1; y < y2; ++y) {
        if (draw) cv::line(img, {x2, y}, {x2, y+1}, color, thickness);
        current_length++;
        if (current_length >= dash_length) { draw = !draw; current_length = 0; }
    }
}

} // end anonymous namespace

// Helper function to draw contours around a mask
void draw_mask_contour(cv::Mat& img, const cv::Mat& mask_8u, const cv::Scalar& color, int thickness) {
    if (mask_8u.empty() || thickness <= 0) return;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(img, contours, -1, color, thickness);
}

// Draw mask edge with different styles (solid, dashed, dotted)
void draw_styled_mask_contour(cv::Mat& img, const cv::Mat& mask_8u, const cv::Scalar& color, 
                              int thickness, int edgeType, cv::Point offset = cv::Point(0, 0)) {
    if (mask_8u.empty() || thickness <= 0) return;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) return;
    
    // For solid lines, use standard drawContours
    if (edgeType == 0) { // SOLID
        cv::drawContours(img, contours, -1, color, thickness, cv::LINE_AA, cv::noArray(), 0, offset);
        return;
    }
    
    // For dashed and dotted lines
    for (const auto& contour : contours) {
        if (contour.size() < 2) continue;
        
        for (size_t i = 0; i < contour.size(); i++) {
            cv::Point pt1 = contour[i] + offset;
            cv::Point pt2 = contour[(i + 1) % contour.size()] + offset;
            
            // Calculate distance between points
            double dist = std::sqrt(std::pow(pt2.x - pt1.x, 2) + std::pow(pt2.y - pt1.y, 2));
            
            if (edgeType == 1) { // DASHED
                // Draw dashed line (dash length = 10, gap length = 5)
                int dashLength = 10;
                int gapLength = 5;
                double totalLength = 0;
                
                while (totalLength < dist) {
                    double remainingDist = dist - totalLength;
                    double currentDashLength = std::min((double)dashLength, remainingDist);
                    
                    if (currentDashLength > 0) {
                        double t1 = totalLength / dist;
                        double t2 = (totalLength + currentDashLength) / dist;
                        
                        cv::Point dashPt1(pt1.x + (pt2.x - pt1.x) * t1, pt1.y + (pt2.y - pt1.y) * t1);
                        cv::Point dashPt2(pt1.x + (pt2.x - pt1.x) * t2, pt1.y + (pt2.y - pt1.y) * t2);
                        
                        cv::line(img, dashPt1, dashPt2, color, thickness, cv::LINE_AA);
                    }
                    
                    totalLength += currentDashLength + gapLength;
                }
            } 
            else if (edgeType == 2) { // DOTTED
                // Draw dotted line (dot every 10 pixels)
                int dotSpacing = 10;
                int numDots = std::max(1, (int)(dist / dotSpacing));
                
                for (int j = 0; j < numDots; j++) {
                    double t = j / (double)numDots;
                    cv::Point dotPt(pt1.x + (pt2.x - pt1.x) * t, pt1.y + (pt2.y - pt1.y) * t);
                    cv::circle(img, dotPt, thickness, color, -1, cv::LINE_AA);
                }
            }
        }
    }
}

// NEW Helper: Simulate drawing a glow effect by drawing larger, transparent rectangles
void draw_glow_rect(cv::Mat& img, const cv::Rect& rect, const cv::Scalar& color, int max_offset = 5, int steps = 3) {
    if (steps <= 0 || max_offset <= 0) return;
    for (int i = steps; i > 0; --i) {
        float current_alpha = 0.15f * (1.0f - (float)i / steps); // Fade out alpha
        int offset = max_offset * i / steps;
        cv::Rect glow_rect = rect + cv::Size(2 * offset, 2 * offset); // Expand rect
        glow_rect -= cv::Point(offset, offset);                      // Center it
        glow_rect &= cv::Rect(0, 0, img.cols, img.rows);            // Clip to bounds

        if (glow_rect.width > 0 && glow_rect.height > 0) {
            cv::Mat roi = img(glow_rect);
            cv::Mat glow_layer(roi.size(), CV_8UC3, color);
            cv::addWeighted(roi, 1.0, glow_layer, current_alpha, 0, roi); // Low alpha blend
        }
    }
}

// NEW Helper: Draw simple grid pattern on a Mat ROI
void draw_grid_pattern(cv::Mat& roi, const cv::Scalar& color, int spacing = 5) {
    if (roi.empty() || spacing <= 0) return;
    for (int y = 0; y < roi.rows; y += spacing) {
        cv::line(roi, cv::Point(0, y), cv::Point(roi.cols, y), color * 0.4, 1); // Faint grid
    }
    for (int x = 0; x < roi.cols; x += spacing) {
        cv::line(roi, cv::Point(x, 0), cv::Point(x, roi.rows), color * 0.4, 1); // Faint grid
    }
}

// Techy color palette (保持不变)
// Using vibrant, slightly desaturated colors with potential for alpha blending
const cv::Scalar tech_colors[] = {
    {0, 255, 255}, // Cyan
    {255, 0, 255}, // Magenta
    {0, 255, 0},   // Lime Green
    {255, 128, 0}, // Orange
    {0, 128, 255}, // Sky Blue
    {128, 0, 255}, // Purple
    {255, 255, 0}, // Yellow
    {0, 255, 128}, // Teal
    {255, 0, 128}, // Pink
    {128, 255, 0}, // Light Green
    {128, 128, 255}, // Lavender
    {255, 128, 128}, // Light Red
    {128, 255, 128}, // Mint Green
    {255, 200, 0},  // Gold
    {200, 200, 200}, // Light Gray
    {100, 255, 220}, // Aqua
    {255, 100, 50},  // Coral
    {220, 0, 100},   // Ruby
    {70, 200, 120}   // Emerald
};
const int num_tech_colors = sizeof(tech_colors) / sizeof(tech_colors[0]);

/*************************************************************/
/************************* Yolo方法具体实现 *************************/
/*************************************************************/
Yolo::Yolo() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
    
    detection_style = 0; // Default style
    is_segmentation_model = false;
    num_mask_coeffs = 0;
    det_output_name = "output";
    seg_output_name = "";
    mask_threshold = 0.4f;
    
    // 默认启用跟踪，使用默认参数
    enable_tracking = true;
    current_tracking_params = TrackingParams(); // 初始化为默认值
    // Initialize the tracker on creation
    // g_tracker = new EnhancedTracking(current_tracking_params); // Initialize here or in load?
    
    // 初始化类别过滤相关变量
    class_filtering_enabled = false;
    enabled_class_ids.clear();
    
    // Initialize style parameters with default values
    style_line_thickness = 2;
    style_line_type = 0; // CV_LINE_8
    style_box_alpha = 0.3f;
    style_box_color = 0xFF00FF00; // ARGB format: Green with alpha
    style_font_size = 1.0f;
    style_font_type = 0; // CV_FONT_HERSHEY_SIMPLEX
    style_text_color = 0xFFFFFFFF; // ARGB format: White with alpha
    style_mask_alpha = 0.5f;
    style_mask_contrast = 1.0f;
    style_custom_parameters_set = false; // 初始化为false，表示尚未设置自定义参数
}

Yolo::~Yolo() {
    // Clean up the tracker
    delete g_tracker;
    g_tracker = nullptr;
}

// Method to set the overall detection visualization style (保持不变)
void Yolo::setDetectionStyle(int style) {
    if (style >= 0 && style <= 9) { 
        this->detection_style = style;
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", "Set detection style to %d", style);
    } else {
        __android_log_print(ANDROID_LOG_WARN, "YoloStyle", "Invalid style index %d requested. Must be 0-9.", style);
        this->detection_style = 0;
    }
}

// 设置掩码阈值的方法 (保持不变)
void Yolo::setMaskThreshold(float threshold) {
    if (threshold >= 0.0f && threshold <= 1.0f) {
        this->mask_threshold = threshold;
        __android_log_print(ANDROID_LOG_INFO, "YoloMask", "设置掩码阈值为 %.2f", threshold);
    } else {
        __android_log_print(ANDROID_LOG_WARN, "YoloMask", "阈值 %.2f 超出范围，必须在0-1之间", threshold);
    }
}

// 实现新的跟踪参数设置方法
void Yolo::setTrackingParams(const TrackingParams& params) {
    current_tracking_params = params; // Store the complete params, including modes
    
    // Reinitialize the global tracker instance with the new parameters
    delete g_tracker; 
    g_tracker = new EnhancedTracking(current_tracking_params); // Pass the updated params
    
    // Log the update, including modes
    __android_log_print(ANDROID_LOG_INFO, "YoloTracker", 
        "Tracking params updated: Mode=%d, SubMode=%d, IoU=%.2f, MaxAge=%d, MinHits=%d, MaskTrack=%d",
        params.tracking_mode, params.motion_submode, 
        params.iou_threshold, params.max_age, params.min_hits, params.enable_mask_tracking);
}

// 启用/禁用跟踪方法
void Yolo::setTrackingEnabled(bool enable) {
    enable_tracking = enable;
    __android_log_print(ANDROID_LOG_INFO, "YoloTracker", "Tracking %s", enable ? "enabled" : "disabled");
}

bool Yolo::isTrackingEnabled() const {
    return enable_tracking;
}

// set/is EnableMaskTracking 现在通过 setTrackingParams 控制
void Yolo::setEnableMaskTracking(bool enable) {
    // 更新参数结构体并重新设置
    current_tracking_params.enable_mask_tracking = enable;
    setTrackingParams(current_tracking_params); 
    __android_log_print(ANDROID_LOG_INFO, "YoloTracker", "Mask Tracking %s", enable ? "enabled" : "disabled");
}

bool Yolo::isMaskTrackingEnabled() const {
    // 返回当前参数中的设置
    return current_tracking_params.enable_mask_tracking;
}

const std::vector<const char*>& Yolo::getClassNames() const {
    // 根据当前数据集类型返回对应的类名列表
    if (dataset_type == DATASET_OIV) {
        static const std::vector<const char*> oiv_names(oiv_class_names, oiv_class_names + sizeof(oiv_class_names) / sizeof(oiv_class_names[0]));
        return oiv_names;
    } else {
        // 默认返回COCO类名列表
        static const std::vector<const char*> coco_names(coco_class_names, coco_class_names + sizeof(coco_class_names) / sizeof(coco_class_names[0]));
        return coco_names;
    }
}

DatasetType Yolo::getDatasetType() const {
    return dataset_type;
}

/**
 * Set style parameters with text style and full text background support
 * This method sets the appearance parameters for object detection visualization
 */
bool Yolo::setStyleParameters(int styleId, int lineThickness, int lineType,
                             float boxAlpha, int boxColor, float fontSize,
                             int fontType, int textColor, int textStyle,
                             bool fullTextBackground,
                             float maskAlpha, float maskContrast,
                             int maskEdgeThickness, int maskEdgeType, int maskEdgeColor) {
    // 记录旧值用于调试
    int old_box_color = this->style_box_color;
    int old_text_color = this->style_text_color;
    int old_line_type = this->style_line_type;
    int old_text_style = this->style_text_style;
    bool old_full_text_background = this->style_full_text_background;
    
    // 标记哪些参数被显式修改，-1 表示未修改
    bool modified_line_thickness = (lineThickness > 0);
    bool modified_line_type = (lineType >= 0);
    bool modified_box_alpha = (boxAlpha >= 0);
    bool modified_box_color = (boxColor != 0);
    bool modified_font_size = (fontSize > 0);
    bool modified_font_type = (fontType >= 0);
    bool modified_text_color = (textColor != 0);
    bool modified_text_style = (textStyle >= 0);
    bool modified_full_text_background = true; // 参数总是被显式传入
    bool modified_mask_alpha = (maskAlpha >= 0);
    bool modified_mask_contrast = (maskContrast > 0);
    bool modified_mask_edge_thickness = (maskEdgeThickness > 0);
    bool modified_mask_edge_type = (maskEdgeType >= 0);
    bool modified_mask_edge_color = (maskEdgeColor != 0);
    
    // 先设置风格ID，加载该风格的默认参数
    this->setDetectionStyle(styleId);
    
    // 仅更新显式传入的参数，保留风格其他特性
    if (modified_line_thickness) this->style_line_thickness = lineThickness;
    if (modified_line_type) this->style_line_type = lineType;
    if (modified_box_alpha) this->style_box_alpha = boxAlpha;
    if (modified_box_color) this->style_box_color = boxColor;
    if (modified_font_size) this->style_font_size = fontSize;
    if (modified_font_type) this->style_font_type = fontType;
    if (modified_text_color) this->style_text_color = textColor;
    if (modified_text_style) this->style_text_style = textStyle;
    this->style_full_text_background = fullTextBackground;
    if (modified_mask_alpha) this->style_mask_alpha = maskAlpha;
    if (modified_mask_contrast) this->style_mask_contrast = maskContrast;
    if (modified_mask_edge_thickness) this->style_mask_edge_thickness = maskEdgeThickness;
    if (modified_mask_edge_type) this->style_mask_edge_type = maskEdgeType;
    if (modified_mask_edge_color) this->style_mask_edge_color = maskEdgeColor;
    
    // 根据文字风格设置对应的文字和背景颜色
    if (modified_text_style) {
        switch (textStyle) {
            case TEXT_STYLE_TECH:  // 科技风格 - 蓝底白字
                if (!modified_text_color) this->style_text_color = 0xFFFFFFFF; // 白色
                break;
            case TEXT_STYLE_FUTURE: // 未来风格 - 深蓝渐变底色
                if (!modified_text_color) this->style_text_color = 0xFF00FFFF; // 青色
                break;
            case TEXT_STYLE_NEON: // 霓虹风格 - 黑底亮色描边
                if (!modified_text_color) this->style_text_color = 0xFF00FF99; // 亮绿色
                break;
            case TEXT_STYLE_MILITARY: // 军事风格 - 绿底黄字
                if (!modified_text_color) this->style_text_color = 0xFFFFFF00; // 黄色
                break;
            case TEXT_STYLE_MINIMAL: // 简约风格 - 半透明底色
                if (!modified_text_color) this->style_text_color = 0xFFFFFFFF; // 白色
                break;
        }
    }
    
    // 至少有一个参数被修改时，标记自定义参数已设置
    this->style_custom_parameters_set = (modified_line_thickness || modified_line_type || 
                                        modified_box_alpha || modified_box_color || 
                                        modified_font_size || modified_font_type || 
                                        modified_text_color || modified_text_style ||
                                        modified_full_text_background ||
                                        modified_mask_alpha || modified_mask_contrast ||
                                        modified_mask_edge_thickness || modified_mask_edge_type ||
                                        modified_mask_edge_color);
    
    // Log the parameters for debugging
    int r = (this->style_box_color >> 16) & 0xFF;
    int g = (this->style_box_color >> 8) & 0xFF;
    int b = this->style_box_color & 0xFF;
    int a = (this->style_box_color >> 24) & 0xFF;
    
    int textR = (this->style_text_color >> 16) & 0xFF;
    int textG = (this->style_text_color >> 8) & 0xFF;
    int textB = this->style_text_color & 0xFF;
    int textA = (this->style_text_color >> 24) & 0xFF;
    
    int maskEdgeR = (this->style_mask_edge_color >> 16) & 0xFF;
    int maskEdgeG = (this->style_mask_edge_color >> 8) & 0xFF;
    int maskEdgeB = this->style_mask_edge_color & 0xFF;
    int maskEdgeA = (this->style_mask_edge_color >> 24) & 0xFF;
    
    // 输出旧值与新值的对比
    __android_log_print(ANDROID_LOG_INFO, "YoloStyle", "参数变化 - 风格ID: %d", styleId);
    // 不再输出框颜色变化
    if (modified_text_color) __android_log_print(ANDROID_LOG_INFO, "YoloStyle", "  文本颜色: 0x%08X -> 0x%08X", old_text_color, textColor);
    if (modified_line_type) __android_log_print(ANDROID_LOG_INFO, "YoloStyle", "  线型: %d -> %d", old_line_type, lineType);
    
    __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                       "设置风格参数 - ID: %d", styleId);
                       
    // 仅输出已修改的参数
    if (modified_line_thickness || modified_line_type)
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                       "  线条: 粗细=%d, 类型=%d", this->style_line_thickness, this->style_line_type);
    if (modified_box_alpha)
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                       "  框: 透明度=%.2f", this->style_box_alpha);
    if (modified_font_size || modified_font_type || modified_text_color)
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                       "  字体: 大小=%.2f, 类型=%d, 颜色=0x%08X (RGBA: %d,%d,%d,%d)", 
                       this->style_font_size, this->style_font_type, this->style_text_color, textR, textG, textB, textA);
    if (modified_mask_alpha || modified_mask_contrast)
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                       "  掩码: 透明度=%.2f, 对比度=%.2f", 
                       this->style_mask_alpha, this->style_mask_contrast);
    if (modified_mask_edge_thickness || modified_mask_edge_type || modified_mask_edge_color)
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                       "  掩码边缘: 粗细=%d, 类型=%d, 颜色=0x%08X (RGBA: %d,%d,%d,%d)", 
                       this->style_mask_edge_thickness, this->style_mask_edge_type, 
                       this->style_mask_edge_color, maskEdgeR, maskEdgeG, maskEdgeB, maskEdgeA);
    
    return true;
}

/**
 * Get current style parameters for a given style ID
 * This method returns the default parameters for the requested style
 */
bool Yolo::getStyleParameters(int styleId, int* lineThickness, int* lineType,
                             float* boxAlpha, int* boxColor, float* fontSize,
                             int* fontType, int* textColor,
                             float* maskAlpha, float* maskContrast,
                             int* maskEdgeThickness, int* maskEdgeType, int* maskEdgeColor) {
    // Set the default style temporarily to retrieve its parameters
    int originalStyle = this->detection_style;
    this->setDetectionStyle(styleId);
    
    // 用于调试
    __android_log_print(ANDROID_LOG_DEBUG, "YoloStyle", 
        "Getting style parameters for style ID: %d", styleId);
    
    // Initialize values based on the selected style
    switch (styleId) {
        case 0: // Cyber Grid
            *lineThickness = 2;
            *lineType = 2; // Corner brackets
            *boxAlpha = 0.2f;
            *fontSize = 1.0f;
            *fontType = 0; // CV_FONT_HERSHEY_SIMPLEX
            *maskAlpha = 0.2f;
            *maskContrast = 1.0f;
            // ARGB format: white
            *boxColor = 0xFFFFFFFF;
            *textColor = 0xFFFFFFFF;
            // Default mask edge parameters
            *maskEdgeThickness = 2;
            *maskEdgeType = 0; // Solid
            *maskEdgeColor = 0xFFFFFFFF; // White
            break;
            
        case 1: // Holographic Glow
            *lineThickness = 2;
            *lineType = 0; // Solid
            *boxAlpha = 0.15f;
            *fontSize = 1.0f;
            *fontType = 0;
            *maskAlpha = 0.15f;
            *maskContrast = 1.0f;
            *boxColor = 0xFFFFFF00; // ARGB: Yellow (FFFF00)
            *textColor = 0xFFFFFFFF; // White
            *maskEdgeThickness = 2;
            *maskEdgeType = 0; // Solid
            *maskEdgeColor = 0xFFFFFF00; // Yellow
            break;
            
        case 2: // Pulse Glitch
            *lineThickness = 3;
            *lineType = 1; // Dashed
            *boxAlpha = 0.2f;
            *fontSize = 1.0f;
            *fontType = 0;
            *maskAlpha = 0.2f;
            *maskContrast = 1.0f;
            *boxColor = 0xFF00FF00; // ARGB: Green (00FF00)
            *textColor = 0xFF00FF00;
            *maskEdgeThickness = 3;
            *maskEdgeType = 1; // Dashed
            *maskEdgeColor = 0xFF00FF00; // Green
            break;
            
        case 3: // Plasma Flow
            *lineThickness = 5;
            *lineType = 2; // Corner brackets
            *boxAlpha = 0.45f;
            *fontSize = 1.0f;
            *fontType = 0;
            *maskAlpha = 0.45f;
            *maskContrast = 1.0f;
            *boxColor = 0xFFFF00FF; // ARGB: Magenta (FF00FF)
            *textColor = 0xFFFFFFFF;
            *maskEdgeThickness = 5;
            *maskEdgeType = 2; // Dotted
            *maskEdgeColor = 0xFFFF00FF; // Magenta
            break;
            
        case 4: // Data Stream
            *lineThickness = 2;
            *lineType = 0; // Solid
            *boxAlpha = 0.25f;
            *fontSize = 1.0f;
            *fontType = 0;
            *maskAlpha = 0.25f;
            *maskContrast = 1.0f;
            *boxColor = 0xFFFF0000; // ARGB: Red (FF0000)
            *textColor = 0xFFFFFFFF;
            *maskEdgeThickness = 2;
            *maskEdgeType = 0; // Solid
            *maskEdgeColor = 0xFFFF0000; // Red
            break;
            
        case 5: // Target Lock
            *lineThickness = 3;
            *lineType = 2; // Corner brackets
            *boxAlpha = 0.0f;
            *fontSize = 1.0f;
            *fontType = 0;
            *maskAlpha = 0.0f;
            *maskContrast = 1.0f;
            *boxColor = 0xFF00FFFF; // ARGB: Cyan (00FFFF)
            *textColor = 0xFFFFFFFF;
            *maskEdgeThickness = 3;
            *maskEdgeType = 0; // Solid
            *maskEdgeColor = 0xFF00FFFF; // Cyan
            break;
            
        case 6: // Original Pulse
            *lineThickness = 1;
            *lineType = 1; // Dashed
            *boxAlpha = 0.0f;
            *fontSize = 1.0f;
            *fontType = 0;
            *maskAlpha = 0.0f;
            *maskContrast = 1.0f;
            *boxColor = 0xFF00FF00; // ARGB: Green (00FF00)
            *textColor = 0xFF00FF00;
            *maskEdgeThickness = 1;
            *maskEdgeType = 0; // Solid
            *maskEdgeColor = 0xFF00FF00; // Green
            break;
            
        case 7: // Neon Flow
            *lineThickness = 2;
            *lineType = 0; // Solid
            *boxAlpha = 0.3f;
            *fontSize = 1.0f;
            *fontType = 0;
            *maskAlpha = 0.3f;
            *maskContrast = 1.0f;
            *boxColor = 0xFFFF00FF; // ARGB: Magenta (FF00FF)
            *textColor = 0xFFFFFFFF;
            *maskEdgeThickness = 2;
            *maskEdgeType = 0; // Solid
            *maskEdgeColor = 0xFFFF00FF; // Magenta
            break;
            
        case 8: // Digital Matrix
            *lineThickness = 1;
            *lineType = 0; // Solid
            *boxAlpha = 0.2f;
            *fontSize = 1.0f;
            *fontType = 0;
            *maskAlpha = 0.2f;
            *maskContrast = 1.0f;
            *boxColor = 0xFF00FF00; // ARGB: Green (00FF00)
            *textColor = 0xFF00FF00;
            *maskEdgeThickness = 1;
            *maskEdgeType = 0; // Solid
            *maskEdgeColor = 0xFF00FF00; // Green
            break;
            
        case 9: // Quantum Spectrum
            *lineThickness = 3;
            *lineType = 2; // Corner brackets
            *boxAlpha = 0.3f;
            *fontSize = 1.0f;
            *fontType = 0;
            *maskAlpha = 0.3f;
            *maskContrast = 1.0f;
            *boxColor = 0xFFFF8000; // ARGB: Orange (FF8000)
            *textColor = 0xFFFFFFFF;
            *maskEdgeThickness = 3;
            *maskEdgeType = 0; // Solid
            *maskEdgeColor = 0xFFFF8000; // Orange
            break;
            
        default:
            // Default style parameters
            *lineThickness = 2;
            *lineType = 0; // Solid
            *boxAlpha = 0.3f;
            *fontSize = 1.0f;
            *fontType = 0;
            *maskAlpha = 0.3f;
            *maskContrast = 1.0f;
            *boxColor = 0xFF00FF00; // ARGB: Green (00FF00)
            *textColor = 0xFFFFFFFF;
            *maskEdgeThickness = 2;
            *maskEdgeType = 0; // Solid
            *maskEdgeColor = 0xFFFF0000; // Red
            break;
    }
    
    // Log the returned parameters for debugging
    __android_log_print(ANDROID_LOG_DEBUG, "YoloStyle", 
        "Style %d parameters: lineThickness=%d, lineType=%d, boxColor=0x%08X, textColor=0x%08X",
        styleId, *lineThickness, *lineType, *boxColor, *textColor);
    
    // Restore original style
    this->setDetectionStyle(originalStyle);
    
    return true;
}

/**
 * 用来加载Yolo模型
 * @param mgr
 * @param modeltype String identifying the model variant (e.g., "yolov8n", "yolov8s", "yolov8s-seg", "yolov5s", "yolov7", etc.)
 * @param _target_size
 * @param _mean_vals
 * @param _norm_vals
 * @param use_gpu
 * @return
 */
int Yolo::load(AAssetManager *mgr, const char *modeltype, int _target_size, const float *_mean_vals,
               const float *_norm_vals, bool use_gpu) {
    // Store the model type
    model_type = modeltype;

    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    ncnn::set_cpu_powersave(2);
    // ncnn::set_omp_num_threads(ncnn::get_big_cpu_count()); // Disable OpenMP for debugging
    ncnn::set_omp_num_threads(1); // Use only one thread for debugging
    yolo.opt = ncnn::Option();
#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Vulkan compute %s", use_gpu ? "enabled" : "disabled");
#endif
    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    // 检查是否为OIV数据集模型 (模型名称中包含"oiv" 或 "_oiv")
    std::string mt = modeltype;
    __android_log_print(ANDROID_LOG_DEBUG, "YoloLoad", "Checking model name: %s for OIV dataset", mt.c_str());
    
    // 将模型名称转为小写以实现不区分大小写的匹配
    std::string mt_lower = mt;
    for (char& c : mt_lower) {
        c = std::tolower(c);
    }
    
    // 检查各种可能的OIV模型命名格式
    bool is_oiv_model = (mt_lower.find("oiv") != std::string::npos);
    
    __android_log_print(ANDROID_LOG_DEBUG, "YoloLoad", "OIV model detection: name=%s, lowercase=%s, is_oiv=%d", 
                        mt.c_str(), mt_lower.c_str(), is_oiv_model ? 1 : 0);
    
    if (is_oiv_model) {
        // 设置为OIV数据集
        dataset_type = DATASET_OIV;
        // 切换全局类别名称指针到OIV类别
        class_names = oiv_class_names;
        __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "Detected OIV dataset model: %s, dataset_type=%d", modeltype, dataset_type);
    } else {
        // 默认为COCO数据集
        dataset_type = DATASET_COCO;
        // 切换全局类别名称指针到COCO类别
        class_names = coco_class_names;
        __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "Using COCO dataset for model: %s, dataset_type=%d", modeltype, dataset_type);
    }

    // 检查是否为分割模型
    if (mt.find("-seg") != std::string::npos) {
        is_segmentation_model = true;
        
        // Different segmentation models have different numbers of mask coefficients
        if (mt.find("yolov8n-seg") != std::string::npos) {
            num_mask_coeffs = 32;  // YOLOv8n-seg typically uses 32 coefficients
            __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "Detected YOLOv8n-seg model, using %d mask coefficients", num_mask_coeffs);
        } else if (mt.find("yolov8s-seg") != std::string::npos) {
            num_mask_coeffs = 32;  // YOLOv8s-seg uses 32 coefficients
            __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "Detected YOLOv8s-seg model, using %d mask coefficients", num_mask_coeffs);
        } else {
            num_mask_coeffs = 32;  // Default for other segmentation models
            __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "Generic segmentation model, using %d mask coefficients", num_mask_coeffs);
        }
        
        // Update output names based on NCNN log hints
        // Try different configurations of output names
        det_output_name = "output"; // Primary guess
        seg_output_name = "seg";    // Primary guess
        
        __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "Loading SEGMENTATION model. Output layers: det='%s', seg='%s'", 
                           det_output_name.c_str(), seg_output_name.c_str());
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Model %s is segmentation model: %d", modeltype, is_segmentation_model ? 1 : 0);
    }
    else {
        is_segmentation_model = false;
        num_mask_coeffs = 0;
        det_output_name = "output"; // Default for detection models
        seg_output_name = "";
        __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "Loading DETECTION model. Output layer: det='%s'", det_output_name.c_str());
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Model %s is segmentation model: %d", modeltype, is_segmentation_model ? 1 : 0);
    }

    // 支持多种命名格式和文件扩展名
    const char* extensions[][2] = {
        {".param", ".bin"},                // 标准命名
        {".ncnn.param", ".ncnn.bin"},      // 官方导出命名
        {"_ncnn_model.param", "_ncnn_model.bin"}, // ultralytics导出命名
        {"/model.ncnn.param", "/model.ncnn.bin"}  // 带子目录模式
    };
    
    char parampath[256];
    char modelpath[256];
    bool model_loaded = false;
    
    // 尝试所有可能的扩展名组合
    for (const auto& ext : extensions) {
        sprintf(parampath, "%s%s", modeltype, ext[0]);
        sprintf(modelpath, "%s%s", modeltype, ext[1]);
        
        __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "尝试加载模型: %s, %s", parampath, modelpath);
        
        // 尝试加载模型
        int ret1 = yolo.load_param(mgr, parampath);
        if (ret1 != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "YoloLoad", "加载参数文件 %s 失败，错误: %d", parampath, ret1);
            continue;
        }
        
        int ret2 = yolo.load_model(mgr, modelpath);
        if (ret2 != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "YoloLoad", "加载模型文件 %s 失败，错误: %d", modelpath, ret2);
            continue;
        }
        
        model_loaded = true;
        __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "成功加载模型: %s, %s", parampath, modelpath);
        break;
    }
    
    if (!model_loaded) {
        __android_log_print(ANDROID_LOG_ERROR, "YoloLoad", "所有模型格式尝试均失败");
        return -1;
    }
    
    // Diagnose model structure to help with debugging
    std::vector<ncnn::Layer*> layers = yolo.layers();
    __android_log_print(ANDROID_LOG_DEBUG, "YoloLoad", "Model loaded with %d layers", (int)layers.size());
    
    // Find and log input blob names
    std::vector<int> input_indexes = yolo.input_indexes();
    std::vector<int> output_indexes = yolo.output_indexes();
    
    __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "Model has %d inputs and %d outputs", 
                       (int)input_indexes.size(), (int)output_indexes.size());
                       
    for (int i = 0; i < input_indexes.size(); i++) {
        if (i < yolo.blobs().size()) {
            const ncnn::Blob& blob = yolo.blobs()[input_indexes[i]];
            __android_log_print(ANDROID_LOG_DEBUG, "YoloLoad", "Input blob %d: name='%s', shape=[%d,%d,%d]", 
                               i, blob.name.c_str(), blob.shape.w, blob.shape.h, blob.shape.c);
        }
    }
    
    for (int i = 0; i < output_indexes.size(); i++) {
        if (i < yolo.blobs().size()) {
            const ncnn::Blob& blob = yolo.blobs()[output_indexes[i]];
            __android_log_print(ANDROID_LOG_DEBUG, "YoloLoad", "Output blob %d: name='%s', shape=[%d,%d,%d]", 
                               i, blob.name.c_str(), blob.shape.w, blob.shape.h, blob.shape.c);
        }
    }

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    __android_log_print(ANDROID_LOG_INFO, "ncnn", "yolo->load success!");
    return 0;
}

/**
 * 输入图片，输出检测结果
 * @param rgb
 * @param objects
 * @param prob_threshold
 * @param nms_threshold
 * @return
 */
int Yolo::detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold,
                 float nms_threshold) {
    int width = rgb.cols;
    int height = rgb.rows;
    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
        h = target_size;
        w = w * scale;
    }
    
    // Safe image preprocessing - without try-catch
    ncnn::Mat in;
    in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);
    if (in.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed during image preprocessing");
        return -1;
    }
    
    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    
    // Copy border without try-catch
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                        ncnn::BORDER_CONSTANT, 114.f);  // 使用114填充，与YOLOv8训练一致
    if (in_pad.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed during image padding");
        return -1;
    }
    
    in_pad.substract_mean_normalize(0, norm_vals);

    // 创建推理器
    ncnn::Extractor ex = yolo.create_extractor();
    bool input_found = false;
    
    // 动态适配多种可能的输入层名称
    const char* input_names[] = {"images", "in0", "input", "data", "input.1", "x", "input0"};
    for (const char* name : input_names) {
        int ret = ex.input(name, in_pad);
        if (ret == 0) {
            input_found = true;
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "成功使用输入层名称: '%s'", name);
            break;
        }
    }
    
    if (!input_found) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "所有输入层名称尝试失败");
        return -1;
    }

    std::vector<Object> proposals;
    ncnn::Mat out;
    ncnn::Mat proto;  // For mask prototypes
    ncnn::Mat mask_coeffs_tensor; // For mask coefficients
    std::string coeff_output_name = ""; // Store name of coefficient tensor
    bool det_output_success = false;
    bool seg_output_success = false;
    bool coeff_output_success = false;

    // Extract outputs based on model type
    if (is_segmentation_model) {
        // 1. Try to extract DETECTION output
        const char* det_fallbacks[] = {"output", "out0", "detection", "output0", "382", "383"}; // 添加更多可能的输出名称
        for (const char* name : det_fallbacks) {
            int ret = ex.extract(name, out);
            if (ret == 0) {
                det_output_success = true;
                det_output_name = name; // Remember working name
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Successfully used '%s' for detection output", name);
                break;
            }
        }
        if (!det_output_success) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to extract detection output for segmentation model");
            return -1;
        }

        // 2. Try to extract MASK COEFFICIENTS output
        const char* coeff_fallbacks[] = {"out1", "mask_coeff", "384", "385"}; // 添加更多可能的系数输出层名称
        for (const char* name : coeff_fallbacks) {
            // Skip names already used
            if (std::string(name) == det_output_name) continue;

            int ret = ex.extract(name, mask_coeffs_tensor);
            if (ret == 0) {
                coeff_output_success = true;
                coeff_output_name = name; // Remember working name
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Successfully used '%s' for mask coefficient output (C=%d, H=%d, W=%d)",
                                   name, mask_coeffs_tensor.c, mask_coeffs_tensor.h, mask_coeffs_tensor.w);
                break;
            }
        }
        // It's okay if coefficients are not separate; generate_proposals handles combined output.
        if (!coeff_output_success) {
             __android_log_print(ANDROID_LOG_WARN, "ncnn", "Could not find separate mask coefficient tensor. Assuming coefficients are in detection output.");
        }

        // 3. Try to extract MASK PROTOTYPES output
        const char* seg_fallbacks[] = {"seg", "proto", "mask_proto", "out2", "386", "387"}; // 添加更多可能的原型层名称
        for (const char* name : seg_fallbacks) {
            // Skip names already used for detection or coeffs
            if (std::string(name) == det_output_name) continue;
            if (coeff_output_success && std::string(name) == coeff_output_name) continue;

            int ret = ex.extract(name, proto); // Extract into 'proto' variable
            if (ret == 0) {
                seg_output_success = true;
                seg_output_name = name; // Remember working name
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Successfully used '%s' for segmentation prototype output", name);
                __android_log_print(ANDROID_LOG_INFO, "ncnn", "Segmentation prototype dimensions: C=%d, H=%d, W=%d",
                                  proto.c, proto.h, proto.w);
                break;
            }
        }
        if (!seg_output_success) {
            __android_log_print(ANDROID_LOG_WARN, "ncnn", "Failed to extract segmentation prototype output. Disabling segmentation for this run.");
            is_segmentation_model = false; // Fall back to detection only
        } else {
            // Store the extracted prototypes
            mask_protos = proto; // Assign the successfully extracted prototypes
        }
    } else {
        // Regular detection model - extract only detection output
        bool success = false;
        const char* fallbacks[] = {"output", "out0", "detection", "output0", "382", "383"};
        for (const char* name : fallbacks) {
            int ret = ex.extract(name, out);
            if (ret == 0) {
                success = true;
                det_output_name = name;
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Successfully used '%s' for detection output", name);
                break;
            }
        }
        if (!success) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "All output name attempts failed for detection model");
            return -1;
        }
    }
    
    // 检查输出张量的形状，并进行必要的调整
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "检测到的输出张量形状: dims=%d, w=%d, h=%d, c=%d", 
                      out.dims, out.w, out.h, out.c);
                      
    // 自适应处理不同形状的输出
    if (out.dims == 2) {
        if (out.w > out.h) {
            // YOLOv8导出格式常见情况: (h, w) -> (w, h)，需要转置
            __android_log_print(ANDROID_LOG_INFO, "ncnn", "检测到需要转置的格式: %dx%d -> %dx%d", 
                            out.w, out.h, out.h, out.w);
            out = out.reshape(out.h, out.w);
        }
    } else if (out.dims == 3) {
        // 处理3D张量格式 (批次, 通道, 检测数)
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "检测到3D输出格式: %dx%dx%d", 
                        out.c, out.h, out.w);
        if (out.c == 1) {
            // 单个批次情况，移除批次维度
            out = out.reshape(out.h, out.w);
            __android_log_print(ANDROID_LOG_INFO, "ncnn", "转换3D输出为2D: %dx%d", out.h, out.w);
        } else {
            // 复杂情况，尝试转置处理
            __android_log_print(ANDROID_LOG_WARN, "ncnn", "未知的3D输出结构，尝试其他处理方式");
            // 根据实际测试情况可能需要更多特殊处理
        }
    }

    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);

    // Get number of classes based on dataset type
    int num_class = 80; // Default for COCO
    if (dataset_type == DATASET_OIV) {
        // Count OIV classes - using the size of the array
        num_class = sizeof(oiv_class_names) / sizeof(oiv_class_names[0]);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using OIV dataset with %d classes", num_class);
    }

    // Pass segmentation flag and coeff count to generate_proposals
    // generate_proposals will handle combined (det+coeff) or separate (det only) cases
    generate_proposals(grid_strides, out, prob_threshold, proposals,
                    is_segmentation_model && !coeff_output_success, // Only expect coeffs in 'out' if coeff tensor wasn't found
                    num_mask_coeffs, num_class);

    // 应用类别过滤 - 如果启用了过滤
    if (class_filtering_enabled && !enabled_class_ids.empty()) {
        std::vector<Object> filtered_proposals;
        for (const auto& obj : proposals) {
            if (enabled_class_ids.find(obj.label) != enabled_class_ids.end()) {
                filtered_proposals.push_back(obj);
            }
        }
        
        if (filtered_proposals.empty() && !proposals.empty()) {
            __android_log_print(ANDROID_LOG_INFO, "YoloClasses", "过滤后没有检测结果，原始检测到 %zu 个对象", 
                               proposals.size());
        }
        
        proposals = filtered_proposals;
    }

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    objects.resize(count);

    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // --- Assign Mask Coefficients (if separate tensor was found) ---
    if (is_segmentation_model && coeff_output_success && !mask_coeffs_tensor.empty()) {
        // Check consistency: number of proposals should match height of coeff tensor if C=1 or W=num_coeffs
        int num_proposals_in_coeffs = 0;
        if (mask_coeffs_tensor.dims == 2 && mask_coeffs_tensor.w == num_mask_coeffs) {
            num_proposals_in_coeffs = mask_coeffs_tensor.h; // Shape [N, 32]
        } else if (mask_coeffs_tensor.dims == 3 && mask_coeffs_tensor.c == 1 && mask_coeffs_tensor.w == num_mask_coeffs) {
             num_proposals_in_coeffs = mask_coeffs_tensor.h; // Shape [H, 32, 1] -> view as [H, 32]
        } else if (mask_coeffs_tensor.dims == 3 && mask_coeffs_tensor.c == num_mask_coeffs) {
            num_proposals_in_coeffs = mask_coeffs_tensor.h * mask_coeffs_tensor.w; // Shape [32, H, W] -> view as [32, N] -> transpose
             __android_log_print(ANDROID_LOG_WARN, "ncnn", "Coefficient tensor shape [C, H, W] requires transpose/reshape not fully implemented yet.");
        } else {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Unsupported mask coefficient tensor shape: Dims=%d, C=%d, H=%d, W=%d",
                              mask_coeffs_tensor.dims, mask_coeffs_tensor.c, mask_coeffs_tensor.h, mask_coeffs_tensor.w);
        }

        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Assigning coefficients from separate tensor (H=%d, W=%d)", mask_coeffs_tensor.h, mask_coeffs_tensor.w);

        for (int i = 0; i < count; i++) {
            Object& obj = objects[i]; // Use reference
            int gindex = obj.gindex; // Grid index from proposal generation

            // Check if gindex is valid for the coefficient tensor's height dimension
            if (gindex >= 0 && gindex < num_proposals_in_coeffs && gindex < mask_coeffs_tensor.h) {
                 // Assuming coeffs are [H, W] where W=num_mask_coeffs or [H, W, 1] where W=num_mask_coeffs
                const float* coeff_ptr = mask_coeffs_tensor.row(gindex);

                if (coeff_ptr) {
                    obj.mask_coeffs.resize(num_mask_coeffs);
                    // Check bounds before copying
                    int coeffs_to_copy = std::min(num_mask_coeffs, mask_coeffs_tensor.w);
                    for (int c = 0; c < coeffs_to_copy; c++) {
                         obj.mask_coeffs[c] = coeff_ptr[c];
                    }
                     // Zero pad if necessary (should not happen if W==num_mask_coeffs)
                    for (int c = coeffs_to_copy; c < num_mask_coeffs; c++) {
                         obj.mask_coeffs[c] = 0.0f;
                    }
                    __android_log_print(ANDROID_LOG_VERBOSE, "ncnn", "Assigned %d coeffs to object %d (gindex %d)", num_mask_coeffs, i, gindex);
                } else {
                     __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Object %d (gindex %d): Failed to get coefficient pointer from tensor.", i, gindex);
                    obj.mask_coeffs.clear(); // Ensure invalid state
                }
            } else {
                 __android_log_print(ANDROID_LOG_WARN, "ncnn", "Object %d: gindex %d out of bounds for coefficient tensor height %d. Clearing coeffs.", i, gindex, mask_coeffs_tensor.h);
                obj.mask_coeffs.clear(); // Clear coeffs if index is invalid
            }
        }
    } else if (is_segmentation_model && !coeff_output_success) {
         __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Coefficients expected in detection output (from generate_proposals).");
         // Coefficients should already be in objects[i].mask_coeffs from generate_proposals
    }

    // --- Generate Masks for all detected objects (if segmentation model & prototypes found) ---
    if (is_segmentation_model && seg_output_success && !objects.empty() && !mask_protos.empty())
    {
        // Verify mask_protos dimensions and reshape if needed for yolov8s-seg
        bool valid_protos = false;
        
        // Handle yolov8s-seg case where proto tensor is [32, 5120] or similar (dims=2)
        if (mask_protos.dims == 2 && mask_protos.h == num_mask_coeffs) {
            __android_log_print(ANDROID_LOG_INFO, "ncnn", "Detected yolov8s-seg 2D tensor format: [%d, %d]",
                              mask_protos.h, mask_protos.w);
            
            // For yolov8s-seg, we'll handle the 2D tensor directly instead of reshaping
            // This approach works with the tensor as-is without requiring reshape
            
            // Calculate proto size based on tensor shape
            int proto_h = 0;
            int proto_w = 0;
            
            // Specific handling for common sizes in YOLOv8s-seg
            if (mask_protos.w == 5120) {
                // For 5120, the layout is likely [32, 160, 32]
                proto_h = 160;
                proto_w = 32;
            } else if (mask_protos.w == 6400) {
                // For 6400, the layout might be [32, 160, 40]
                proto_h = 160;
                proto_w = 40;
            } else {
                // Try to infer reasonable dimensions
                // For almost square dimensions, try to make it square-ish
                int proto_size = (int)sqrt(mask_protos.w);
                if (std::abs(proto_size * proto_size - mask_protos.w) < 0.1 * mask_protos.w) {
                    // Close enough to square
                    proto_h = proto_size;
                    proto_w = proto_size;
                } else {
                    // Default assumption for YOLOv8 models with 640x640 input: feature map is 1/4 of input
                    proto_h = 160;
                    proto_w = mask_protos.w / 160;
                    if (proto_w <= 0) {
                        // Last resort default
                        proto_h = 80;
                        proto_w = mask_protos.w / 80;
                        if (proto_w <= 0) proto_w = 32; // absolute fallback
                    }
                }
            }
            
            // Mark as valid - we'll process the 2D tensor directly
            valid_protos = true;
            
            // Log special handling
            __android_log_print(ANDROID_LOG_INFO, "ncnn", "Using special 2D mask processing for yolov8s-seg with proto dimensions: H=%d, W=%d",
                              proto_h, proto_w);
            
            // Override the regular mask processing after this check
            if (valid_protos) {
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Processing masks with 2D protos: rows=%d, cols=%d", 
                                   num_mask_coeffs, mask_protos.w);

                // Process each detection
                for (int i = 0; i < count; i++) {
                    // Check if mask coefficients are valid
                    if (objects[i].mask_coeffs.empty() || objects[i].mask_coeffs.size() != num_mask_coeffs) {
                        __android_log_print(ANDROID_LOG_WARN, "ncnn", "Object %d: Invalid or missing mask coefficients (%zu != %d). Skipping mask.", 
                                          i, objects[i].mask_coeffs.size(), num_mask_coeffs);
                        continue; 
                    }

                    // --- Processing masks differently for 2D tensor format ---
                     
                    // Get coefficient vector for this detection
                    std::vector<float>& coeffs = objects[i].mask_coeffs;

                    // Debug coefficient vector
                    __android_log_print(ANDROID_LOG_DEBUG, "YoloMask", "Object %d: Coefficients size=%zu, first val=%.4f", 
                                        i, coeffs.size(), coeffs.empty() ? 0.0f : coeffs[0]);

                    // Check if coefficients are very small/empty, which might indicate they weren't properly extracted
                    bool all_zero = true;
                    float coeff_sum = 0.0f;
                    float coeff_abs_max = 0.0f;
                    for (size_t c = 0; c < coeffs.size(); c++) {
                        coeff_sum += coeffs[c];
                        coeff_abs_max = std::max(coeff_abs_max, std::abs(coeffs[c]));
                        if (std::abs(coeffs[c]) > 1e-6) {
                            all_zero = false;
                        }
                    }

                    if (all_zero || coeff_abs_max < 0.01f) {
                        __android_log_print(ANDROID_LOG_WARN, "YoloMask", "Object %d: Very weak coefficients (max=%.4f, sum=%.4f). Mask may be poor quality.", 
                                            i, coeff_abs_max, coeff_sum);
                    }

                    // Create a mask of appropriate size for the detection's bounding box
                    const cv::Rect_<float>& bbox = objects[i].rect;
                    cv::Mat mask_float(round(bbox.height), round(bbox.width), CV_32F, cv::Scalar(0));
                    
                    // Calculate scaling factors between original image and prototype feature map
                    float ratio_w = (float)proto_w / in_pad.w;
                    float ratio_h = (float)proto_h / in_pad.h;
                    
                    // Get prototype region corresponding to this bounding box
                    cv::Rect_<float> bbox_in_padded;
                    bbox_in_padded.x = (bbox.x * scale) + (wpad / 2);
                    bbox_in_padded.y = (bbox.y * scale) + (hpad / 2);
                    bbox_in_padded.width = bbox.width * scale;
                    bbox_in_padded.height = bbox.height * scale;
                    
                    cv::Rect_<float> proto_roi;
                    proto_roi.x = bbox_in_padded.x * ratio_w;
                    proto_roi.y = bbox_in_padded.y * ratio_h;
                    proto_roi.width = bbox_in_padded.width * ratio_w;
                    proto_roi.height = bbox_in_padded.height * ratio_h;
                    
                    // Ensure ROI is within bounds
                    proto_roi.x = std::max(0.f, proto_roi.x);
                    proto_roi.y = std::max(0.f, proto_roi.y);
                    proto_roi.width = std::min(proto_w - proto_roi.x, proto_roi.width);
                    proto_roi.height = std::min(proto_h - proto_roi.y, proto_roi.height);
                    
                    if (proto_roi.width <= 0 || proto_roi.height <= 0) {
                        __android_log_print(ANDROID_LOG_WARN, "ncnn", "Object %d: Invalid proto ROI. Skipping mask.", i);
                        continue;
                    }
                    
                    // Convert to integer coordinates
                    cv::Rect roi_int(round(proto_roi.x), round(proto_roi.y), 
                                   round(proto_roi.width), round(proto_roi.height));
                    
                    // Calculate mask directly from coefficients and prototype
                    // For each pixel in the mask
                    for (int y = 0; y < mask_float.rows; y++) {
                        for (int x = 0; x < mask_float.cols; x++) {
                            // Map this pixel to prototype coordinates
                            int proto_x = roi_int.x + (x * roi_int.width / mask_float.cols);
                            int proto_y = roi_int.y + (y * roi_int.height / mask_float.rows);
                            
                            // Calculate mask value using linear combination of coefficients and prototypes
                            float mask_val = 0.0f;
                            for (int c = 0; c < num_mask_coeffs; c++) {
                                // Get the prototype value at this location
                                // For 2D tensor, we access it as mask_protos[c][proto_y*proto_w + proto_x]
                                int proto_idx = proto_y*proto_w + proto_x;
                                if (proto_idx < mask_protos.w) {
                                    const float* row_ptr = mask_protos.row(c);
                                    float proto_val = row_ptr[proto_idx];
                                    mask_val += coeffs[c] * proto_val;
                                }
                            }
                            
                            // Apply sigmoid
                            mask_float.at<float>(y, x) = sigmoid(mask_val);
                        }
                    }
                    
                    // Apply threshold to create binary mask
                    bool is_s_seg = model_type.find("yolov8s-seg") != std::string::npos;
                    float applied_threshold = is_s_seg ? (mask_threshold * 0.5f) : mask_threshold;
                    cv::Mat binary_mask = mask_float > applied_threshold;
                    
                    // Apply additional processing for yolov8s-seg to improve the mask quality
                    if (is_s_seg) {
                        // Convert to 8-bit for morphology operations
                        cv::Mat mask_8u;
                        binary_mask.convertTo(mask_8u, CV_8U, 255);
                        
                        // Get size-appropriate structuring element (kernel)
                        int small_morph_size = std::max(1, std::min(3, (int)(std::min(mask_8u.cols, mask_8u.rows) / 60)));
                        cv::Mat small_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
                                                                     cv::Size(2*small_morph_size+1, 2*small_morph_size+1),
                                                                     cv::Point(small_morph_size, small_morph_size));
                        
                        // Create larger kernel for more aggressive operations
                        int large_morph_size = std::max(2, std::min(7, (int)(std::min(mask_8u.cols, mask_8u.rows) / 40)));
                        cv::Mat large_element = cv::getStructuringElement(cv::MORPH_RECT, 
                                                                      cv::Size(2*large_morph_size+1, 1), // Horizontal kernel
                                                                      cv::Point(large_morph_size, 0));
                                                  
                        // First close with horizontal kernel to connect lines
                        cv::morphologyEx(mask_8u, mask_8u, cv::MORPH_CLOSE, large_element);
                        
                        // Then close with circular kernel to smooth out and fill remaining small gaps
                        cv::morphologyEx(mask_8u, mask_8u, cv::MORPH_CLOSE, small_element);
                        
                        // Apply dilate to expand the mask slightly
                        cv::dilate(mask_8u, mask_8u, small_element, cv::Point(-1,-1), 1);
                        
                        // For yolov8s-seg, use a lower threshold to keep more of the mask
                        binary_mask = mask_8u > 100; // More permissive threshold after morphology
                        
                        __android_log_print(ANDROID_LOG_INFO, "YoloMask", "Applied enhanced morphological processing for yolov8s-seg mask");
                    }
                    
                    // Check if mask has enough foreground pixels
                    if (cv::countNonZero(binary_mask) < 10) {
                        __android_log_print(ANDROID_LOG_WARN, "YoloDraw", "Mask has too few non-zero pixels. Creating placeholder mask.");
                        binary_mask = cv::Mat::zeros(binary_mask.size(), CV_8U);
                        cv::ellipse(binary_mask, 
                                   cv::Point(binary_mask.cols/2, binary_mask.rows/2),
                                   cv::Size(binary_mask.cols/3, binary_mask.rows/3),
                                   0, 0, 360, cv::Scalar(255), -1);
                    }
                    
                    // Store the mask
                    objects[i].mask = binary_mask;
                    
                    // Log mask creation success
                    int nonzero = cv::countNonZero(binary_mask);
                    int total = binary_mask.rows * binary_mask.cols;
                    float coverage = (float)nonzero / total * 100.0f;
                    __android_log_print(ANDROID_LOG_INFO, "YoloMask", "Generated mask for yolov8s-seg: %dx%d, nonzero=%d (%.1f%%)",
                                      binary_mask.cols, binary_mask.rows, nonzero, coverage);
                }
                
                // Skip the standard mask processing by returning
                return 0;
            }
        }
        // Original yolov8n-seg case where proto tensor is [32, H, W]
        else if (mask_protos.dims == 3 && mask_protos.c == num_mask_coeffs) {
            valid_protos = true;
        }
        
        if (!valid_protos) {
             __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Invalid mask_protos dimensions: dims=%d, c=%d, expected_c=%d. Check model output layer for prototypes.",
                               mask_protos.dims, mask_protos.c, num_mask_coeffs);
            // Don't continue with invalid prototypes
        } else {
            int proto_h = mask_protos.h;
            int proto_w = mask_protos.w;
            
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Processing masks with protos shape: C=%d, H=%d, W=%d", 
                                num_mask_coeffs, proto_h, proto_w);

            // Create a CV Mat wrapper for mask_protos (no data copy)
            // Reshape from [C, H, W] to [C, H*W] for matmul
            cv::Mat protos_mat(num_mask_coeffs, proto_h * proto_w, CV_32F, (float*)mask_protos.data);

            for (int i = 0; i < count; i++)
            {
                // Check if mask coefficients are valid
                if (objects[i].mask_coeffs.empty() || objects[i].mask_coeffs.size() != num_mask_coeffs) {
                    __android_log_print(ANDROID_LOG_WARN, "ncnn", "Object %d: Invalid or missing mask coefficients (%zu != %d). Skipping mask.", 
                                        i, objects[i].mask_coeffs.size(), num_mask_coeffs);
                    continue; 
                }

                // --- Matrix Multiplication: coeffs * protos ---
                // Coeffs: [1, num_mask_coeffs]
                // Protos: [num_mask_coeffs, proto_h * proto_w]
                // Result: [1, proto_h * proto_w]
                cv::Mat coeffs_mat(1, num_mask_coeffs, CV_32F, objects[i].mask_coeffs.data());
                cv::Mat mask_pred_mat = coeffs_mat * protos_mat; // [1, proto_h * proto_w]

                // Reshape result to [proto_h, proto_w]
                cv::Mat mask_pred_cv = mask_pred_mat.reshape(1, proto_h); // [proto_h, proto_w]

                // Apply sigmoid
                mat_sigmoid(mask_pred_cv); // Apply sigmoid inplace

                // --- Upsample and Crop Mask ---
                // Get the bounding box in the *original* image coordinates
                const cv::Rect_<float>& bbox = objects[i].rect;
                
                // Calculate bounding box in the *padded/resized* input coordinates (in_pad)
                // This is needed to crop the correct region from the full prototype mask
                cv::Rect_<float> bbox_in_padded;
                bbox_in_padded.x = (bbox.x * scale) + (wpad / 2);
                bbox_in_padded.y = (bbox.y * scale) + (hpad / 2);
                bbox_in_padded.width = bbox.width * scale;
                bbox_in_padded.height = bbox.height * scale;

                // Calculate the corresponding region in the prototype mask (which matches in_pad dimensions / 4 or similar downscale)
                // Assuming mask_protos corresponds to in_pad dimensions divided by a factor (e.g., 4 for stride 32 output)
                // Find the downscaling factor (proto vs in_pad dimensions)
                // It seems proto dims (H, W) might correspond to in_pad dims / 4 ? Check model structure.
                // For YOLOv8, proto size is typically input_size/4 (e.g., 640 -> 160)
                float ratio_w = (float)proto_w / in_pad.w;
                float ratio_h = (float)proto_h / in_pad.h;

                cv::Rect_<float> proto_roi;
                proto_roi.x = bbox_in_padded.x * ratio_w;
                proto_roi.y = bbox_in_padded.y * ratio_h;
                proto_roi.width = bbox_in_padded.width * ratio_w;
                proto_roi.height = bbox_in_padded.height * ratio_h;

                // Ensure ROI is within prototype mask bounds
                proto_roi &= cv::Rect_<float>(0, 0, proto_w, proto_h);

                if (proto_roi.width <= 0 || proto_roi.height <= 0) {
                    __android_log_print(ANDROID_LOG_WARN, "ncnn", "Object %d: Calculated proto ROI has zero or negative size [%.1f x %.1f]. Skipping mask.", 
                                        i, proto_roi.width, proto_roi.height);
                     continue; // Skip if ROI is invalid
                }

                // Crop the relevant part from the full prediction mask
                cv::Mat cropped_mask_pred;
                cv::Rect roi_int(round(proto_roi.x), round(proto_roi.y), round(proto_roi.width), round(proto_roi.height));
                
                // Additional boundary check before cropping
                 if (roi_int.x < 0 || roi_int.y < 0 || roi_int.x + roi_int.width > mask_pred_cv.cols || roi_int.y + roi_int.height > mask_pred_cv.rows) {
                     __android_log_print(ANDROID_LOG_WARN, "ncnn", "Object %d: Proto ROI [%d, %d, %d, %d] exceeds mask bounds [%d, %d]. Adjusting.", 
                                         i, roi_int.x, roi_int.y, roi_int.width, roi_int.height, mask_pred_cv.cols, mask_pred_cv.rows);
                     // Adjust ROI to fit within bounds
                     roi_int &= cv::Rect(0, 0, mask_pred_cv.cols, mask_pred_cv.rows);
                     if (roi_int.width <= 0 || roi_int.height <= 0) {
                         __android_log_print(ANDROID_LOG_WARN, "ncnn", "Object %d: Adjusted ROI has zero size. Skipping mask.", i);
                         continue;
                     }
                 }

                mask_pred_cv(roi_int).copyTo(cropped_mask_pred);


                // Resize the cropped mask to the original bounding box size
                cv::Mat final_mask_float;
                // 使用条件判断替代try-catch
                bool resize_ok = !cropped_mask_pred.empty() && 
                                round(bbox.width) > 0 && round(bbox.height) > 0;
                                
                if (resize_ok) {
                cv::resize(cropped_mask_pred, final_mask_float, 
                           cv::Size(round(bbox.width), round(bbox.height)), 
                           0, 0, cv::INTER_LINEAR); // Use INTER_LINEAR or INTER_NEAREST
                              
                    // 检查resize结果
                    resize_ok = !final_mask_float.empty() && 
                               final_mask_float.size() == cv::Size(round(bbox.width), round(bbox.height));
                }
                
                if (!resize_ok) {
                    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to resize mask");
                    continue; // Skip this mask if resize fails
                }

                // Instead of direct thresholding to binary, store the float mask
                cv::Mat thresholded_mask;
                
                // Special handling for yolov8s-seg
                bool is_s_seg = model_type.find("yolov8s-seg") != std::string::npos;
                float applied_threshold = is_s_seg ? (mask_threshold * 0.5f) : mask_threshold;
                thresholded_mask = final_mask_float > applied_threshold;
                
                // Special case - if mask has no foreground pixels, create a simple elliptical placeholder
                // This ensures visibility even when thresholding fails
                if (cv::countNonZero(thresholded_mask) < 10) {
                    __android_log_print(ANDROID_LOG_WARN, "YoloDraw", "Mask has too few non-zero pixels (%d). Creating placeholder mask.", 
                                       cv::countNonZero(thresholded_mask));
                    thresholded_mask = cv::Mat::zeros(thresholded_mask.size(), CV_8U);
                    cv::ellipse(thresholded_mask, 
                               cv::Point(thresholded_mask.cols/2, thresholded_mask.rows/2),
                               cv::Size(thresholded_mask.cols/3, thresholded_mask.rows/3),
                               0, 0, 360, cv::Scalar(255), -1);
                }
                
                // Count non-zero pixels to check mask quality
                int nonzero_pixels = cv::countNonZero(thresholded_mask);
                int total_pixels = thresholded_mask.rows * thresholded_mask.cols;
                float coverage_percent = (float)nonzero_pixels / total_pixels * 100.0f;
                
                __android_log_print(ANDROID_LOG_INFO, "YoloMask", "Generated mask: %dx%d, nonzero=%d (%.1f%%), min=%.3f, max=%.3f", 
                                  final_mask_float.cols, final_mask_float.rows,
                                  nonzero_pixels, coverage_percent,
                                  *std::min_element(final_mask_float.ptr<float>(0), 
                                                   final_mask_float.ptr<float>(0) + final_mask_float.total()),
                                  *std::max_element(final_mask_float.ptr<float>(0), 
                                                   final_mask_float.ptr<float>(0) + final_mask_float.total()));
                
                // Convert to 8U format (0 or 255) and store
                objects[i].mask = thresholded_mask;
                
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Generated mask for object %d, Size: %dx%d", 
                                    i, objects[i].mask.cols, objects[i].mask.rows);
            }
        }
    }
    
    // 按面积排序对象 (保持不变)
    struct {
        bool operator()(const Object &a, const Object &b) const {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);
    
    // --- 跟踪器更新 --- 
    if (enable_tracking && g_tracker) {
        __android_log_print(ANDROID_LOG_INFO, "YoloTracker", "Using EnhancedTracking with KF: %zu detections in", objects.size());
        
        // 确保掩码尺寸在传入跟踪器前是正确的 (如果启用掩码跟踪)
        if (is_segmentation_model && current_tracking_params.enable_mask_tracking) {
            __android_log_print(ANDROID_LOG_INFO, "YoloTracker", "Checking mask sizes before tracking update (Mask Tracking Enabled)");
            for (auto& obj : objects) {
                if (!obj.mask.empty()) {
                    if (obj.mask.cols != round(obj.rect.width) || obj.mask.rows != round(obj.rect.height)) {
                        if (round(obj.rect.width) > 0 && round(obj.rect.height) > 0) {
                            cv::Mat resized_mask;
                            bool resize_success = false;
                            if (!obj.mask.empty()) {
                                cv::resize(obj.mask, resized_mask, 
                                          cv::Size(round(obj.rect.width), round(obj.rect.height)), 
                                          0, 0, cv::INTER_NEAREST);
                                resize_success = !resized_mask.empty() && resized_mask.size() == cv::Size(round(obj.rect.width), round(obj.rect.height));
                            }
                            if (resize_success) {
                                obj.mask = resized_mask;
                            } else {
                                obj.mask = cv::Mat();
                                __android_log_print(ANDROID_LOG_WARN, "YoloTracker", "Mask resize failed before tracker update, mask cleared.");
                            }
                        } else {
                            obj.mask = cv::Mat();
                            __android_log_print(ANDROID_LOG_WARN, "YoloTracker", "Invalid bbox size (%dx%d) before tracker update, mask cleared.", (int)round(obj.rect.width), (int)round(obj.rect.height));
                        }
                    }
                }
            }
        }
        
        // 更新跟踪器并获取结果
        objects = g_tracker->update(objects);
        
        __android_log_print(ANDROID_LOG_INFO, "YoloTracker", "Tracking result: %zu tracked objects", objects.size());
    } else {
        // 如果跟踪被禁用，清除所有对象的 track_id
        for (auto& obj : objects) {
            obj.track_id = -1;
        }
        __android_log_print(ANDROID_LOG_INFO, "YoloTracker", "Tracking disabled, returning raw detections: %zu", objects.size());
    }

    return 0;
}

// Draw method (Refactored for Enhanced Styles)
int Yolo::draw(cv::Mat &rgb, const std::vector<Object> &objects) {
    static std::vector<Object> prev_objects;
    static cv::Mat prev_mask;
    static bool first_frame = true;
    
    // 使用互斥锁保护绘图操作
    std::lock_guard<std::mutex> lock(draw_mutex);
    
    // 重用上一帧绘制的对象和掩码，减少内存分配开销
    if (first_frame) {
        prev_objects.reserve(100);  // 预分配足够空间
        first_frame = false;
    }
    
    // 清除上一帧的绘制内容（如果需要）
    
    // Draw trajectory visualization if enabled and tracking is active
    if (enable_tracking && g_tracker && current_tracking_params.enable_trajectory_viz) {
        // Get trajectory parameters
        const TrackingParams& params = current_tracking_params;
        
        // Get access to the tracks for visualization
        const std::vector<EnhancedTracking::TrackingObject>& tracks = g_tracker->getTracks();
        
        // Draw trajectories using the helper function in EnhancedTracking
        g_tracker->drawTrajectoryPaths(rgb, tracks, params);
        
        // Log that we're drawing trajectories
        if (!tracks.empty()) {
            __android_log_print(ANDROID_LOG_INFO, "YoloTracker", 
                "Drawing trajectories for %zu tracks with params: length=%d, thickness=%.1f, colorMode=%d", 
                tracks.size(), params.trajectory_length, params.trajectory_thickness, params.trajectory_color_mode);
        }
    }
    
    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];

        // --- Style Configuration (based on detection_style) ---
        cv::Scalar box_color, mask_color, contour_color, text_color, text_bg_color_base;
        float mask_alpha = 0.5f;
        float text_bg_alpha = 0.6f;
        int box_thickness = 2;
        int contour_thickness = 1;
        int corner_length_ratio = 5;
        bool draw_full_box = false;
        bool draw_dashed_box = false;
        bool draw_corners = true;
        bool fill_mask = true;
        bool bracket_corners = false;
        bool draw_double_box = false;
        bool use_gradient_mask = false; // Specific flag for gradient mask fill

        int style_index = this->detection_style;
        int color_index = obj.label % num_tech_colors;

        // Define base colors from class label
        box_color = tech_colors[color_index % num_tech_colors]; // Assign to already declared variable
        cv::Scalar light_color = box_color * 0.7 + cv::Scalar(255, 255, 255) * 0.3;
        cv::Scalar dark_color = box_color * 0.5 + cv::Scalar(0, 0, 0) * 0.5; // Define dark_color here

        // 记录原始风格索引，用于调试
        int original_style_index = style_index;
        
        // 强制应用我们的自定义样式设置，这些设置优先级高于预定义风格
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                          "强制应用自定义样式参数 - 风格ID: %d", style_index);
        
        // 检查是否有自定义框颜色，如果有则完全覆盖预设颜色
        // if (style_box_color != 0) {
        //     // 获取ARGB颜色分量
        //     int r = (style_box_color >> 16) & 0xFF;
        //     int g = (style_box_color >> 8) & 0xFF;
        //     int b = style_box_color & 0xFF;
        //     
        //     // OpenCV使用BGR顺序
        //     box_color = cv::Scalar(b, g, r);
        //     mask_color = box_color;
        //     contour_color = box_color;
        //     
        //     // 派生其他相关颜色
        //     light_color = box_color * 0.7 + cv::Scalar(255, 255, 255) * 0.3;
        //     dark_color = box_color * 0.5 + cv::Scalar(0, 0, 0) * 0.5;
        //     
        //     __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
        //                      "框颜色设置为 BGR(%d,%d,%d)", b, g, r);
        // }
        
        // 检查是否有自定义线型，如果有则完全覆盖预设线型
        if (style_line_type >= 0) {
            // 重置所有线型相关变量
            draw_dashed_box = false;
            draw_full_box = false;
            draw_corners = false;
            bracket_corners = false;
            
            if (style_line_type == 1) { // 虚线
                draw_dashed_box = true;
                draw_full_box = true;
                __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                               "线型设置为虚线 (style_line_type=1)");
            } else if (style_line_type == 2) { // 角标
                draw_corners = true;
                bracket_corners = true;
                __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                               "线型设置为角标 (style_line_type=2)");
            } else { // 实线
                draw_full_box = true;
                __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                               "线型设置为实线 (style_line_type=0)");
            }
        }
        
        // 检查是否有自定义线条粗细
        if (style_line_thickness > 0) {
            box_thickness = style_line_thickness;
            contour_thickness = std::max(1, box_thickness / 2);
            __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                             "线条粗细设置为 %d", box_thickness);
        }
        
        // 检查是否有自定义遮罩透明度
        if (style_mask_alpha >= 0) {
            mask_alpha = style_mask_alpha;
            __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                             "遮罩透明度设置为 %.2f", mask_alpha);
        }
        
        // 现在重新设置风格索引，但我们的自定义设置已经优先应用
        style_index = this->detection_style;
        
        // 在这里记录最终使用的设置，用于调试
        __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                          "最终绘制设置: style_index=%d, box_color=BGR(%d,%d,%d), line_type=%d, line_thickness=%d",
                          style_index, 
                          cvRound(box_color[0]), cvRound(box_color[1]), cvRound(box_color[2]),
                          style_line_type, box_thickness);

        // Store original custom style parameters set via setStyleParameters
        cv::Scalar original_box_color = box_color;
        cv::Scalar original_mask_color = mask_color;
        cv::Scalar original_contour_color = contour_color;
        float original_mask_alpha = mask_alpha;
        cv::Scalar original_text_color = text_color;
        int original_box_thickness = box_thickness;
        bool original_draw_full_box = draw_full_box;
        bool original_draw_dashed_box = draw_dashed_box;
        bool original_draw_corners = draw_corners;
        bool original_bracket_corners = bracket_corners;

        // Log the style index for debugging
        __android_log_print(ANDROID_LOG_DEBUG, "Yolo", "ORIGINAL STYLE INDEX: %d", style_index);

        // Define the 10 styles with refinements
        switch (style_index) {
            case 0: // Style 0: Cyber Grid - Refined
                box_color = tech_colors[color_index % num_tech_colors];
                mask_color = box_color;
                contour_color = cv::Scalar(255, 255, 255);
                mask_alpha = 0.2f;
                text_bg_color_base = cv::Scalar(50, 50, 50); // Dark Gray BG
                text_color = box_color;
                text_bg_alpha = 0.7f; // Slightly more opaque
                box_thickness = 2; // Thicker corners
                contour_thickness = 1;
                corner_length_ratio = 4; // Slightly longer corners
                draw_corners = true;
                draw_full_box = false;
                fill_mask = true;
                // Added: Inner corner brackets
                break;
            case 1: // Style 1: Holographic Glow - Refined
                box_color = tech_colors[color_index % num_tech_colors];
                mask_color = box_color;
                contour_color = cv::Scalar(255, 100, 255);
                mask_alpha = 0.15f;
                text_bg_color_base = box_color; // Class color base for gradient
                text_color = cv::Scalar(255, 255, 255);
                text_bg_alpha = 0.6f; // Control gradient opacity via helper
                box_thickness = 2;
                contour_thickness = 1;
                draw_full_box = true; // Keep solid box for glow base
                draw_corners = false;
                draw_dashed_box = false;
                fill_mask = true;
                // Added: Subtle glow, vertical text gradient
                break;
            case 2: // Style 2: Pulse Glitch - Refined
                box_color = tech_colors[color_index % num_tech_colors];
                mask_color = box_color;
                contour_color = box_color;
                mask_alpha = 0.2f;
                text_bg_color_base = cv::Scalar(40, 40, 40); // Dark Gray BG base
                text_color = box_color;
                text_bg_alpha = 0.7f;
                box_thickness = 3; // Keep thick box
                contour_thickness = 2;
                draw_full_box = true;
                draw_dashed_box = false;
                draw_corners = false; // No corners, add glitch lines instead
                fill_mask = true;
                // Added: Glitch lines, diagonal text gradient (simulated)
                break;
            case 3: // Style 3: Plasma Flow - Refined
                box_color = tech_colors[color_index % num_tech_colors];
                mask_color = box_color;
                contour_color = cv::Scalar(255, 255, 255);
                mask_alpha = 0.45f; // Slightly more opaque mask
                text_bg_color_base = box_color;
                text_color = cv::Scalar(255, 255, 255);
                text_bg_alpha = 0.7f; // Stronger gradient text bg
                box_thickness = 5; // Thicker, gradient corners
                contour_thickness = 2;
                corner_length_ratio = 3;
                draw_corners = true;
                draw_full_box = false;
                fill_mask = true;
                // Added: Gradient corner brackets, stronger text gradient
                break;
            case 4: // Style 4: Data Stream - Refined
                box_color = tech_colors[color_index % num_tech_colors];
                mask_color = box_color;
                contour_color = cv::Scalar(255, 255, 255);
                mask_alpha = 0.25f;
                text_bg_color_base = box_color;
                text_color = cv::Scalar(255, 255, 255);
                text_bg_alpha = 0.7f;
                box_thickness = 2;
                contour_thickness = 1;
                draw_corners = false;
                draw_full_box = true;
                draw_dashed_box = false;
                fill_mask = true;
                // Added: Scanlines inside box, horizontal text gradient
                break;
            case 5: // Style 5: Target Lock - Refined
                box_color = tech_colors[color_index % num_tech_colors];
                mask_color = box_color;
                contour_color = box_color;
                mask_alpha = 0.0f;
                text_bg_color_base = cv::Scalar(30, 0, 0); // Dark Red BG base
                text_color = box_color;
                text_bg_alpha = 0.8f;
                box_thickness = 3;
                contour_thickness = 2;
                corner_length_ratio = 4;
                bracket_corners = true; // Keep brackets
                draw_corners = false;
                draw_full_box = false;
                fill_mask = false;
                // Added: Targeting '+' symbols, simulated radial text gradient
                break;
            case 6: // Style 6: Original Pulse - Refined
                box_color = tech_colors[color_index % num_tech_colors];
                mask_color = box_color;
                contour_color = box_color;
                mask_alpha = 0.0f;
                text_bg_color_base = cv::Scalar(40, 40, 40);
                text_color = box_color;
                text_bg_alpha = 0.7f;
                box_thickness = 1; // Keep thin dash
                contour_thickness = 2;
                draw_full_box = true;
                draw_dashed_box = true; // Keep dash
                draw_corners = false;
                fill_mask = false;
                // Added: Cleaner dash, subtle text bg brackets
                break;
            case 7: // Style 7: Neon Flow - Refined
                box_color = tech_colors[color_index % num_tech_colors];
                mask_color = box_color;
                contour_color = cv::Scalar(255, 255, 255);
                mask_alpha = 0.3f;
                text_bg_color_base = box_color; // Vibrant gradient base
                text_color = cv::Scalar(255, 255, 255);
                text_bg_alpha = 0.6f;
                box_thickness = 2;
                contour_thickness = 1;
                draw_full_box = true; // Keep box for glow base
                draw_dashed_box = false;
                draw_corners = false; // Use glowing points instead
                fill_mask = true;
                // Added: Glowing circles at corners, outer box glow, vibrant text gradient
                break;
            case 8: // Style 8: Digital Matrix - Refined
                box_color = tech_colors[color_index % num_tech_colors];
                mask_color = box_color;
                contour_color = cv::Scalar(200, 255, 200);
                mask_alpha = 0.2f;
                text_bg_color_base = cv::Scalar(20, 30, 20); // Dark green BG base
                text_color = cv::Scalar(100, 255, 100);
                text_bg_alpha = 0.85f; // More opaque
                box_thickness = 1;
                contour_thickness = 1;
                draw_full_box = true;
                draw_dashed_box = false;
                draw_double_box = true; // Keep double box
                draw_corners = false;
                fill_mask = true;
                // Added: Grid lines in mask, grid pattern on text bg
                break;
            case 9: // Style 9: Quantum Spectrum - Refined
                box_color = tech_colors[color_index % num_tech_colors];
                mask_color = tech_colors[(color_index + 3) % num_tech_colors];
                contour_color = tech_colors[(color_index + 6) % num_tech_colors];
                mask_alpha = 0.3f; // Slightly more visible mask
                text_bg_color_base = cv::Scalar(30, 30, 50); // Dark blue BG base
                text_color = box_color;
                text_bg_alpha = 0.75f;
                box_thickness = 3; // Thicker multi-color corners
                contour_thickness = 2; // Thicker contour
                draw_corners = true; // Keep corners
                corner_length_ratio = 3;
                draw_full_box = false;
                fill_mask = true;
                use_gradient_mask = true; // Enable gradient mask
                // Added: Gradient mask fill, diagonal text gradient
                break;
            default: // Default Style (Matches refined Style 0)
                box_color = tech_colors[color_index % num_tech_colors];
                mask_color = box_color;
                contour_color = cv::Scalar(255, 255, 255);
                mask_alpha = 0.2f;
                text_bg_color_base = cv::Scalar(50, 50, 50);
                text_color = box_color;
                text_bg_alpha = 0.7f;
                box_thickness = 2;
                contour_thickness = 1;
                corner_length_ratio = 4;
                draw_corners = true;
                draw_full_box = false;
                fill_mask = true;
                break;
        }

        // Restore custom style parameters if set via setStyleParameters
        if (style_custom_parameters_set) {
            __android_log_print(ANDROID_LOG_DEBUG, "Yolo", "Restoring custom style parameters");
            
            // Restore box color from custom settings
            box_color = original_box_color;
            mask_color = original_box_color;
            contour_color = original_contour_color;
            
            // Restore line type settings
            if (style_line_type == 0) { // Solid
                __android_log_print(ANDROID_LOG_DEBUG, "Yolo", "Applying SOLID line type");
                draw_full_box = true;
                draw_dashed_box = false;
                draw_corners = false;
                bracket_corners = false;
            } else if (style_line_type == 1) { // Dashed
                __android_log_print(ANDROID_LOG_DEBUG, "Yolo", "Applying DASHED line type");
                draw_full_box = true;
                draw_dashed_box = true;
                draw_corners = false;
                bracket_corners = false;
            } else if (style_line_type == 2) { // Corner brackets
                __android_log_print(ANDROID_LOG_DEBUG, "Yolo", "Applying CORNER BRACKETS line type");
                draw_full_box = false;
                draw_dashed_box = false;
                draw_corners = true;
                bracket_corners = style_line_type == 2;
            }
            
            // Restore thickness and alpha
            box_thickness = style_line_thickness;
            mask_alpha = style_mask_alpha;
            
            // Restore text color
            text_color = original_text_color;
            
            // Log the final settings being used for drawing
            __android_log_print(ANDROID_LOG_DEBUG, "Yolo", "FINAL DRAWING SETTINGS - Box Color: (%.0f,%.0f,%.0f), Line Type: %d, Thickness: %d, Mask Alpha: %.2f", 
                box_color[0], box_color[1], box_color[2], 
                style_line_type, box_thickness, mask_alpha);
        }

        // --- Label Text/Background Color Decision ---
        // Use a standard dark semi-transparent background for better readability across styles
        cv::Scalar fixed_text_bg_color = cv::Scalar(0, 0, 0); // Black background base
        float fixed_text_bg_alpha = 0.65f;
        
        // 如果有自定义文本颜色，则应用它
        if (style_custom_parameters_set && style_text_color != 0) {
            // 获取ARGB颜色分量
            int r = (style_text_color >> 16) & 0xFF;
            int g = (style_text_color >> 8) & 0xFF;
            int b = style_text_color & 0xFF;
            
            // OpenCV使用BGR顺序
            text_color = cv::Scalar(b, g, r);
            
            __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                           "文本颜色设置为 BGR(%d,%d,%d)", b, g, r);
        } else {
            // 默认使用白色文本，保证在黑色背景上的可读性
            text_color = cv::Scalar(255, 255, 255);
        }

        // --- Draw Segmentation Mask (if available) ---
        cv::Mat visible_mask_for_drawing; // Store the processed mask for drawing
        if (is_segmentation_model && !obj.mask.empty()) {
            cv::Mat processed_mask;
            if (obj.mask.cols <= 0 || obj.mask.rows <= 0) {
                // Skip if mask dimensions are invalid
            } else {
                // Ensure mask is CV_8U (0 or 255)
                if (obj.mask.type() == CV_32F) {
                    processed_mask = (obj.mask > this->mask_threshold) * 255;
                    processed_mask.convertTo(processed_mask, CV_8U);
                } else if (obj.mask.type() == CV_8U) {
                    processed_mask = obj.mask.clone(); // Use clone to avoid modifying original
                    // Handle 0/1 masks
                    double minVal, maxVal;
                    cv::minMaxLoc(processed_mask, &minVal, &maxVal);
                    if (maxVal == 1.0) {
                       processed_mask *= 255;
                    }
                } else {
                    processed_mask.release();
                }

                // Check for sufficient non-zero pixels
                if (!processed_mask.empty() && cv::countNonZero(processed_mask) < 10) {
                    processed_mask.release(); // Treat as empty if too few pixels
                }

                const int x = round(obj.rect.x);
                const int y = round(obj.rect.y);
                const int width = round(obj.rect.width);
                const int height = round(obj.rect.height);

                // Validate coordinates and dimensions
                if (!processed_mask.empty() && x >= 0 && y >= 0 && x + width <= rgb.cols && y + height <= rgb.rows && width > 0 && height > 0) {
                    // Resize if necessary
                    if (processed_mask.cols != width || processed_mask.rows != height) {
                        cv::Mat resized_mask;
                        cv::resize(processed_mask, resized_mask, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
                        if (!resized_mask.empty() && resized_mask.size() == cv::Size(width, height)) {
                            processed_mask = resized_mask;
                        } else {
                            processed_mask.release(); // Invalidate mask if resize fails
                        }
                    }

                    // If mask is still valid, store it for drawing
                    if (!processed_mask.empty()) {
                         visible_mask_for_drawing = processed_mask; // Use this for drawing steps

                         cv::Rect roi_rect(x, y, width, height);
                         cv::Mat roi = rgb(roi_rect);

                         // --- Apply Mask Fill ---
                         if (fill_mask && mask_alpha > 0.01) {
                            if (use_gradient_mask && style_index == 9) { // Special gradient fill for Quantum
                                // Create a gradient between mask_color and contour_color
                                cv::Mat gradient_layer(roi.size(), CV_8UC3);
                                for (int r = 0; r < roi.rows; ++r) {
                                    float ratio = (float)r / roi.rows;
                                    cv::Scalar current_color = mask_color * (1.0f - ratio) + contour_color * ratio;
                                    gradient_layer.row(r).setTo(current_color);
                                }
                                // Correct addWeighted call for gradient mask
                                cv::Mat blended_roi_gradient; // Temporary buffer
                                cv::addWeighted(gradient_layer, mask_alpha, roi, 1.0 - mask_alpha, 0.0, blended_roi_gradient);
                                blended_roi_gradient.copyTo(roi, visible_mask_for_drawing); // Apply using mask
                            } else {
                                // Standard alpha blending
                                cv::Mat colored_mask_layer(roi.size(), CV_8UC3, mask_color);
                                cv::Mat blended_roi; // Temporary buffer for blending result
                                cv::addWeighted(colored_mask_layer, mask_alpha, roi, 1.0 - mask_alpha, 0.0, blended_roi);
                                blended_roi.copyTo(roi, visible_mask_for_drawing); // Apply using mask
                            }
                            // Style 8: Add grid pattern overlay on mask
                            if (style_index == 8) {
                                draw_grid_pattern(roi, contour_color * 0.5, 6); // Draw grid on ROI
                            }
                         }

                         // --- Draw Mask Contour ---
                         if (contour_thickness > 0) {
                             // Get mask edge parameters from style
                             int mask_edge_thickness = this->style_mask_edge_thickness > 0 ? 
                                 this->style_mask_edge_thickness : contour_thickness;
                             
                             int mask_edge_type = this->style_mask_edge_type;
                             
                             // Use the mask edge color if specified, otherwise use the contour color
                             cv::Scalar mask_edge_color;
                             if (this->style_mask_edge_color != 0) {
                                 int r = (this->style_mask_edge_color >> 16) & 0xFF;
                                 int g = (this->style_mask_edge_color >> 8) & 0xFF;
                                 int b = this->style_mask_edge_color & 0xFF;
                                 mask_edge_color = cv::Scalar(b, g, r); // BGR format for OpenCV
                             } else {
                                 mask_edge_color = contour_color;
                             }
                             
                             // Draw contour with style
                             draw_styled_mask_contour(rgb, visible_mask_for_drawing, 
                                                      mask_edge_color, mask_edge_thickness, 
                                                      mask_edge_type, cv::Point(x, y));
                         }
                    }
                }
            }
        } // End mask drawing block


        // --- Draw Bounding Box ---
        const float x_f = obj.rect.x;
        const float y_f = obj.rect.y;
        const float width_f = obj.rect.width;
        const float height_f = obj.rect.height;

        if (!std::isnan(x_f) && !std::isnan(y_f) && !std::isnan(width_f) && !std::isnan(height_f) &&
            !std::isinf(x_f) && !std::isinf(y_f) && !std::isinf(width_f) && !std::isinf(height_f) &&
            width_f > 0 && height_f > 0)
        {
            const int x = round(x_f);
            const int y = round(y_f);
            const int width = round(width_f);
            const int height = round(height_f);

            cv::Rect box_rect = cv::Rect(x, y, width, height) & cv::Rect(0, 0, rgb.cols, rgb.rows);
            if (box_rect.width <= 0 || box_rect.height <= 0) {
                // Skip drawing if box is entirely outside
            } else {
                 // --- Draw Box based on Style ---
                 const int corner_length = (corner_length_ratio > 0 && (width > 5 || height > 5)) ?
                     std::max(1, std::min(width, height) / corner_length_ratio) :
                     std::max(1, std::min(3, std::min(width/2, height/2)));

                 // Style-specific outer glows / base elements
                 if (style_index == 1) draw_glow_rect(rgb, box_rect, box_color, 6, 3); // Holographic glow
                 if (style_index == 7) draw_glow_rect(rgb, box_rect, box_color, 8, 4); // Neon glow

                 // Main box/corner drawing
                 if (draw_full_box) {
                     if (draw_dashed_box) {
                         // Style 6: Refined Dashed Box
                         draw_dashed_rectangle(rgb, box_rect, box_color, box_thickness, 8); // Use helper
                     } else {
                         // Solid Box (Styles 1, 2, 4, 7, 8)
                         cv::rectangle(rgb, box_rect, box_color, box_thickness, cv::LINE_AA);
                         // Style 8: Refined Double Box
                         if (draw_double_box && box_rect.width > box_thickness * 2 + 4 && box_rect.height > box_thickness * 2 + 4) {
                             cv::Rect inner_rect(box_rect.x + box_thickness + 2, box_rect.y + box_thickness + 2,
                                                box_rect.width - 2*(box_thickness + 2), box_rect.height - 2*(box_thickness + 2));
                             if (inner_rect.width > 0 && inner_rect.height > 0) {
                                cv::rectangle(rgb, inner_rect, box_color * 0.8 + cv::Scalar(50,50,50)*0.2, std::max(1, box_thickness - 1), cv::LINE_AA);
                             }
                         }
                         // Style 4: Add Scanlines
                         if (style_index == 4) {
                             for (int scan_y = box_rect.y; scan_y < box_rect.br().y; scan_y += 4) {
                                 cv::line(rgb, cv::Point(box_rect.x, scan_y), cv::Point(box_rect.br().x, scan_y), box_color * 0.3, 1);
                             }
                         }
                     }
                 }

                 // Draw Corners (Styles 0, 3, 9)
                 if (draw_corners && corner_length > 0) {
                    cv::Scalar tl_c=box_color, tr_c=box_color, bl_c=box_color, br_c=box_color;
                    // Style 3 & 9 might use different corner colors (gradient effect)
                    if (style_index == 3 || style_index == 9) {
                        tl_c = tech_colors[(color_index + 1) % num_tech_colors];
                        tr_c = tech_colors[(color_index + 2) % num_tech_colors];
                        bl_c = tech_colors[(color_index + 3) % num_tech_colors];
                        br_c = tech_colors[(color_index + 4) % num_tech_colors];
                    }
                    cv::Point tl=box_rect.tl(), tr=cv::Point(box_rect.br().x, box_rect.y);
                    cv::Point bl=cv::Point(box_rect.x, box_rect.br().y), br=box_rect.br();
                    cv::line(rgb, tl, cv::Point(tl.x + corner_length, tl.y), tl_c, box_thickness, cv::LINE_AA);
                    cv::line(rgb, tl, cv::Point(tl.x, tl.y + corner_length), tl_c, box_thickness, cv::LINE_AA);
                    cv::line(rgb, cv::Point(tr.x - corner_length, tr.y), tr, tr_c, box_thickness, cv::LINE_AA);
                    cv::line(rgb, tr, cv::Point(tr.x, tr.y + corner_length), tr_c, box_thickness, cv::LINE_AA);
                    cv::line(rgb, cv::Point(bl.x, bl.y - corner_length), bl, bl_c, box_thickness, cv::LINE_AA);
                    cv::line(rgb, bl, cv::Point(bl.x + corner_length, bl.y), bl_c, box_thickness, cv::LINE_AA);
                    cv::line(rgb, cv::Point(br.x - corner_length, br.y), br, br_c, box_thickness, cv::LINE_AA);
                    cv::line(rgb, cv::Point(br.x, br.y - corner_length), br, br_c, box_thickness, cv::LINE_AA);

                    // Style 0: Add inner corners
                    if (style_index == 0 && box_thickness > 1 && corner_length > 3) {
                        int inner_offset = box_thickness;
                        int inner_len = corner_length - inner_offset > 0 ? corner_length - inner_offset : 1;
                        cv::Scalar inner_c = box_color * 0.6 + cv::Scalar(255,255,255)*0.4; // Lighter inner
                        cv::line(rgb, tl+cv::Point(inner_offset,inner_offset), cv::Point(tl.x + inner_len, tl.y+inner_offset), inner_c, 1);
                        cv::line(rgb, tl+cv::Point(inner_offset,inner_offset), cv::Point(tl.x+inner_offset, tl.y + inner_len), inner_c, 1);
                        // ... (add for other 3 corners similarly)
                    }
                 }

                 // Draw Bracket Corners (Style 5)
                 if (bracket_corners && corner_length > 0) {
                     draw_corner_brackets(rgb, box_rect, box_color, box_thickness, 100 * corner_length / std::min(width, height)); // Use helper
                     // Style 5: Add '+' target symbols
                      int cross_len = std::max(2, corner_length / 3);
                      cv::Point cross_tl = box_rect.tl() + cv::Point(corner_length/2, corner_length/2);
                      cv::line(rgb, cross_tl + cv::Point(-cross_len, 0), cross_tl + cv::Point(cross_len, 0), box_color, 1);
                      cv::line(rgb, cross_tl + cv::Point(0, -cross_len), cross_tl + cv::Point(0, cross_len), box_color, 1);
                      // ... (add for other 3 corners)
                 }

                 // Style-specific point decorations
                 if (style_index == 2) { // Pulse Glitch Lines
                     int glitch_len = 5;
                     cv::line(rgb, box_rect.tl(), box_rect.tl() + cv::Point(-glitch_len, -glitch_len), box_color, 1);
                     cv::line(rgb, box_rect.br(), box_rect.br() + cv::Point(glitch_len, glitch_len), box_color, 1);
                     cv::line(rgb, cv::Point(box_rect.br().x, box_rect.y), cv::Point(box_rect.br().x+glitch_len, box_rect.y-glitch_len), box_color, 1);
                     cv::line(rgb, cv::Point(box_rect.x, box_rect.br().y), cv::Point(box_rect.x-glitch_len, box_rect.br().y+glitch_len), box_color, 1);
                 }
                 if (style_index == 7) { // Neon Glowing Circles
                     int circle_radius = box_thickness + 2;
                     cv::circle(rgb, box_rect.tl(), circle_radius, box_color * 0.5 + cv::Scalar(128,128,128)*0.5, -1); // Base circle
                     cv::circle(rgb, box_rect.tl(), circle_radius-1 > 0? circle_radius-1 : 1, box_color, -1); // Inner bright circle
                     // ... (add for other 3 corners)
                 }

                 // --- Draw Class Label ---
                 char text[256];
                 
                 // For bilingual display, we need to add Chinese name here
                 // Get the English class name first
                 int display_label = obj.label;
                 
                 // 当使用OVI数据集时，将显示的类别ID减1
                 if (dataset_type == DATASET_OIV) {
                     // 确保类别ID不会变成负数
                     if (display_label > 0) {
                         display_label -= 1;
                         __android_log_print(ANDROID_LOG_DEBUG, "YoloDraw", "OVI类别ID调整: 原始=%d, 显示=%d", 
                                         obj.label, display_label);
                     }
                 }
                 
                 const char* english_name = class_names[display_label];
                 
                 // Get Chinese name if available, otherwise use empty string
                 std::string chinese_name = "";
                 if (!chinese_class_names.empty() && display_label < chinese_class_names.size()) {
                     chinese_name = chinese_class_names[display_label];
                 }
                 
                 // Create formatted string with bilingual name, probability and track ID
                 if (!chinese_name.empty()) {
                     // Format: Only English Name + Probability + Track ID (removed Chinese text)
                     snprintf(text, sizeof(text), "%s %.1f%%%s", 
                              english_name, 
                              obj.prob * 100,
                              (obj.track_id >= 0) ? cv::format(" ID:%d", obj.track_id).c_str() : "");
                 } else {
                     // Only English name
                     snprintf(text, sizeof(text), "%s %.1f%%%s", 
                              english_name, 
                              obj.prob * 100,
                              (obj.track_id >= 0) ? cv::format(" ID:%d", obj.track_id).c_str() : "");
                 }

                 // Increase font size and thickness for better visibility
                 int baseLine = 0;
                 double font_scale = 0.65;  // Increased from 0.55
                 int font_thickness = 2;    // Increased from 1 for bold text
                 
                 // Apply custom font size if set - 移到这里提前应用，以便正确计算文本大小
                 float actual_font_scale = font_scale;
                 if (style_font_size > 0) {
                     actual_font_scale = style_font_size;
                 }
                 
                 // Apply custom font type if set - 移到这里提前应用，以便正确计算文本大小
                 int font_face = cv::FONT_HERSHEY_SIMPLEX;
                 if (style_font_type >= 0 && style_font_type <= 7) {
                     font_face = style_font_type;
                 }
                 
                 // 使用实际的字体大小和类型来计算文本尺寸
                 cv::Size label_size = cv::getTextSize(text, font_face, actual_font_scale, font_thickness, &baseLine);

                 // Positioning logic
                 int label_x = std::max(0, x);
                 int label_y = y - label_size.height - baseLine - 4;
                 if (label_y < 0) { label_y = y + baseLine + 4; }
                 label_y = std::min(label_y, rgb.rows - baseLine - 1);
                 if (label_x + label_size.width > rgb.cols) { label_x = rgb.cols - label_size.width - 1; }
                 label_x = std::max(0, label_x);

                 // --- Draw Text Background with Style Refinements ---
                 // 根据实际字体大小调整文字背景的边距
                 int text_pad_x = std::max(5, int(actual_font_scale * 8));
                 int text_pad_y = std::max(3, int(actual_font_scale * 6));
                 
                 cv::Rect text_bg_rect(label_x - text_pad_x, 
                                      label_y - label_size.height - baseLine - text_pad_y,
                                      label_size.width + 2 * text_pad_x, 
                                      label_size.height + baseLine + 2 * text_pad_y);
                 text_bg_rect &= cv::Rect(0, 0, rgb.cols, rgb.rows);

                 if(text_bg_rect.width > 0 && text_bg_rect.height > 0) {
                    cv::Mat text_roi = rgb(text_bg_rect);
                    
                    // Increase contrast by using darker background
                    cv::Scalar fixed_text_bg_color = cv::Scalar(0, 0, 0); // Black background
                    float fixed_text_bg_alpha = 0.75f;  // More opaque for better contrast
                    
                    // Apply custom box alpha if set
                    if (style_box_alpha >= 0) {
                        fixed_text_bg_alpha = style_box_alpha + 0.1f; // Add a bit more opacity for text background
                    }
                    
                    // Base transparent background with higher contrast
                    cv::Mat bg_color_mat(text_roi.size(), CV_8UC3, fixed_text_bg_color);
                    cv::addWeighted(text_roi, 1.0 - fixed_text_bg_alpha, bg_color_mat, fixed_text_bg_alpha, 0, text_roi);

                    // Style specific background decorations
                    if (style_index == 1) { // Holographic - Vertical Gradient BG
                       draw_gradient_rect(text_roi, cv::Rect(0,0,text_roi.cols, text_roi.rows), box_color * 0.8, box_color * 0.4, false);
                    } else if (style_index == 2 || style_index == 9) { // Pulse/Quantum - Diagonal Gradient BG (Simulated)
                        for(int r=0; r<text_roi.rows; ++r) {
                            for(int c=0; c<text_roi.cols; ++c) {
                                float ratio = (float)(r+c) / (text_roi.rows + text_roi.cols);
                                cv::Vec3b& pixel = text_roi.at<cv::Vec3b>(r,c);
                                cv::Scalar mixed = box_color * (1.0-ratio) + cv::Scalar(50,50,50) * ratio;
                                pixel[0] = cv::saturate_cast<uchar>(pixel[0]*0.5 + mixed[0]*0.5); // Blend with existing bg
                                pixel[1] = cv::saturate_cast<uchar>(pixel[1]*0.5 + mixed[1]*0.5);
                                pixel[2] = cv::saturate_cast<uchar>(pixel[2]*0.5 + mixed[2]*0.5);
                            }
                        }
                    } else if (style_index == 3) { // Plasma - Stronger Vertical Gradient BG
                         draw_gradient_rect(text_roi, cv::Rect(0,0,text_roi.cols, text_roi.rows), box_color, dark_color, false);
                    } else if (style_index == 4 || style_index == 7) { // Data/Neon - Horizontal Gradient BG
                        draw_gradient_rect(text_roi, cv::Rect(0,0,text_roi.cols, text_roi.rows), box_color, dark_color, true);
                    } else if (style_index == 5) { // Target Lock - Radial Gradient BG (Simulated)
                        cv::Point center(text_roi.cols/2, text_roi.rows/2);
                        float max_dist = sqrt(pow(center.x, 2) + pow(center.y, 2));
                         for(int r=0; r<text_roi.rows; ++r) {
                            for(int c=0; c<text_roi.cols; ++c) {
                                float dist = sqrt(pow(c-center.x, 2) + pow(r-center.y, 2));
                                float ratio = dist / max_dist;
                                cv::Vec3b& pixel = text_roi.at<cv::Vec3b>(r,c);
                                cv::Scalar mixed = box_color * (1.0-ratio) + text_bg_color_base * ratio;
                                pixel[0] = cv::saturate_cast<uchar>(pixel[0]*0.4 + mixed[0]*0.6);
                                pixel[1] = cv::saturate_cast<uchar>(pixel[1]*0.4 + mixed[1]*0.6);
                                pixel[2] = cv::saturate_cast<uchar>(pixel[2]*0.4 + mixed[2]*0.6);
                            }
                        }
                    } else if (style_index == 6) { // Original Pulse - Text BG brackets
                        int bl = 3; // Bracket length
                        cv::line(text_roi, {0,0}, {bl,0}, text_color, 1); cv::line(text_roi, {0,0}, {0,bl}, text_color, 1);
                        cv::line(text_roi, {text_roi.cols-bl, text_roi.rows-1}, {text_roi.cols-1, text_roi.rows-1}, text_color, 1);
                        cv::line(text_roi, {text_roi.cols-1, text_roi.rows-bl}, {text_roi.cols-1, text_roi.rows-1}, text_color, 1);
                    } else if (style_index == 8) { // Digital Matrix - Text BG Grid
                         draw_grid_pattern(text_roi, text_color, 4);
                    }
                     
                    // Style 0: Add simple border to text BG
                    if (style_index == 0) {
                       cv::rectangle(rgb, text_bg_rect, box_color, 1);
                    }
                 }

                 // --- Draw Text ---
                 // Apply custom text color if set
                 cv::Scalar bright_text_color = cv::Scalar(255, 255, 255);  // Default white text
                 
                 // Apply custom text color if set
                 if (style_text_color != 0) {
                     // Extract RGB components from ARGB color
                     int r = (style_text_color >> 16) & 0xFF;
                     int g = (style_text_color >> 8) & 0xFF;
                     int b = style_text_color & 0xFF;
                     
                     // Log actual color values for debugging
                     __android_log_print(ANDROID_LOG_DEBUG, "YoloStyle", 
                         "Applied text color: (R=%d,G=%d,B=%d) -> BGR(B=%d,G=%d,R=%d)",
                         r, g, b, b, g, r);
                     
                     // OpenCV中颜色顺序是BGR，而Java中是ARGB，所以需要交换R和B
                     bright_text_color = cv::Scalar(b, g, r); // OpenCV uses BGR
                 }
                 
                 // Add text outline/shadow for better visibility across backgrounds
                 cv::putText(rgb, text, cv::Point(label_x + 1, label_y + 1), 
                             font_face, actual_font_scale, cv::Scalar(0, 0, 0), font_thickness + 1, cv::LINE_AA);
                 
                 // Main text with specified color
                 cv::putText(rgb, text, cv::Point(label_x, label_y),
                             font_face, actual_font_scale, bright_text_color, font_thickness, cv::LINE_AA);
            }
        } else {
             // Log invalid coordinates if needed
        }
    } // End object loop

    // 更新上一帧数据
    prev_objects = objects;
    if (!prev_mask.empty() && !is_mask_enabled) {
        prev_mask.release(); // 释放不需要的掩码内存
    }
    
    return 0;
}

// 新增：从指定的模型名称读取类别文件
int Yolo::loadClassesFromFile(AAssetManager *mgr, const char *modelName) {
    // 构建类别文件路径：使用模型名称加.txt后缀
    std::string classFileName = modelName;
    classFileName += ".txt";
    
    // 打开资源文件
    AAsset* asset = AAssetManager_open(mgr, classFileName.c_str(), AASSET_MODE_BUFFER);
    if (!asset) {
        __android_log_print(ANDROID_LOG_ERROR, "YoloClasses", "无法打开类别文件: %s", classFileName.c_str());
        return -1;
    }
    
    // 获取文件大小
    size_t size = AAsset_getLength(asset);
    
    // 将文件内容读入缓冲区
    char* buffer = new char[size + 1];
    AAsset_read(asset, buffer, size);
    buffer[size] = '\0'; // 确保字符串正确终止
    
    // 关闭资源
    AAsset_close(asset);
    
    // 解析类别文件内容
    std::stringstream ss(buffer);
    std::string line;
    std::vector<std::string> class_list;
    
    while (std::getline(ss, line)) {
        // 跳过空行
        if (line.empty()) continue;
        
        // 提取类别名称
        class_list.push_back(line);
    }
    
    // 释放缓冲区
    delete[] buffer;
    
    // 检查是否成功加载了类别
    if (class_list.empty()) {
        __android_log_print(ANDROID_LOG_WARN, "YoloClasses", "类别文件为空或格式不正确: %s", classFileName.c_str());
        return -2;
    }
    
    // 输出类别数量
    __android_log_print(ANDROID_LOG_INFO, "YoloClasses", "已从 %s 加载 %zu 个类别", 
                       classFileName.c_str(), class_list.size());
    
    // 默认启用所有类别 (清除现有过滤)
    clearClassFilter();
    
    return class_list.size();
}

// 新增：设置启用的类别ID
void Yolo::setEnabledClassIds(const std::set<int>& classIds) {
    if (classIds.empty()) {
        // 如果传入空集合，则禁用类别过滤
        clearClassFilter();
    } else {
        // 设置启用的类别ID
        enabled_class_ids = classIds;
        class_filtering_enabled = true;
        
        // 统计启用的类别数量
        __android_log_print(ANDROID_LOG_INFO, "YoloClasses", "已设置 %zu 个启用的类别", 
                           enabled_class_ids.size());
    }
}

// 新增：清除类别过滤
void Yolo::clearClassFilter() {
    enabled_class_ids.clear();
    class_filtering_enabled = false;
    __android_log_print(ANDROID_LOG_INFO, "YoloClasses", "已清除类别过滤，启用所有类别");
}

// 新增：获取类别过滤状态
bool Yolo::isClassFilteringEnabled() const {
    return class_filtering_enabled;
}

// 新增：设置中文类别名称
void Yolo::setChineseClassNames(const char** names, int count) {
    chinese_class_names.clear();
    chinese_class_names.reserve(count);
    
    for (int i = 0; i < count; i++) {
        if (names[i]) {
            chinese_class_names.push_back(std::string(names[i]));
        } else {
            chinese_class_names.push_back(std::string("")); // Empty string for null entries
        }
    }
    
    __android_log_print(ANDROID_LOG_INFO, "YoloClasses", "已设置 %d 个中文类别名称", count);
}

/**
 * Reset style parameters for specified style ID to default
 * This method clears any custom parameters and restores the default style
 */
bool Yolo::resetStyleParameters(int styleId) {
    // 先保存当前风格以便恢复
    int current_style = this->detection_style;
    
    // 设置为指定风格，这会加载默认参数
    this->setDetectionStyle(styleId);
    
    // 清除自定义参数标志
    this->style_custom_parameters_set = false;
    
    // 输出日志
    __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                       "重置风格参数 - ID: %d", styleId);
    __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                       "  线条: 粗细=%d, 类型=%d", this->style_line_thickness, this->style_line_type);
    __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                       "  框: 透明度=%.2f, 颜色=0x%08X", 
                       this->style_box_alpha, this->style_box_color);
    __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                       "  字体: 大小=%.2f, 类型=%d, 颜色=0x%08X", 
                       this->style_font_size, this->style_font_type, this->style_text_color);
    __android_log_print(ANDROID_LOG_INFO, "YoloStyle", 
                       "  掩码: 透明度=%.2f, 对比度=%.2f", 
                       this->style_mask_alpha, this->style_mask_contrast);
    
    // 如果重置的是当前风格，保持当前风格
    // 否则恢复之前的风格
    if (styleId != current_style) {
        this->setDetectionStyle(current_style);
    }
    
    return true;
}

// 在文件末尾添加
/**
 * 根据预定义配置加载模型，提高兼容性
 */
int Yolo::loadWithConfig(AAssetManager *mgr, const char *modeltype, const ModelConfig& config, bool use_gpu) {
    // 保存模型配置以便后续使用
    model_config = config;
    
    // 保存模型类型
    model_type = modeltype;

    // 初始化NCNN
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(1);
    yolo.opt = ncnn::Option();
#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Vulkan compute %s", use_gpu ? "enabled" : "disabled");
#endif
    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    // 应用配置
    is_segmentation_model = config.is_segmentation;
    num_mask_coeffs = config.num_mask_coeffs;
    det_output_name = config.detection_output;
    seg_output_name = config.seg_output;
    // Store the model config for later use
    model_config = config;
    target_size = config.input_size;
    
    __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "加载模型配置: name=%s, input_size=%d, classes=%d, is_seg=%d", 
                       config.name.c_str(), config.input_size, config.num_classes, config.is_segmentation ? 1 : 0);
    
    // 设置数据集类型
    if (config.name.find("oiv") != std::string::npos) {
        dataset_type = DATASET_OIV;
        class_names = oiv_class_names;
        __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "使用OIV数据集");
    } else {
        dataset_type = DATASET_COCO;
        class_names = coco_class_names;
        __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "使用COCO数据集");
    }

    // 支持多种命名格式和文件扩展名
    const char* extensions[][2] = {
        {".param", ".bin"},
        {".ncnn.param", ".ncnn.bin"},
        {"_ncnn_model.param", "_ncnn_model.bin"},
        {"/model.ncnn.param", "/model.ncnn.bin"}
    };
    
    char parampath[256];
    char modelpath[256];
    bool model_loaded = false;
    
    // 尝试所有可能的扩展名组合
    for (const auto& ext : extensions) {
        sprintf(parampath, "%s%s", modeltype, ext[0]);
        sprintf(modelpath, "%s%s", modeltype, ext[1]);
        
        __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "尝试加载模型: %s, %s", parampath, modelpath);
        
        // 尝试加载模型
        int ret1 = yolo.load_param(mgr, parampath);
        if (ret1 != 0) continue;
        
        int ret2 = yolo.load_model(mgr, modelpath);
        if (ret2 != 0) continue;
        
        model_loaded = true;
        __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "成功加载模型: %s, %s", parampath, modelpath);
        break;
    }
    
    if (!model_loaded) {
        __android_log_print(ANDROID_LOG_ERROR, "YoloLoad", "所有模型格式尝试均失败");
        return -1;
    }
    
    // 记录模型结构信息
    std::vector<ncnn::Layer*> layers = yolo.layers();
    __android_log_print(ANDROID_LOG_DEBUG, "YoloLoad", "模型加载完成: %d 层", (int)layers.size());
    
    // 记录输入输出信息
    std::vector<int> input_indexes = yolo.input_indexes();
    std::vector<int> output_indexes = yolo.output_indexes();
    
    __android_log_print(ANDROID_LOG_INFO, "YoloLoad", "模型有 %d 个输入和 %d 个输出", 
                       (int)input_indexes.size(), (int)output_indexes.size());
    
    // 设置预处理参数
    mean_vals[0] = 0.0f;
    mean_vals[1] = 0.0f;
    mean_vals[2] = 0.0f;
    norm_vals[0] = 1.0f / 255.0f;
    norm_vals[1] = 1.0f / 255.0f;
    norm_vals[2] = 1.0f / 255.0f;

    __android_log_print(ANDROID_LOG_INFO, "ncnn", "模型加载成功!");
    return 0;
}

/**
 * 根据模型名称自动检测适合的配置
 */
bool Yolo::detectModelConfig(const char *modeltype, ModelConfig& outConfig) {
    // 默认配置
    outConfig = ModelConfig();
    outConfig.name = modeltype;
    
    // 根据模型名称设置合适的配置
    std::string model_name = modeltype;
    
    // 检查模型名称
    bool is_seg = (model_name.find("-seg") != std::string::npos);
    bool is_pose = (model_name.find("-pose") != std::string::npos);
    bool is_obb = (model_name.find("-obb") != std::string::npos);
    bool is_cls = (model_name.find("-cls") != std::string::npos);
    bool is_oiv = (model_name.find("oiv") != std::string::npos);
    
    // 检查模型大小
    bool is_nano = (model_name.find("n.") != std::string::npos) || (model_name.find("yolov8n") != std::string::npos);
    bool is_small = (model_name.find("s.") != std::string::npos) || (model_name.find("yolov8s") != std::string::npos);
    bool is_medium = (model_name.find("m.") != std::string::npos) || (model_name.find("yolov8m") != std::string::npos);
    bool is_large = (model_name.find("l.") != std::string::npos) || (model_name.find("yolov8l") != std::string::npos);
    bool is_xlarge = (model_name.find("x.") != std::string::npos) || (model_name.find("yolov8x") != std::string::npos);
    
    // 设置输入尺寸
    if (is_obb) {
        outConfig.input_size = 1024; // OBB模型通常用1024尺寸
    } else {
        outConfig.input_size = 640;  // 其他模型用640尺寸
    }
    
    // 设置类别数量
    if (is_oiv) {
        outConfig.num_classes = 607; // OIV数据集
    } else if (is_cls) {
        outConfig.num_classes = 1000; // ImageNet分类
    } else if (is_pose) {
        outConfig.num_classes = 1;   // 姿态检测只有一个类别(人)
    } else {
        outConfig.num_classes = 80;  // COCO数据集
    }
    
    // 设置模型类型
    outConfig.is_segmentation = is_seg;
    
    // 设置掩码系数数量
    if (is_seg) {
        outConfig.num_mask_coeffs = 32;
    }
    
    // 根据模型版本设置适当的输入输出名称
    // 对于旧版模型
    if (model_name.find("yolov8") != std::string::npos && model_name.find("ncnn") == std::string::npos) {
        outConfig.input_name = "images";
        outConfig.detection_output = "output";
        if (is_seg) outConfig.seg_output = "seg";
    } 
    // 对于最新版的NCNN转换模型
    else if (model_name.find("ncnn") != std::string::npos) {
        outConfig.input_name = "in0";
        outConfig.detection_output = "out0";
        if (is_seg) {
            outConfig.seg_output = "out1";
            outConfig.coeff_output = "out2";
        }
    } 
    // 对于数字索引的输出模型
    else if (model_name.find("onnx") != std::string::npos) {
        outConfig.input_name = "input";
        outConfig.detection_output = "382"; // 常见ONNX转NCNN的输出索引
        if (is_seg) outConfig.seg_output = "383";
    }
    // 默认情况
    else {
        outConfig.input_name = "images";
        outConfig.detection_output = "output";
        if (is_seg) outConfig.seg_output = "seg";
    }
    
    // 对于某些模型，输出需要转置
    // 根据经验设置转置标志
    if (model_name.find("ncnn") != std::string::npos || model_name.find("yolov8") != std::string::npos) {
        outConfig.transpose_output = true;
    }
    
    __android_log_print(ANDROID_LOG_INFO, "YoloConfig", "检测模型配置: name=%s, size=%d, classes=%d, is_seg=%d, transpose=%d",
                      model_name.c_str(), outConfig.input_size, outConfig.num_classes, 
                      outConfig.is_segmentation ? 1 : 0, outConfig.transpose_output ? 1 : 0);
    
    return true;
}

/**
 * 模型验证工具：检查模型结构并验证输入输出
 * 返回值: 0成功, -1失败
 */
int Yolo::validateModel(AAssetManager* mgr, const char* model_path) {
    __android_log_print(ANDROID_LOG_INFO, "YoloValidate", "开始验证模型: %s", model_path);
    
    // 创建临时网络
    ncnn::Net net;
    
    // 配置网络选项
    net.opt.num_threads = 1;
    
    // 尝试多种可能的文件扩展名
    const char* extensions[][2] = {
        {".param", ".bin"},
        {".ncnn.param", ".ncnn.bin"},
        {"_ncnn_model.param", "_ncnn_model.bin"},
        {"/model.ncnn.param", "/model.ncnn.bin"}
    };
    
    bool model_loaded = false;
    
    for (const auto& ext : extensions) {
        char parampath[256];
        char modelpath[256];
        
        sprintf(parampath, "%s%s", model_path, ext[0]);
        sprintf(modelpath, "%s%s", model_path, ext[1]);
        
        __android_log_print(ANDROID_LOG_INFO, "YoloValidate", "尝试加载: %s, %s", parampath, modelpath);
        
        // 尝试加载参数文件
        int ret1 = net.load_param(mgr, parampath);
        if (ret1 != 0) {
            __android_log_print(ANDROID_LOG_ERROR, "YoloValidate", "加载参数文件失败: %d", ret1);
            continue;
        }
        
        // 尝试加载模型文件
        int ret2 = net.load_model(mgr, modelpath);
        if (ret2 != 0) {
            __android_log_print(ANDROID_LOG_ERROR, "YoloValidate", "加载模型文件失败: %d", ret2);
            continue;
        }
        
        model_loaded = true;
        __android_log_print(ANDROID_LOG_INFO, "YoloValidate", "成功加载模型: %s, %s", parampath, modelpath);
        break;
    }
    
    if (!model_loaded) {
        __android_log_print(ANDROID_LOG_ERROR, "YoloValidate", "所有格式尝试均失败");
        return -1;
    }
    
    // 分析模型结构
    std::vector<ncnn::Layer*> layers = net.layers();
    int layer_count = layers.size();
    __android_log_print(ANDROID_LOG_INFO, "YoloValidate", "模型包含 %d 层", layer_count);
    
    // 记录输入输出信息
    std::vector<int> input_indexes = net.input_indexes();
    std::vector<int> output_indexes = net.output_indexes();
    
    __android_log_print(ANDROID_LOG_INFO, "YoloValidate", "模型有 %d 个输入和 %d 个输出", 
                       (int)input_indexes.size(), (int)output_indexes.size());
    
    // 分析和打印输入层信息
    for (int i = 0; i < input_indexes.size(); i++) {
        int idx = input_indexes[i];
        if (idx < net.blobs().size()) {
            const ncnn::Blob& blob = net.blobs()[idx];
            __android_log_print(ANDROID_LOG_INFO, "YoloValidate", "输入 %d: name='%s', shape=[%d,%d,%d]", 
                              i, blob.name.c_str(), blob.shape.w, blob.shape.h, blob.shape.c);
        }
    }
    
    // 分析和打印输出层信息
    for (int i = 0; i < output_indexes.size(); i++) {
        int idx = output_indexes[i];
        if (idx < net.blobs().size()) {
            const ncnn::Blob& blob = net.blobs()[idx];
            __android_log_print(ANDROID_LOG_INFO, "YoloValidate", "输出 %d: name='%s', shape=[%d,%d,%d]", 
                              i, blob.name.c_str(), blob.shape.w, blob.shape.h, blob.shape.c);
        }
    }
    
    // 创建测试输入
    cv::Mat test_img(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
    ncnn::Mat in = ncnn::Mat::from_pixels(test_img.data, ncnn::Mat::PIXEL_BGR2RGB, 
                                         test_img.cols, test_img.rows);
    
    // 归一化
    const float norm_vals[3] = {1/255.0f, 1/255.0f, 1/255.0f};
    in.substract_mean_normalize(0, norm_vals);
    
    // 测试可能的输入输出名称组合
    std::vector<std::string> input_names = {"images", "in0", "input", "data", "input.1", "x", "input0"};
    std::vector<std::string> output_names = {"output", "out0", "detection", "output0", "382", "383", "384"};
    
    bool success = false;
    std::string working_input_name;
    std::string working_output_name;
    
    // 测试各种输入输出组合
    for (const auto& in_name : input_names) {
        for (const auto& out_name : output_names) {
            ncnn::Extractor ex = net.create_extractor();
            if (ex.input(in_name.c_str(), in) != 0) {
                continue;
            }
            
            ncnn::Mat out;
            if (ex.extract(out_name.c_str(), out) != 0) {
                continue;
            }
            
            // 成功找到输入输出名称组合
            __android_log_print(ANDROID_LOG_INFO, "YoloValidate", "成功组合: 输入='%s', 输出='%s', 输出形状=[%d,%d,%d]", 
                              in_name.c_str(), out_name.c_str(), out.c, out.h, out.w);
            success = true;
            
            // 记录有效的名称组合
            working_input_name = in_name;
            working_output_name = out_name;
            
            // 尝试查找分割输出，如果存在
            for (const auto& seg_name : {"seg", "out1", "386", "387"}) {
                ncnn::Mat seg_out;
                if (ex.extract(seg_name, seg_out) == 0) {
                    __android_log_print(ANDROID_LOG_INFO, "YoloValidate", "找到分割输出: name='%s', shape=[%d,%d,%d]", 
                                      seg_name, seg_out.c, seg_out.h, seg_out.w);
                }
            }
        }
    }
    
    if (success) {
        __android_log_print(ANDROID_LOG_INFO, "YoloValidate", "模型验证成功! 建议使用: input='%s', output='%s'", 
                          working_input_name.c_str(), working_output_name.c_str());
        return 0;
    } else {
        __android_log_print(ANDROID_LOG_ERROR, "YoloValidate", "未找到有效的输入输出组合");
        return -1;
    }
}