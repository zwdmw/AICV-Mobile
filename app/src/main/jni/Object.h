#pragma once
#include <opencv2/core/core.hpp>
#include <vector>

struct Object
{
    // Generally it is the coordinates of the top left corner + width and height
    cv::Rect_<float> rect;
    int label;
    float prob;
    cv::Mat mask;
    std::vector<float> mask_coeffs; // Added for segmentation coefficients
    int gindex; // Added for grid index tracking
    
    // New fields for segmentation
    float grid_x;       // Grid cell x position in feature map
    float grid_y;       // Grid cell y position in feature map
    int mask_feat_stride; // Feature stride for this detection
    
    // 添加跟踪ID字段
    int track_id = -1; // 默认为-1表示未跟踪
};