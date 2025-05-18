#ifndef OCSORT_H
#define OCSORT_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <deque>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>

// Kalman Filter for OC-SORT tracking
class KalmanBoxTracker {
private:
    // State vector: [x, y, s, r, vx, vy, vs, vr]
    // x, y: center position
    // s: scale (area)
    // r: aspect ratio
    // vx, vy, vs, vr: respective velocities
    cv::KalmanFilter kf;
    int id;
    std::deque<cv::Rect> history;
    int time_since_update;
    int hits;
    int age;
    float last_observation_confidence;
    int label;
    
    // Observation history for OC-SORT
    std::deque<cv::Mat> observations;
    int max_history_size;
    
    cv::Mat last_position;
    
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;

public:
    KalmanBoxTracker(const cv::Rect& bbox, int track_id, int obj_label, float conf) 
        : id(track_id), 
          time_since_update(0), 
          hits(1), 
          age(1), 
          last_observation_confidence(conf),
          label(obj_label),
          max_history_size(10),
          rng(std::random_device{}()),
          dist(-0.03f, 0.03f) {
        
        // Initialize Kalman filter with 8 state variables and 4 measurement variables
        kf.init(8, 4, 0, CV_32F);
        
        // Initialize state - center (x,y), area, aspect ratio, and velocities
        float cx = bbox.x + bbox.width / 2.0f;
        float cy = bbox.y + bbox.height / 2.0f;
        float s = bbox.width * bbox.height;  // area
        float r = bbox.width / (float)bbox.height;  // aspect ratio
        
        // Initialize state
        cv::Mat state(8, 1, CV_32F);
        state.at<float>(0) = cx;
        state.at<float>(1) = cy;
        state.at<float>(2) = s;
        state.at<float>(3) = r;
        state.at<float>(4) = 0;  // vx
        state.at<float>(5) = 0;  // vy
        state.at<float>(6) = 0;  // vs (change in area)
        state.at<float>(7) = 0;  // vr (change in aspect ratio)
        kf.statePost = state;
        
        // Set up the transition matrix (motion model)
        cv::Mat F = cv::Mat::eye(8, 8, CV_32F);
        F.at<float>(0, 4) = 1.0f;  // x' = x + vx
        F.at<float>(1, 5) = 1.0f;  // y' = y + vy
        F.at<float>(2, 6) = 1.0f;  // s' = s + vs
        F.at<float>(3, 7) = 1.0f;  // r' = r + vr
        kf.transitionMatrix = F;
        
        // Measurement matrix (what we can observe directly: x, y, s, r)
        cv::Mat H = cv::Mat::zeros(4, 8, CV_32F);
        H.at<float>(0, 0) = 1.0f;  // observe x
        H.at<float>(1, 1) = 1.0f;  // observe y
        H.at<float>(2, 2) = 1.0f;  // observe s
        H.at<float>(3, 3) = 1.0f;  // observe r
        kf.measurementMatrix = H;
        
        // Process noise covariance matrix
        cv::Mat Q = cv::Mat::eye(8, 8, CV_32F);
        Q *= 0.01f;
        // Increase process noise for velocities
        Q.at<float>(4, 4) = 0.09f;
        Q.at<float>(5, 5) = 0.09f;
        Q.at<float>(6, 6) = 0.09f;
        Q.at<float>(7, 7) = 0.09f;
        kf.processNoiseCov = Q;
        
        // Measurement noise covariance matrix
        cv::Mat R = cv::Mat::eye(4, 4, CV_32F);
        R *= 0.1f;
        kf.measurementNoiseCov = R;
        
        // Error covariance matrix
        cv::Mat P = cv::Mat::eye(8, 8, CV_32F);
        P *= 10.0f;
        kf.errorCovPost = P;
        
        // Store first observation
        last_position = state(cv::Rect(0, 0, 1, 4)).clone();
        
        // Add to observation history
        observations.push_back(convertBboxToObservation(bbox));
        
        // Add to position history
        history.push_back(bbox);
    }
    
    cv::Mat convertBboxToObservation(const cv::Rect& bbox) {
        cv::Mat obs(4, 1, CV_32F);
        obs.at<float>(0) = bbox.x + bbox.width / 2.0f;  // center x
        obs.at<float>(1) = bbox.y + bbox.height / 2.0f;  // center y
        obs.at<float>(2) = bbox.width * bbox.height;     // area
        obs.at<float>(3) = bbox.width / (float)bbox.height;  // aspect ratio
        return obs;
    }
    
    cv::Rect predictNext() {
        if (time_since_update > 0) {
            // If no recent updates, add small noise to prevent prediction collapse
            kf.statePost.at<float>(0) += dist(rng);
            kf.statePost.at<float>(1) += dist(rng);
        }
        
        // Predict next state
        cv::Mat prediction = kf.predict();
        
        // Extract predicted bounding box
        float cx = prediction.at<float>(0);
        float cy = prediction.at<float>(1);
        float s = prediction.at<float>(2);  // area
        float r = prediction.at<float>(3);  // aspect ratio
        
        float w = std::sqrt(s * r);
        float h = std::sqrt(s / r);
        
        return cv::Rect(cx - w/2, cy - h/2, w, h);
    }
    
    void update(const cv::Rect& bbox, float conf) {
        // Update hit count and age
        hits += 1;
        time_since_update = 0;
        last_observation_confidence = conf;
        
        // Create measurement
        cv::Mat measurement = convertBboxToObservation(bbox);
        
        // Store in history
        observations.push_back(measurement);
        if (observations.size() > max_history_size) {
            observations.pop_front();
        }
        
        history.push_back(bbox);
        if (history.size() > max_history_size) {
            history.pop_front();
        }
        
        // Perform Kalman correction
        kf.correct(measurement);
        
        // Store the corrected position
        last_position = kf.statePost(cv::Rect(0, 0, 1, 4)).clone();
    }
    
    // OC-SORT: Calculate Velocity from Observation History
    cv::Mat calculateVelocityDirection() {
        if (observations.size() < 2) {
            cv::Mat zero_vec = cv::Mat::zeros(2, 1, CV_32F);
            return zero_vec;
        }
        
        // Use the most recent observations for velocity calculation
        const cv::Mat& current = observations.back();
        const cv::Mat& prev = observations[observations.size() - 2];
        
        // Calculate displacement vector (x,y only)
        cv::Mat direction(2, 1, CV_32F);
        direction.at<float>(0) = current.at<float>(0) - prev.at<float>(0);
        direction.at<float>(1) = current.at<float>(1) - prev.at<float>(1);
        
        // Normalize to unit vector if non-zero
        float norm = cv::norm(direction);
        if (norm > 1e-5) {
            direction /= norm;
        }
        
        return direction;
    }
    
    cv::Rect getBoundingBox() {
        cv::Mat state = kf.statePost;
        float cx = state.at<float>(0);
        float cy = state.at<float>(1);
        float s = state.at<float>(2);  // area
        float r = state.at<float>(3);  // aspect ratio
        
        float w = std::sqrt(s * r);
        float h = std::sqrt(s / r);
        
        return cv::Rect(cx - w/2, cy - h/2, w, h);
    }
    
    // OC-SORT: Velocity-enhanced IoU calculation
    float velocityDirectionSimilarity(const KalmanBoxTracker& other) {
        cv::Mat v1 = calculateVelocityDirection();
        cv::Mat v2 = other.calculateVelocityDirection();
        
        // Simple cosine similarity
        float dot_product = v1.dot(v2);
        float norm1 = cv::norm(v1);
        float norm2 = cv::norm(v2);
        
        if (norm1 < 1e-5 || norm2 < 1e-5) return 0.0f;
        
        return dot_product / (norm1 * norm2);
    }
    
    // Getters
    int getID() const { return id; }
    int getTimeSinceUpdate() const { return time_since_update; }
    int getHits() const { return hits; }
    int getAge() const { return age; }
    float getConfidence() const { return last_observation_confidence; }
    int getLabel() const { return label; }
    
    // Mark as updated with no matching detection
    void incrementTimeSinceUpdate() { 
        time_since_update++; 
        age++;
    }
};

class OCSort {
private:
    float iou_threshold;
    int max_age;
    int min_hits;
    float asso_threshold;
    float velocity_weight;
    int next_id;
    std::vector<KalmanBoxTracker> trackers;
    
    // Calculate IoU between two rectangles
    float calculateIoU(const cv::Rect& r1, const cv::Rect& r2) {
        int x1 = std::max(r1.x, r2.x);
        int y1 = std::max(r1.y, r2.y);
        int x2 = std::min(r1.x + r1.width, r2.x + r2.width);
        int y2 = std::min(r1.y + r1.height, r2.y + r2.height);
        
        if (x2 < x1 || y2 < y1) return 0.0f;
        
        float intersection = (x2 - x1) * (y2 - y1);
        float area1 = r1.width * r1.height;
        float area2 = r2.width * r2.height;
        
        return intersection / (area1 + area2 - intersection);
    }
    
    // Modified OC-SORT association cost
    float associationCost(const KalmanBoxTracker& tracker, const cv::Rect& detection) {
        // Calculate basic IoU cost
        float iou = calculateIoU(tracker.getBoundingBox(), detection);
        
        // Convert cost to distance (1-IoU)
        float cost = 1.0f - iou;
        
        return cost;
    }
    
    // Linear assignment using greedy algorithm (for simplicity)
    void linearAssignment(const std::vector<std::vector<float>>& cost_matrix,
                          float cost_threshold,
                          std::vector<std::pair<int, int>>& matches,
                          std::vector<int>& unmatched_detections,
                          std::vector<int>& unmatched_trackers) {
        
        int num_trackers = cost_matrix.size();
        int num_detections = num_trackers > 0 ? cost_matrix[0].size() : 0;
        
        // Initialize all trackers and detections as unmatched
        unmatched_trackers.resize(num_trackers);
        for (int i = 0; i < num_trackers; i++) {
            unmatched_trackers[i] = i;
        }
        
        unmatched_detections.resize(num_detections);
        for (int i = 0; i < num_detections; i++) {
            unmatched_detections[i] = i;
        }
        
        // No detections or trackers, nothing to match
        if (num_trackers == 0 || num_detections == 0) {
            return;
        }
        
        // Create a copy of the cost matrix that we can modify
        std::vector<std::vector<float>> cost_matrix_copy = cost_matrix;
        
        // Greedy assignment algorithm
        while (!unmatched_trackers.empty() && !unmatched_detections.empty()) {
            // Find lowest cost assignment
            float min_cost = FLT_MAX;
            int min_tracker_idx = -1;
            int min_detection_idx = -1;
            
            for (size_t i = 0; i < unmatched_trackers.size(); i++) {
                int tracker_idx = unmatched_trackers[i];
                
                for (size_t j = 0; j < unmatched_detections.size(); j++) {
                    int detection_idx = unmatched_detections[j];
                    
                    float cost = cost_matrix_copy[tracker_idx][detection_idx];
                    
                    if (cost < min_cost) {
                        min_cost = cost;
                        min_tracker_idx = i;
                        min_detection_idx = j;
                    }
                }
            }
            
            // If the minimum cost is above threshold, stop matching
            if (min_cost > cost_threshold || min_tracker_idx < 0) {
                break;
            }
            
            // Add to matches
            matches.push_back({unmatched_trackers[min_tracker_idx], unmatched_detections[min_detection_idx]});
            
            // Remove from unmatched lists
            unmatched_trackers.erase(unmatched_trackers.begin() + min_tracker_idx);
            unmatched_detections.erase(unmatched_detections.begin() + min_detection_idx);
        }
    }

public:
    OCSort(float iou_thresh = 0.3, int max_age_frames = 30, int min_hits_start = 3,
           float assoc_thresh = 0.7, float vel_weight = 0.2)
        : iou_threshold(iou_thresh),
          max_age(max_age_frames),
          min_hits(min_hits_start),
          asso_threshold(assoc_thresh),
          velocity_weight(vel_weight),
          next_id(0) {}
    
    std::vector<Object> update(const std::vector<Object>& detections) {
        // Predict new locations of all trackers
        std::vector<cv::Rect> predicted_boxes;
        for (auto& tracker : trackers) {
            cv::Rect predicted = tracker.predictNext();
            predicted_boxes.push_back(predicted);
        }
        
        // Prepare data structures for assignment
        std::vector<std::pair<int, int>> matches;
        std::vector<int> unmatched_trackers;
        std::vector<int> unmatched_detections;
        
        // If we have detections and trackers, perform assignment
        if (!detections.empty() && !trackers.empty()) {
            // Create cost matrix for assignment
            std::vector<std::vector<float>> cost_matrix(trackers.size(), 
                                                       std::vector<float>(detections.size()));
            
            // Fill cost matrix
            for (size_t i = 0; i < trackers.size(); i++) {
                for (size_t j = 0; j < detections.size(); j++) {
                    // Skip if labels don't match
                    if (trackers[i].getLabel() != detections[j].label) {
                        cost_matrix[i][j] = 1.0f;  // Maximum cost
                        continue;
                    }
                    
                    cost_matrix[i][j] = associationCost(trackers[i], detections[j].rect);
                }
            }
            
            // Perform assignment
            linearAssignment(cost_matrix, 1.0f - iou_threshold, matches, 
                             unmatched_detections, unmatched_trackers);
        } else {
            // No matches possible, all trackers and detections are unmatched
            for (size_t i = 0; i < trackers.size(); i++) {
                unmatched_trackers.push_back(i);
            }
            
            for (size_t i = 0; i < detections.size(); i++) {
                unmatched_detections.push_back(i);
            }
        }
        
        // Update matched trackers with new detections
        for (const auto& match : matches) {
            trackers[match.first].update(detections[match.second].rect, detections[match.second].prob);
        }
        
        // Create new trackers for unmatched detections
        for (int idx : unmatched_detections) {
            KalmanBoxTracker tracker(detections[idx].rect, next_id++, detections[idx].label, detections[idx].prob);
            trackers.push_back(tracker);
        }
        
        // Increment age for unmatched trackers
        for (int idx : unmatched_trackers) {
            trackers[idx].incrementTimeSinceUpdate();
        }
        
        // Remove trackers that have been missing for too long
        for (auto it = trackers.begin(); it != trackers.end();) {
            if (it->getTimeSinceUpdate() > max_age) {
                it = trackers.erase(it);
            } else {
                ++it;
            }
        }
        
        // Generate outputs - only include trackers that have been seen enough times
        std::vector<Object> results;
        for (const auto& tracker : trackers) {
            if (tracker.getHits() >= min_hits && tracker.getTimeSinceUpdate() <= 1) {
                Object obj;
                obj.rect = tracker.getBoundingBox();
                obj.label = tracker.getLabel();
                obj.prob = tracker.getConfidence();
                obj.track_id = tracker.getID();
                
                results.push_back(obj);
            }
        }
        
        return results;
    }
};

// Global tracker instance
static OCSort g_ocsort(0.3, 30, 3, 0.7, 0.2);

#endif // OCSORT_H 