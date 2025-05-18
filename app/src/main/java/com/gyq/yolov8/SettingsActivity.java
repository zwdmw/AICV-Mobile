package com.gyq.yolov8;

import androidx.appcompat.app.AppCompatActivity;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.RadioGroup;
import android.widget.RadioButton;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;
import com.google.android.material.switchmaterial.SwitchMaterial;
import java.util.Locale;

public class SettingsActivity extends AppCompatActivity {

    private static final String TAG = "SettingsActivity";

    // Add Request Code for starting this activity
    public static final int REQUEST_CODE_SETTINGS = 1002; // Or any unique integer

    // SharedPreferences keys
    public static final String PREF_IOU_THRESHOLD = "iou_threshold";
    public static final String PREF_MAX_AGE = "max_age";
    public static final String PREF_MIN_HITS = "min_hits";
    public static final String PREF_KALMAN_PROCESS_NOISE = "kalman_process_noise";
    public static final String PREF_KALMAN_MEASUREMENT_NOISE = "kalman_measurement_noise";
    public static final String PREF_ENABLE_MASK_TRACKING = "enable_mask_tracking";
    // New keys for trajectory visualization
    public static final String PREF_ENABLE_TRAJECTORY_VIZ = "enable_trajectory_viz";
    public static final String PREF_TRAJECTORY_LENGTH = "trajectory_length";
    public static final String PREF_TRAJECTORY_THICKNESS = "trajectory_thickness";
    public static final String PREF_TRAJECTORY_COLOR_MODE = "trajectory_color_mode";
    // New keys for continuous tracking
    public static final String PREF_ENABLE_CONTINUOUS_TRACKING = "enable_continuous_tracking";
    public static final String PREF_PREDICTION_THRESHOLD = "prediction_threshold";
    public static final String PREF_MAX_PREDICTION_FRAMES = "max_prediction_frames";
    public static final String PREF_PREDICTION_LINE_TYPE = "prediction_line_type";
    // 跟踪模式常量
    public static final String PREF_TRACKING_MODE = "tracking_mode";
    public static final int TRACKING_MODE_STABLE = 0;    // 稳定模式
    public static final int TRACKING_MODE_HANDHELD = 1;  // 手持模式
    public static final int TRACKING_MODE_CUSTOM = 2;    // 自定义模式

    // 新增: 用于保存当前选中的运动子模式的键
    public static final String PREF_MOTION_SUBMODE = "motion_submode";

    // 运动子模式常量
    public static final String PREF_MOTION_SUBMODE_STABLE = "motion_submode_stable";
    public static final String PREF_MOTION_SUBMODE_HANDHELD = "motion_submode_handheld";
    public static final int MOTION_SUBMODE_STATIC = 0;
    public static final int MOTION_SUBMODE_CV = 1;      // 匀速
    public static final int MOTION_SUBMODE_CA = 2;      // 加速
    public static final int MOTION_SUBMODE_VARIABLE = 3; // 变速
    public static final int MOTION_SUBMODE_GENERAL = 4; // 通用 (默认)
    
    // 空间分布模式常量
    public static final String PREF_SPATIAL_DISTRIBUTION = "spatial_distribution";
    public static final String PREF_SPATIAL_DISTRIBUTION_STABLE = "spatial_distribution_stable"; 
    public static final String PREF_SPATIAL_DISTRIBUTION_HANDHELD = "spatial_distribution_handheld";
    public static final int SPATIAL_SPARSE = 0;      // 稀疏模式
    public static final int SPATIAL_NORMAL = 1;      // 正常模式
    public static final int SPATIAL_DENSE = 2;       // 密集模式
    public static final int SPATIAL_VERY_DENSE = 3;  // 超密集模式
    public static final int DEFAULT_SPATIAL_DISTRIBUTION = SPATIAL_NORMAL;

    // Default values
    public static final float DEFAULT_IOU_THRESHOLD = 0.3f;
    public static final int DEFAULT_MAX_AGE = 30;
    public static final int DEFAULT_MIN_HITS = 3;
    public static final float DEFAULT_KALMAN_PROCESS_NOISE = 1e-3f; // 1e-3
    public static final float DEFAULT_KALMAN_MEASUREMENT_NOISE = 1e-2f; // 1e-2
    public static final boolean DEFAULT_ENABLE_MASK_TRACKING = true;
    // New default values for trajectory visualization
    public static final boolean DEFAULT_ENABLE_TRAJECTORY_VIZ = false;
    public static final int DEFAULT_TRAJECTORY_LENGTH = 20;
    public static final float DEFAULT_TRAJECTORY_THICKNESS = 2.0f;
    public static final int DEFAULT_TRAJECTORY_COLOR_MODE = 0;
    // Default values for continuous tracking
    public static final boolean DEFAULT_ENABLE_CONTINUOUS_TRACKING = false;
    public static final float DEFAULT_PREDICTION_THRESHOLD = 0.6f;
    public static final int DEFAULT_MAX_PREDICTION_FRAMES = 30;
    public static final int DEFAULT_PREDICTION_LINE_TYPE = 1; // 虚线
    // 默认跟踪模式
    public static final int DEFAULT_TRACKING_MODE = TRACKING_MODE_STABLE;
    // 默认运动子模式
    public static final int DEFAULT_MOTION_SUBMODE = MOTION_SUBMODE_GENERAL;

    private SeekBar seekBarIouThreshold;
    private TextView textViewIouThresholdValue;
    private EditText editTextMaxAge;
    private EditText editTextMinHits;
    private SeekBar seekBarKalmanProcessNoise;
    private TextView textViewKalmanProcessNoiseValue;
    private SeekBar seekBarKalmanMeasurementNoise;
    private TextView textViewKalmanMeasurementNoiseValue;
    private SwitchMaterial switchEnableMaskTracking;
    private Button buttonSaveSettings;
    // New UI controls for trajectory visualization
    private SwitchMaterial switchEnableTrajectoryViz;
    private SeekBar seekBarTrajectoryLength;
    private TextView textViewTrajectoryLengthValue;
    private SeekBar seekBarTrajectoryThickness;
    private TextView textViewTrajectoryThicknessValue;
    private SeekBar seekBarTrajectoryColorMode;
    private TextView textViewTrajectoryColorModeValue;
    // UI controls for continuous tracking
    private SwitchMaterial switchEnableContinuousTracking;
    private SeekBar seekBarPredictionThreshold;
    private TextView textViewPredictionThresholdValue;
    private SeekBar seekBarMaxPredictionFrames;
    private TextView textViewMaxPredictionFramesValue;
    private SeekBar seekBarPredictionLineType;
    private TextView textViewPredictionLineTypeValue;

    // 跟踪模式控件
    private RadioGroup radioGroupTrackingMode;
    private RadioButton radioButtonStableMode;
    private RadioButton radioButtonHandheldMode;
    private RadioButton radioButtonCustomMode;

    // 二级运动模式控件
    private LinearLayout layoutMotionSubMode;
    private RadioGroup radioGroupMotionSubMode;
    private RadioButton radioButtonMotionStatic;
    private RadioButton radioButtonMotionCV;
    private RadioButton radioButtonMotionCA;
    private RadioButton radioButtonMotionVariable;
    private RadioButton radioButtonMotionGeneral;
    
    // 空间分布模式控件
    private LinearLayout layoutSpatialDistribution;
    private RadioGroup radioGroupSpatialDistribution;
    private RadioButton radioButtonSpatialSparse;
    private RadioButton radioButtonSpatialNormal;
    private RadioButton radioButtonSpatialDense;
    private RadioButton radioButtonSpatialVeryDense;

    private SharedPreferences sharedPreferences;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);
        setTitle("Tracking Settings");

        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);

        findViews();
        loadSettings();
        setupListeners();
    }

    private void findViews() {
        seekBarIouThreshold = findViewById(R.id.seekBarIouThreshold);
        textViewIouThresholdValue = findViewById(R.id.textViewIouThresholdValue);
        editTextMaxAge = findViewById(R.id.editTextMaxAge);
        editTextMinHits = findViewById(R.id.editTextMinHits);
        seekBarKalmanProcessNoise = findViewById(R.id.seekBarKalmanProcessNoise);
        textViewKalmanProcessNoiseValue = findViewById(R.id.textViewKalmanProcessNoiseValue);
        seekBarKalmanMeasurementNoise = findViewById(R.id.seekBarKalmanMeasurementNoise);
        textViewKalmanMeasurementNoiseValue = findViewById(R.id.textViewKalmanMeasurementNoiseValue);
        switchEnableMaskTracking = findViewById(R.id.switchEnableMaskTracking);
        // New UI elements
        switchEnableTrajectoryViz = findViewById(R.id.switchEnableTrajectoryViz);
        seekBarTrajectoryLength = findViewById(R.id.seekBarTrajectoryLength);
        textViewTrajectoryLengthValue = findViewById(R.id.textViewTrajectoryLengthValue);
        seekBarTrajectoryThickness = findViewById(R.id.seekBarTrajectoryThickness);
        textViewTrajectoryThicknessValue = findViewById(R.id.textViewTrajectoryThicknessValue);
        seekBarTrajectoryColorMode = findViewById(R.id.seekBarTrajectoryColorMode);
        textViewTrajectoryColorModeValue = findViewById(R.id.textViewTrajectoryColorModeValue);
        // UI controls for continuous tracking
        switchEnableContinuousTracking = findViewById(R.id.switchEnableContinuousTracking);
        seekBarPredictionThreshold = findViewById(R.id.seekBarPredictionThreshold);
        textViewPredictionThresholdValue = findViewById(R.id.textViewPredictionThresholdValue);
        seekBarMaxPredictionFrames = findViewById(R.id.seekBarMaxPredictionFrames);
        textViewMaxPredictionFramesValue = findViewById(R.id.textViewMaxPredictionFramesValue);
        seekBarPredictionLineType = findViewById(R.id.seekBarPredictionLineType);
        textViewPredictionLineTypeValue = findViewById(R.id.textViewPredictionLineTypeValue);
        
        buttonSaveSettings = findViewById(R.id.buttonSaveSettings);

        // 跟踪模式控件
        radioGroupTrackingMode = findViewById(R.id.radioGroupTrackingMode);
        radioButtonStableMode = findViewById(R.id.radioButtonStableMode);
        radioButtonHandheldMode = findViewById(R.id.radioButtonHandheldMode);
        radioButtonCustomMode = findViewById(R.id.radioButtonCustomMode);

        // 二级运动模式控件
        layoutMotionSubMode = findViewById(R.id.layoutMotionSubMode);
        radioGroupMotionSubMode = findViewById(R.id.radioGroupMotionSubMode);
        radioButtonMotionStatic = findViewById(R.id.radioButtonMotionStatic);
        radioButtonMotionCV = findViewById(R.id.radioButtonMotionCV);
        radioButtonMotionCA = findViewById(R.id.radioButtonMotionCA);
        radioButtonMotionVariable = findViewById(R.id.radioButtonMotionVariable);
        radioButtonMotionGeneral = findViewById(R.id.radioButtonMotionGeneral);

        // 空间分布模式控件
        layoutSpatialDistribution = findViewById(R.id.layoutSpatialDistribution);
        radioGroupSpatialDistribution = findViewById(R.id.radioGroupSpatialDistribution);
        radioButtonSpatialSparse = findViewById(R.id.radioButtonSpatialSparse);
        radioButtonSpatialNormal = findViewById(R.id.radioButtonSpatialNormal);
        radioButtonSpatialDense = findViewById(R.id.radioButtonSpatialDense);
        radioButtonSpatialVeryDense = findViewById(R.id.radioButtonSpatialVeryDense);
    }

    private void loadSettings() {
        float iouThreshold = sharedPreferences.getFloat(PREF_IOU_THRESHOLD, DEFAULT_IOU_THRESHOLD);
        int maxAge = sharedPreferences.getInt(PREF_MAX_AGE, DEFAULT_MAX_AGE);
        int minHits = sharedPreferences.getInt(PREF_MIN_HITS, DEFAULT_MIN_HITS);
        float processNoise = sharedPreferences.getFloat(PREF_KALMAN_PROCESS_NOISE, DEFAULT_KALMAN_PROCESS_NOISE);
        float measurementNoise = sharedPreferences.getFloat(PREF_KALMAN_MEASUREMENT_NOISE, DEFAULT_KALMAN_MEASUREMENT_NOISE);
        boolean enableMaskTracking = sharedPreferences.getBoolean(PREF_ENABLE_MASK_TRACKING, DEFAULT_ENABLE_MASK_TRACKING);
        // Load trajectory visualization settings
        boolean enableTrajectoryViz = sharedPreferences.getBoolean(PREF_ENABLE_TRAJECTORY_VIZ, DEFAULT_ENABLE_TRAJECTORY_VIZ);
        int trajectoryLength = sharedPreferences.getInt(PREF_TRAJECTORY_LENGTH, DEFAULT_TRAJECTORY_LENGTH);
        float trajectoryThickness = sharedPreferences.getFloat(PREF_TRAJECTORY_THICKNESS, DEFAULT_TRAJECTORY_THICKNESS);
        int trajectoryColorMode = sharedPreferences.getInt(PREF_TRAJECTORY_COLOR_MODE, DEFAULT_TRAJECTORY_COLOR_MODE);
        // 加载连续跟踪模式设置
        boolean enableContinuousTracking = sharedPreferences.getBoolean(PREF_ENABLE_CONTINUOUS_TRACKING, DEFAULT_ENABLE_CONTINUOUS_TRACKING);
        float predictionThreshold = sharedPreferences.getFloat(PREF_PREDICTION_THRESHOLD, DEFAULT_PREDICTION_THRESHOLD);
        int maxPredictionFrames = sharedPreferences.getInt(PREF_MAX_PREDICTION_FRAMES, DEFAULT_MAX_PREDICTION_FRAMES);
        int predictionLineType = sharedPreferences.getInt(PREF_PREDICTION_LINE_TYPE, DEFAULT_PREDICTION_LINE_TYPE);
        // 加载跟踪模式
        int trackingMode = sharedPreferences.getInt(PREF_TRACKING_MODE, DEFAULT_TRACKING_MODE);

        // Update UI elements
        seekBarIouThreshold.setProgress((int) (iouThreshold * 100));
        textViewIouThresholdValue.setText(String.format(Locale.US, "%.2f", iouThreshold));

        editTextMaxAge.setText(String.valueOf(maxAge));
        editTextMinHits.setText(String.valueOf(minHits));

        // Kalman noise SeekBars use a log scale: progress = (log10(value) + 6) * 100
        // Value = 10^((progress/100) - 6)
        int processNoiseProgress = (int) ((Math.log10(processNoise) + 6) * 100);
        seekBarKalmanProcessNoise.setProgress(processNoiseProgress);
        textViewKalmanProcessNoiseValue.setText(formatScientific(processNoise));

        int measurementNoiseProgress = (int) ((Math.log10(measurementNoise) + 6) * 100);
        seekBarKalmanMeasurementNoise.setProgress(measurementNoiseProgress);
        textViewKalmanMeasurementNoiseValue.setText(formatScientific(measurementNoise));

        switchEnableMaskTracking.setChecked(enableMaskTracking);
        
        // Update trajectory visualization UI elements
        switchEnableTrajectoryViz.setChecked(enableTrajectoryViz);
        seekBarTrajectoryLength.setProgress(trajectoryLength);
        textViewTrajectoryLengthValue.setText(String.valueOf(trajectoryLength));
        
        // Convert thickness to progress (0.5 to 5.0 => 1 to 100)
        int thicknessProgress = (int)((trajectoryThickness - 0.5) * 20);
        seekBarTrajectoryThickness.setProgress(thicknessProgress);
        textViewTrajectoryThicknessValue.setText(String.format(Locale.US, "%.1f", trajectoryThickness));
        
        seekBarTrajectoryColorMode.setProgress(trajectoryColorMode);
        String[] colorModeLabels = {"单色", "渐变", "彩虹"};
        textViewTrajectoryColorModeValue.setText(colorModeLabels[trajectoryColorMode]);

        // Update continuous tracking UI elements
        switchEnableContinuousTracking.setChecked(enableContinuousTracking);
        seekBarPredictionThreshold.setProgress((int) (predictionThreshold * 100));
        textViewPredictionThresholdValue.setText(String.format(Locale.US, "%.2f", predictionThreshold));
        seekBarMaxPredictionFrames.setProgress(maxPredictionFrames);
        textViewMaxPredictionFramesValue.setText(String.valueOf(maxPredictionFrames));
        seekBarPredictionLineType.setProgress(predictionLineType);
        String[] lineTypeLabels = {"虚线", "实线"};
        textViewPredictionLineTypeValue.setText(lineTypeLabels[predictionLineType]);
        
        // 加载跟踪模式
        
        // 根据一级模式显示/隐藏二级模式选项并加载参数
        if (trackingMode == TRACKING_MODE_CUSTOM) {
            layoutMotionSubMode.setVisibility(View.GONE);
            layoutSpatialDistribution.setVisibility(View.GONE);
            // 加载用户自定义参数 (实际上之前已经加载到UI控件了)
            updateUIBasedOnTrackingMode(trackingMode); // 确保控件状态正确
        } else {
            layoutMotionSubMode.setVisibility(View.VISIBLE);
            layoutSpatialDistribution.setVisibility(View.VISIBLE);
            
            // 加载运动子模式
            int motionSubMode = DEFAULT_MOTION_SUBMODE;
            // 加载空间分布模式
            int spatialDistribution = DEFAULT_SPATIAL_DISTRIBUTION;
            
            if (trackingMode == TRACKING_MODE_STABLE) {
                motionSubMode = sharedPreferences.getInt(PREF_MOTION_SUBMODE_STABLE, DEFAULT_MOTION_SUBMODE);
                spatialDistribution = sharedPreferences.getInt(PREF_SPATIAL_DISTRIBUTION_STABLE, DEFAULT_SPATIAL_DISTRIBUTION);
            } else { // Handheld mode
                motionSubMode = sharedPreferences.getInt(PREF_MOTION_SUBMODE_HANDHELD, DEFAULT_MOTION_SUBMODE);
                spatialDistribution = sharedPreferences.getInt(PREF_SPATIAL_DISTRIBUTION_HANDHELD, DEFAULT_SPATIAL_DISTRIBUTION);
            }
            
            // 设置二级模式RadioButton选中状态
            switch (motionSubMode) {
                case MOTION_SUBMODE_STATIC: radioGroupMotionSubMode.check(R.id.radioButtonMotionStatic); break;
                case MOTION_SUBMODE_CV: radioGroupMotionSubMode.check(R.id.radioButtonMotionCV); break;
                case MOTION_SUBMODE_CA: radioGroupMotionSubMode.check(R.id.radioButtonMotionCA); break;
                case MOTION_SUBMODE_VARIABLE: radioGroupMotionSubMode.check(R.id.radioButtonMotionVariable); break;
                default: radioGroupMotionSubMode.check(R.id.radioButtonMotionGeneral); break; // General
            }
            
            // 设置空间分布模式RadioButton选中状态
            switch (spatialDistribution) {
                case SPATIAL_SPARSE: radioGroupSpatialDistribution.check(R.id.radioButtonSpatialSparse); break;
                case SPATIAL_DENSE: radioGroupSpatialDistribution.check(R.id.radioButtonSpatialDense); break;
                case SPATIAL_VERY_DENSE: radioGroupSpatialDistribution.check(R.id.radioButtonSpatialVeryDense); break;
                default: radioGroupSpatialDistribution.check(R.id.radioButtonSpatialNormal); break; // Normal
            }
            
            // 加载预设参数到UI控件
            loadPresetParams(trackingMode, motionSubMode, spatialDistribution);
            updateUIBasedOnTrackingMode(trackingMode); // 确保控件状态正确
        }
        
        // Set primary radio button state based on loaded trackingMode
         switch (trackingMode) {
            case TRACKING_MODE_STABLE:
                radioButtonStableMode.setChecked(true);
                break;
            case TRACKING_MODE_HANDHELD:
                radioButtonHandheldMode.setChecked(true);
                break;
            case TRACKING_MODE_CUSTOM:
                radioButtonCustomMode.setChecked(true);
                break;
        }
    }

    private void setupListeners() {
        seekBarIouThreshold.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float value = progress / 100.0f;
                textViewIouThresholdValue.setText(String.format(Locale.US, "%.2f", value));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) { }
            @Override public void onStopTrackingTouch(SeekBar seekBar) { }
        });

        seekBarKalmanProcessNoise.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float value = (float) Math.pow(10, (progress / 100.0) - 6.0);
                textViewKalmanProcessNoiseValue.setText(formatScientific(value));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) { }
            @Override public void onStopTrackingTouch(SeekBar seekBar) { }
        });

        seekBarKalmanMeasurementNoise.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float value = (float) Math.pow(10, (progress / 100.0) - 6.0);
                textViewKalmanMeasurementNoiseValue.setText(formatScientific(value));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) { }
            @Override public void onStopTrackingTouch(SeekBar seekBar) { }
        });
        
        // Setup trajectory visualization listeners
        seekBarTrajectoryLength.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                textViewTrajectoryLengthValue.setText(String.valueOf(progress));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) { }
            @Override public void onStopTrackingTouch(SeekBar seekBar) { }
        });
        
        seekBarTrajectoryThickness.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                // Convert progress to thickness (1-100 => 0.5-5.0)
                float thickness = 0.5f + (progress / 20.0f);
                textViewTrajectoryThicknessValue.setText(String.format(Locale.US, "%.1f", thickness));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) { }
            @Override public void onStopTrackingTouch(SeekBar seekBar) { }
        });
        
        seekBarTrajectoryColorMode.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                String[] colorModeLabels = {"单色", "渐变", "彩虹"};
                textViewTrajectoryColorModeValue.setText(colorModeLabels[progress]);
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) { }
            @Override public void onStopTrackingTouch(SeekBar seekBar) { }
        });
        
        // 添加轨迹可视化开关的监听器，记录用户切换动作
        switchEnableTrajectoryViz.setOnCheckedChangeListener((buttonView, isChecked) -> {
            Log.d(TAG, "用户设置轨迹可视化: " + (isChecked ? "启用" : "禁用"));
        });
        
        // 连续跟踪参数监听器
        seekBarPredictionThreshold.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float threshold = progress / 100.0f;
                textViewPredictionThresholdValue.setText(String.format(Locale.US, "%.2f", threshold));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) { }
            @Override public void onStopTrackingTouch(SeekBar seekBar) { }
        });
        
        seekBarMaxPredictionFrames.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                textViewMaxPredictionFramesValue.setText(String.valueOf(progress));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) { }
            @Override public void onStopTrackingTouch(SeekBar seekBar) { }
        });
        
        seekBarPredictionLineType.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                String[] lineTypeLabels = {"虚线", "实线"};
                textViewPredictionLineTypeValue.setText(lineTypeLabels[progress]);
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) { }
            @Override public void onStopTrackingTouch(SeekBar seekBar) { }
        });

        // 跟踪模式切换监听
        radioGroupTrackingMode.setOnCheckedChangeListener((group, checkedId) -> {
            int trackingMode;
            if (checkedId == R.id.radioButtonStableMode) {
                trackingMode = TRACKING_MODE_STABLE;
                layoutMotionSubMode.setVisibility(View.VISIBLE);
                layoutSpatialDistribution.setVisibility(View.VISIBLE);
                // 读取当前选中的（或默认的）二级模式并加载预设
                int currentSubModeId = radioGroupMotionSubMode.getCheckedRadioButtonId();
                int spatialId = radioGroupSpatialDistribution.getCheckedRadioButtonId();
                int subMode = getSubModeFromId(currentSubModeId);
                int spatialDistribution = getSpatialDistributionFromId(spatialId);
                
                // 记录轨迹可视化的状态
                boolean beforeViz = switchEnableTrajectoryViz.isChecked();
                loadPresetParams(trackingMode, subMode, spatialDistribution);
                boolean afterViz = switchEnableTrajectoryViz.isChecked();
                
                if (beforeViz != afterViz) {
                    Log.d(TAG, "轨迹可视化设置已保留: " + beforeViz + " → " + afterViz);
                }
            } else if (checkedId == R.id.radioButtonHandheldMode) {
                trackingMode = TRACKING_MODE_HANDHELD;
                layoutMotionSubMode.setVisibility(View.VISIBLE);
                layoutSpatialDistribution.setVisibility(View.VISIBLE);
                int currentSubModeId = radioGroupMotionSubMode.getCheckedRadioButtonId();
                int spatialId = radioGroupSpatialDistribution.getCheckedRadioButtonId();
                int subMode = getSubModeFromId(currentSubModeId);
                int spatialDistribution = getSpatialDistributionFromId(spatialId);
                
                // 记录轨迹可视化的状态
                boolean beforeViz = switchEnableTrajectoryViz.isChecked();
                loadPresetParams(trackingMode, subMode, spatialDistribution);
                boolean afterViz = switchEnableTrajectoryViz.isChecked();
                
                if (beforeViz != afterViz) {
                    Log.d(TAG, "轨迹可视化设置已保留: " + beforeViz + " → " + afterViz);
                }
            } else { // Custom Mode
                trackingMode = TRACKING_MODE_CUSTOM;
                layoutMotionSubMode.setVisibility(View.GONE);
                layoutSpatialDistribution.setVisibility(View.GONE);
                // 加载用户之前保存的自定义设置
                loadCustomSettingsFromPrefs();
            }
            // 明确调用更新UI
            updateUIBasedOnTrackingMode(trackingMode);
            updateTextViewsAfterPreset();
            Log.d(TAG, "跟踪模式已切换为: " + (trackingMode == TRACKING_MODE_STABLE ? "稳定" : 
                  trackingMode == TRACKING_MODE_HANDHELD ? "手持" : "自定义") + "，UI已更新");
        });
        
        // 二级运动模式切换监听
        radioGroupMotionSubMode.setOnCheckedChangeListener((group, checkedId) -> {
             int primaryMode = TRACKING_MODE_STABLE; // 默认假设为稳定模式，实际应获取当前一级模式
             if (radioButtonHandheldMode.isChecked()) {
                 primaryMode = TRACKING_MODE_HANDHELD;
             } // 自定义模式下此监听器不应触发，因为布局隐藏
             
             int subMode = getSubModeFromId(checkedId);
             int spatialDistribution = getSpatialDistributionFromId(
                radioGroupSpatialDistribution.getCheckedRadioButtonId());
             
             // 记录轨迹可视化的状态
             boolean beforeViz = switchEnableTrajectoryViz.isChecked();
             loadPresetParams(primaryMode, subMode, spatialDistribution);
             boolean afterViz = switchEnableTrajectoryViz.isChecked();
             
             // 明确调用更新文本显示
             updateTextViewsAfterPreset();
             Log.d(TAG, "二级运动模式已切换为: " + subMode + "，卡尔曼参数UI已更新");
             
             if (beforeViz != afterViz) {
                 Log.d(TAG, "轨迹可视化设置已保留: " + beforeViz + " → " + afterViz);
             }
        });

        // 添加: 为连续跟踪开关添加监听器，以更新预测参数的可用性
        switchEnableContinuousTracking.setOnCheckedChangeListener((buttonView, isChecked) -> {
            // 如果当前不是自定义模式，则根据开关状态启用/禁用预测参数控件
            if (radioButtonCustomMode == null || !radioButtonCustomMode.isChecked()) {
                seekBarPredictionThreshold.setEnabled(isChecked);
                seekBarMaxPredictionFrames.setEnabled(isChecked);
                seekBarPredictionLineType.setEnabled(isChecked);
            }
        });

        // 空间分布模式切换监听
        radioGroupSpatialDistribution.setOnCheckedChangeListener((group, checkedId) -> {
            // 仅在非自定义模式下有效
            if (radioButtonCustomMode == null || !radioButtonCustomMode.isChecked()) {
                int primaryMode = radioButtonStableMode.isChecked() ? 
                        TRACKING_MODE_STABLE : TRACKING_MODE_HANDHELD;
                int motionSubMode = getSubModeFromId(radioGroupMotionSubMode.getCheckedRadioButtonId());
                int spatialDistribution = getSpatialDistributionFromId(checkedId);
                
                // 记录轨迹可视化的状态
                boolean beforeViz = switchEnableTrajectoryViz.isChecked();
                
                // 重新加载对应的预设参数
                loadPresetParams(primaryMode, motionSubMode, spatialDistribution);
                
                boolean afterViz = switchEnableTrajectoryViz.isChecked();
                
                // 更新文本显示
                updateTextViewsAfterPreset();
                Log.d(TAG, "空间分布模式已切换为: " + spatialDistribution + 
                      "，参数已更新");
                      
                if (beforeViz != afterViz) {
                    Log.d(TAG, "轨迹可视化设置已保留: " + beforeViz + " → " + afterViz);
                }
            }
        });

        buttonSaveSettings.setOnClickListener(v -> saveSettingsAndFinish());
    }

    // 从RadioButton ID获取二级模式常量
    private int getSubModeFromId(int checkedId) {
         if (checkedId == R.id.radioButtonMotionStatic) return MOTION_SUBMODE_STATIC;
         if (checkedId == R.id.radioButtonMotionCV) return MOTION_SUBMODE_CV;
         if (checkedId == R.id.radioButtonMotionCA) return MOTION_SUBMODE_CA;
         if (checkedId == R.id.radioButtonMotionVariable) return MOTION_SUBMODE_VARIABLE;
         return MOTION_SUBMODE_GENERAL; // Default
    }
    
    // 从RadioButton ID获取空间分布模式常量
    private int getSpatialDistributionFromId(int checkedId) {
        if (checkedId == R.id.radioButtonSpatialSparse) return SPATIAL_SPARSE;
        if (checkedId == R.id.radioButtonSpatialDense) return SPATIAL_DENSE;
        if (checkedId == R.id.radioButtonSpatialVeryDense) return SPATIAL_VERY_DENSE;
        return SPATIAL_NORMAL; // 默认为正常模式
    }

    // 辅助方法：加载用户保存的自定义参数 (如果选择自定义模式时需要强制重载)
    private void loadCustomSettingsFromPrefs() {
        // 复用 loadSettings 中的参数加载逻辑，但不调用 loadPresetParams
         float iouThreshold = sharedPreferences.getFloat(PREF_IOU_THRESHOLD, DEFAULT_IOU_THRESHOLD);
         int maxAge = sharedPreferences.getInt(PREF_MAX_AGE, DEFAULT_MAX_AGE);
         int minHits = sharedPreferences.getInt(PREF_MIN_HITS, DEFAULT_MIN_HITS);
         float processNoise = sharedPreferences.getFloat(PREF_KALMAN_PROCESS_NOISE, DEFAULT_KALMAN_PROCESS_NOISE);
         float measurementNoise = sharedPreferences.getFloat(PREF_KALMAN_MEASUREMENT_NOISE, DEFAULT_KALMAN_MEASUREMENT_NOISE);
         boolean enableMaskTracking = sharedPreferences.getBoolean(PREF_ENABLE_MASK_TRACKING, DEFAULT_ENABLE_MASK_TRACKING);
         boolean enableTrajectoryViz = sharedPreferences.getBoolean(PREF_ENABLE_TRAJECTORY_VIZ, DEFAULT_ENABLE_TRAJECTORY_VIZ);
         int trajectoryLength = sharedPreferences.getInt(PREF_TRAJECTORY_LENGTH, DEFAULT_TRAJECTORY_LENGTH);
         float trajectoryThickness = sharedPreferences.getFloat(PREF_TRAJECTORY_THICKNESS, DEFAULT_TRAJECTORY_THICKNESS);
         int trajectoryColorMode = sharedPreferences.getInt(PREF_TRAJECTORY_COLOR_MODE, DEFAULT_TRAJECTORY_COLOR_MODE);
         boolean enableContinuousTracking = sharedPreferences.getBoolean(PREF_ENABLE_CONTINUOUS_TRACKING, DEFAULT_ENABLE_CONTINUOUS_TRACKING);
         float predictionThreshold = sharedPreferences.getFloat(PREF_PREDICTION_THRESHOLD, DEFAULT_PREDICTION_THRESHOLD);
         int maxPredictionFrames = sharedPreferences.getInt(PREF_MAX_PREDICTION_FRAMES, DEFAULT_MAX_PREDICTION_FRAMES);
         int predictionLineType = sharedPreferences.getInt(PREF_PREDICTION_LINE_TYPE, DEFAULT_PREDICTION_LINE_TYPE);

         // Update UI elements
         seekBarIouThreshold.setProgress((int) (iouThreshold * 100));
         editTextMaxAge.setText(String.valueOf(maxAge));
         editTextMinHits.setText(String.valueOf(minHits));
         int processNoiseProgress = (int) ((Math.log10(processNoise) + 6) * 100);
         seekBarKalmanProcessNoise.setProgress(processNoiseProgress);
         int measurementNoiseProgress = (int) ((Math.log10(measurementNoise) + 6) * 100);
         seekBarKalmanMeasurementNoise.setProgress(measurementNoiseProgress);
         switchEnableMaskTracking.setChecked(enableMaskTracking);
         switchEnableTrajectoryViz.setChecked(enableTrajectoryViz);
         seekBarTrajectoryLength.setProgress(trajectoryLength);
         int thicknessProgress = (int)((trajectoryThickness - 0.5) * 20);
         seekBarTrajectoryThickness.setProgress(thicknessProgress);
         seekBarTrajectoryColorMode.setProgress(trajectoryColorMode);
         switchEnableContinuousTracking.setChecked(enableContinuousTracking);
         seekBarPredictionThreshold.setProgress((int) (predictionThreshold * 100));
         seekBarMaxPredictionFrames.setProgress(maxPredictionFrames);
         seekBarPredictionLineType.setProgress(predictionLineType);
         
         updateTextViewsAfterPreset(); // 更新所有关联的TextView
    }

    // 根据当前模式更新UI显示
    private void updateUIBasedOnTrackingMode(int mode) {
        // 在自定义模式下启用所有参数控件，其他模式下禁用
        boolean enableCustomControls = (mode == TRACKING_MODE_CUSTOM);
        
        // 启用/禁用各参数控件
        seekBarIouThreshold.setEnabled(enableCustomControls);
        editTextMaxAge.setEnabled(enableCustomControls);
        editTextMinHits.setEnabled(enableCustomControls);
        seekBarKalmanProcessNoise.setEnabled(enableCustomControls);
        seekBarKalmanMeasurementNoise.setEnabled(enableCustomControls);
        switchEnableMaskTracking.setEnabled(enableCustomControls);
        switchEnableTrajectoryViz.setEnabled(enableCustomControls);
        seekBarTrajectoryLength.setEnabled(enableCustomControls);
        seekBarTrajectoryThickness.setEnabled(enableCustomControls);
        seekBarTrajectoryColorMode.setEnabled(enableCustomControls);
        
        // 修改: 始终允许连续跟踪开关可用
        switchEnableContinuousTracking.setEnabled(true);
        
        // 修改: 连续跟踪相关参数在开启连续跟踪时可用，否则禁用
        boolean enablePredictionControls = switchEnableContinuousTracking.isChecked();
        seekBarPredictionThreshold.setEnabled(enableCustomControls || enablePredictionControls);
        seekBarMaxPredictionFrames.setEnabled(enableCustomControls || enablePredictionControls);
        seekBarPredictionLineType.setEnabled(enableCustomControls || enablePredictionControls);
    }

    private void saveSettingsAndFinish() {
        try {
            // 保存一级跟踪模式
            int trackingMode;
            if (radioButtonStableMode.isChecked()) {
                trackingMode = TRACKING_MODE_STABLE;
            } else if (radioButtonHandheldMode.isChecked()) {
                trackingMode = TRACKING_MODE_HANDHELD;
            } else {
                trackingMode = TRACKING_MODE_CUSTOM;
            }

            SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putInt(PREF_TRACKING_MODE, trackingMode);

            // 如果是一级稳定或手持模式，保存对应的二级运动模式和空间分布模式
            if (trackingMode == TRACKING_MODE_STABLE) {
                int subModeId = radioGroupMotionSubMode.getCheckedRadioButtonId();
                int spatialId = radioGroupSpatialDistribution.getCheckedRadioButtonId();
                editor.putInt(PREF_MOTION_SUBMODE_STABLE, getSubModeFromId(subModeId));
                editor.putInt(PREF_SPATIAL_DISTRIBUTION_STABLE, getSpatialDistributionFromId(spatialId));
            } else if (trackingMode == TRACKING_MODE_HANDHELD) {
                int subModeId = radioGroupMotionSubMode.getCheckedRadioButtonId();
                int spatialId = radioGroupSpatialDistribution.getCheckedRadioButtonId();
                editor.putInt(PREF_MOTION_SUBMODE_HANDHELD, getSubModeFromId(subModeId));
                editor.putInt(PREF_SPATIAL_DISTRIBUTION_HANDHELD, getSpatialDistributionFromId(spatialId));
            }
            
            // 当前的空间分布模式也保存一份，用于传递给JNI
            editor.putInt(PREF_SPATIAL_DISTRIBUTION, getSpatialDistributionFromId(
                radioGroupSpatialDistribution.getCheckedRadioButtonId()));
            
            // --- 总是读取并保存所有当前UI控件的值 --- 
            float iouThreshold = seekBarIouThreshold.getProgress() / 100.0f;
            int maxAge = Integer.parseInt(editTextMaxAge.getText().toString());
            int minHits = Integer.parseInt(editTextMinHits.getText().toString());
            float processNoise = (float) Math.pow(10, (seekBarKalmanProcessNoise.getProgress() / 100.0) - 6.0);
            float measurementNoise = (float) Math.pow(10, (seekBarKalmanMeasurementNoise.getProgress() / 100.0) - 6.0);
            boolean enableMaskTracking = switchEnableMaskTracking.isChecked();
            boolean enableTrajectoryViz = switchEnableTrajectoryViz.isChecked();
            int trajectoryLength = seekBarTrajectoryLength.getProgress();
            float trajectoryThickness = 0.5f + (seekBarTrajectoryThickness.getProgress() / 20.0f);
            int trajectoryColorMode = seekBarTrajectoryColorMode.getProgress();
            boolean enableContinuousTracking = switchEnableContinuousTracking.isChecked();
            float predictionThreshold = seekBarPredictionThreshold.getProgress() / 100.0f;
            int maxPredictionFrames = seekBarMaxPredictionFrames.getProgress();
            int predictionLineType = seekBarPredictionLineType.getProgress();

            if (maxAge <= 0 || minHits <= 0) {
                Toast.makeText(this, "Max Age and Min Hits must be positive integers.", Toast.LENGTH_SHORT).show();
                return;
            }

            // 保存所有参数值
            editor.putFloat(PREF_IOU_THRESHOLD, iouThreshold);
            editor.putInt(PREF_MAX_AGE, maxAge);
            editor.putInt(PREF_MIN_HITS, minHits);
            editor.putFloat(PREF_KALMAN_PROCESS_NOISE, processNoise);
            editor.putFloat(PREF_KALMAN_MEASUREMENT_NOISE, measurementNoise);
            editor.putBoolean(PREF_ENABLE_MASK_TRACKING, enableMaskTracking);
            editor.putBoolean(PREF_ENABLE_TRAJECTORY_VIZ, enableTrajectoryViz);
            editor.putInt(PREF_TRAJECTORY_LENGTH, trajectoryLength);
            editor.putFloat(PREF_TRAJECTORY_THICKNESS, trajectoryThickness);
            editor.putInt(PREF_TRAJECTORY_COLOR_MODE, trajectoryColorMode);
            editor.putBoolean(PREF_ENABLE_CONTINUOUS_TRACKING, enableContinuousTracking);
            editor.putFloat(PREF_PREDICTION_THRESHOLD, predictionThreshold);
            editor.putInt(PREF_MAX_PREDICTION_FRAMES, maxPredictionFrames);
            editor.putInt(PREF_PREDICTION_LINE_TYPE, predictionLineType);
            // --- 参数保存结束 ---
            
            editor.apply();

            Log.i(TAG, "Settings saved: Mode=" + trackingMode + 
                  ", IoU=" + iouThreshold + 
                  ", MaxAge=" + maxAge + 
                  ", ContinuousTracking=" + enableContinuousTracking + 
                  ", MotionCompensation=" + enableContinuousTracking);

            Toast.makeText(this, "Settings saved and applied on return.", Toast.LENGTH_SHORT).show();
            setResult(RESULT_OK); // Indicate settings were changed
            finish(); // Close the activity

        } catch (NumberFormatException e) {
            Toast.makeText(this, "Invalid number format for Max Age or Min Hits.", Toast.LENGTH_SHORT).show();
            Log.e(TAG, "Error parsing settings", e);
        } catch (Exception e) {
            Toast.makeText(this, "Error saving settings.", Toast.LENGTH_SHORT).show();
            Log.e(TAG, "Error saving settings", e);
        }
    }

    private String formatScientific(float value) {
        return String.format(Locale.US, "%.1e", value);
    }

    // 加载预设参数到UI控件
    private void loadPresetParams(int primaryMode, int subMode, int spatialDistribution) {
        // 保存用户当前的轨迹可视化设置
        boolean userTrajectoryVizSetting = switchEnableTrajectoryViz.isChecked();

        // 根据模式组合设置预设值
        // 注意：这些值需要根据实际测试进行精调
        if (primaryMode == TRACKING_MODE_STABLE) {
            switch (subMode) {
                case MOTION_SUBMODE_STATIC:
                    // 静态目标基本参数
                    seekBarIouThreshold.setProgress(35);
                    editTextMaxAge.setText("50");
                    editTextMinHits.setText("5");
                    seekBarKalmanProcessNoise.setProgress(200); // 1e-4
                    seekBarKalmanMeasurementNoise.setProgress(400); // 1e-2
                    switchEnableMaskTracking.setChecked(true);
                    switchEnableContinuousTracking.setChecked(false); // 静态目标通常不需要连续跟踪
                    
                    // 根据空间分布模式调整参数
                    switch (spatialDistribution) {
                        case SPATIAL_SPARSE:
                            // 稀疏场景下可以更宽松
                            seekBarIouThreshold.setProgress(30);
                            editTextMaxAge.setText("60");
                            editTextMinHits.setText("3");
                            seekBarKalmanProcessNoise.setProgress(100); // 1e-5
                            break;
                        case SPATIAL_DENSE:
                            // 密集场景需要更严格
                            seekBarIouThreshold.setProgress(40);
                            editTextMaxAge.setText("40");
                            editTextMinHits.setText("6");
                            break;
                        case SPATIAL_VERY_DENSE:
                            // 超密集场景极其严格
                            seekBarIouThreshold.setProgress(50);
                            editTextMaxAge.setText("30");
                            editTextMinHits.setText("8");
                            // 掩码跟踪必须启用
                            switchEnableMaskTracking.setChecked(true);
                            break;
                        // SPATIAL_NORMAL - 保持默认值
                    }
                    break;
                case MOTION_SUBMODE_CV: // 匀速
                    // 匀速目标基本参数
                    seekBarIouThreshold.setProgress(30);
                    editTextMaxAge.setText("30");
                    editTextMinHits.setText("3");
                    seekBarKalmanProcessNoise.setProgress(300); // 1e-3
                    seekBarKalmanMeasurementNoise.setProgress(400); // 1e-2
                    switchEnableMaskTracking.setChecked(true);
                    switchEnableContinuousTracking.setChecked(true);
                    seekBarPredictionThreshold.setProgress(60);
                    seekBarMaxPredictionFrames.setProgress(30);
                    
                    // 根据空间分布模式调整参数
                    switch (spatialDistribution) {
                        case SPATIAL_SPARSE:
                            seekBarIouThreshold.setProgress(25);
                            editTextMaxAge.setText("40");
                            editTextMinHits.setText("2");
                            seekBarKalmanProcessNoise.setProgress(250); // 5e-4
                            seekBarKalmanMeasurementNoise.setProgress(350); // 5e-3
                            seekBarPredictionThreshold.setProgress(50);
                            seekBarMaxPredictionFrames.setProgress(40);
                            break;
                        case SPATIAL_DENSE:
                            seekBarIouThreshold.setProgress(40);
                            editTextMaxAge.setText("25");
                            editTextMinHits.setText("4");
                            seekBarPredictionThreshold.setProgress(70);
                            seekBarMaxPredictionFrames.setProgress(20);
                            break;
                        case SPATIAL_VERY_DENSE:
                            seekBarIouThreshold.setProgress(45);
                            editTextMaxAge.setText("20");
                            editTextMinHits.setText("5");
                            seekBarKalmanProcessNoise.setProgress(320); // 2e-3
                            seekBarKalmanMeasurementNoise.setProgress(420); // 2e-2
                            switchEnableMaskTracking.setChecked(true);
                            seekBarPredictionThreshold.setProgress(75);
                            seekBarMaxPredictionFrames.setProgress(15);
                            break;
                        // SPATIAL_NORMAL - 保持默认值
                    }
                    break;
                case MOTION_SUBMODE_CA: // 加速
                    seekBarIouThreshold.setProgress(25);
                    editTextMaxAge.setText("20");
                    editTextMinHits.setText("2");
                    seekBarKalmanProcessNoise.setProgress(450); // ~3e-2
                    seekBarKalmanMeasurementNoise.setProgress(450); // ~3e-2
                    switchEnableMaskTracking.setChecked(true);
                    switchEnableContinuousTracking.setChecked(true);
                    seekBarPredictionThreshold.setProgress(70);
                    seekBarMaxPredictionFrames.setProgress(15);
                    break;
                case MOTION_SUBMODE_VARIABLE: // 变速
                    seekBarIouThreshold.setProgress(20);
                    editTextMaxAge.setText("15");
                    editTextMinHits.setText("2");
                    seekBarKalmanProcessNoise.setProgress(500); // 1e-1
                    seekBarKalmanMeasurementNoise.setProgress(500); // 1e-1
                    switchEnableMaskTracking.setChecked(false);
                    switchEnableContinuousTracking.setChecked(true);
                    seekBarPredictionThreshold.setProgress(75);
                    seekBarMaxPredictionFrames.setProgress(10);
                    break;
                case MOTION_SUBMODE_GENERAL: // 通用 (等同于之前的 Stable 默认值)
                default:
                    // Default parameters for Stable - General
                    seekBarIouThreshold.setProgress((int) (DEFAULT_IOU_THRESHOLD * 100));
                    editTextMaxAge.setText(String.valueOf(DEFAULT_MAX_AGE));
                    editTextMinHits.setText(String.valueOf(DEFAULT_MIN_HITS));
                    seekBarKalmanProcessNoise.setProgress((int) ((Math.log10(DEFAULT_KALMAN_PROCESS_NOISE) + 6) * 100));
                    seekBarKalmanMeasurementNoise.setProgress((int) ((Math.log10(DEFAULT_KALMAN_MEASUREMENT_NOISE) + 6) * 100));
                    switchEnableMaskTracking.setChecked(DEFAULT_ENABLE_MASK_TRACKING);
                    switchEnableContinuousTracking.setChecked(DEFAULT_ENABLE_CONTINUOUS_TRACKING);
                    seekBarPredictionThreshold.setProgress((int) (DEFAULT_PREDICTION_THRESHOLD * 100));
                    seekBarMaxPredictionFrames.setProgress(DEFAULT_MAX_PREDICTION_FRAMES);
                    break;
            }
        } else { // TRACKING_MODE_HANDHELD
             switch (subMode) {
                case MOTION_SUBMODE_STATIC:
                    seekBarIouThreshold.setProgress(45); // 手持静态也可能抖动
                    editTextMaxAge.setText("25");
                    editTextMinHits.setText("3");
                    seekBarKalmanProcessNoise.setProgress(400); // 1e-2
                    seekBarKalmanMeasurementNoise.setProgress(500); // 1e-1
                    switchEnableMaskTracking.setChecked(false);
                    switchEnableContinuousTracking.setChecked(false);
                    
                    // 根据空间分布模式调整参数
                    switch (spatialDistribution) {
                        case SPATIAL_SPARSE:
                            // 稀疏场景下可以更宽松
                            seekBarIouThreshold.setProgress(40);
                            editTextMaxAge.setText("30");
                            seekBarKalmanProcessNoise.setProgress(350); // 更低的过程噪声
                            seekBarKalmanMeasurementNoise.setProgress(450); // 更低的测量噪声
                            break;
                        case SPATIAL_DENSE:
                            // 密集场景需要更严格
                            seekBarIouThreshold.setProgress(50);
                            editTextMaxAge.setText("20");
                            editTextMinHits.setText("4");
                            seekBarKalmanProcessNoise.setProgress(450); // 更高的过程噪声
                            seekBarKalmanMeasurementNoise.setProgress(550); // 更高的测量噪声
                            break;
                        case SPATIAL_VERY_DENSE:
                            // 超密集场景极其严格
                            seekBarIouThreshold.setProgress(55);
                            editTextMaxAge.setText("15");
                            editTextMinHits.setText("5");
                            seekBarKalmanProcessNoise.setProgress(500); // 1e-1
                            seekBarKalmanMeasurementNoise.setProgress(600); // 1e0
                            break;
                        // SPATIAL_NORMAL - 保持默认值
                    }
                    break;
                case MOTION_SUBMODE_CV: // 匀速
                    seekBarIouThreshold.setProgress(40);
                    editTextMaxAge.setText("20");
                    editTextMinHits.setText("3");
                    seekBarKalmanProcessNoise.setProgress(450); // ~3e-2
                    seekBarKalmanMeasurementNoise.setProgress(500); // 1e-1
                    switchEnableMaskTracking.setChecked(false);
                    switchEnableContinuousTracking.setChecked(true);
                    seekBarPredictionThreshold.setProgress(65);
                    seekBarMaxPredictionFrames.setProgress(20);
                    
                    // 根据空间分布模式调整参数
                    switch (spatialDistribution) {
                        case SPATIAL_SPARSE:
                            seekBarIouThreshold.setProgress(35);
                            editTextMaxAge.setText("25");
                            editTextMinHits.setText("2");
                            seekBarKalmanProcessNoise.setProgress(400); // 更低的过程噪声
                            seekBarKalmanMeasurementNoise.setProgress(450); // 更低的测量噪声
                            seekBarPredictionThreshold.setProgress(60);
                            seekBarMaxPredictionFrames.setProgress(25);
                            break;
                        case SPATIAL_DENSE:
                            seekBarIouThreshold.setProgress(45);
                            editTextMaxAge.setText("15");
                            editTextMinHits.setText("4");
                            seekBarKalmanProcessNoise.setProgress(500); // 更高的过程噪声
                            seekBarKalmanMeasurementNoise.setProgress(550); // 更高的测量噪声
                            seekBarPredictionThreshold.setProgress(70);
                            seekBarMaxPredictionFrames.setProgress(15);
                            break;
                        case SPATIAL_VERY_DENSE:
                            seekBarIouThreshold.setProgress(50);
                            editTextMaxAge.setText("12");
                            editTextMinHits.setText("5");
                            seekBarKalmanProcessNoise.setProgress(550); // 3e-1
                            seekBarKalmanMeasurementNoise.setProgress(600); // 1e0
                            seekBarPredictionThreshold.setProgress(75);
                            seekBarMaxPredictionFrames.setProgress(12);
                            break;
                        // SPATIAL_NORMAL - 保持默认值
                    }
                    break;
                case MOTION_SUBMODE_CA: // 加速
                    seekBarIouThreshold.setProgress(35);
                    editTextMaxAge.setText("15");
                    editTextMinHits.setText("2");
                    seekBarKalmanProcessNoise.setProgress(550); // ~3e-1
                    seekBarKalmanMeasurementNoise.setProgress(550); // ~3e-1
                    switchEnableMaskTracking.setChecked(false);
                    switchEnableContinuousTracking.setChecked(true);
                    seekBarPredictionThreshold.setProgress(75);
                    seekBarMaxPredictionFrames.setProgress(12);
                    
                    // 根据空间分布模式调整参数
                    switch (spatialDistribution) {
                        case SPATIAL_SPARSE:
                            seekBarIouThreshold.setProgress(30);
                            editTextMaxAge.setText("20");
                            seekBarKalmanProcessNoise.setProgress(500); // 更低的过程噪声
                            seekBarKalmanMeasurementNoise.setProgress(500); // 更低的测量噪声
                            seekBarPredictionThreshold.setProgress(70);
                            seekBarMaxPredictionFrames.setProgress(15);
                            break;
                        case SPATIAL_DENSE:
                            seekBarIouThreshold.setProgress(40);
                            editTextMaxAge.setText("12");
                            editTextMinHits.setText("3");
                            seekBarKalmanProcessNoise.setProgress(600); // 更高的过程噪声
                            seekBarKalmanMeasurementNoise.setProgress(600); // 更高的测量噪声
                            seekBarPredictionThreshold.setProgress(80);
                            seekBarMaxPredictionFrames.setProgress(10);
                            break;
                        case SPATIAL_VERY_DENSE:
                            seekBarIouThreshold.setProgress(45);
                            editTextMaxAge.setText("10");
                            editTextMinHits.setText("4");
                            seekBarKalmanProcessNoise.setProgress(650); // 3e0
                            seekBarKalmanMeasurementNoise.setProgress(650); // 3e0
                            seekBarPredictionThreshold.setProgress(85);
                            seekBarMaxPredictionFrames.setProgress(8);
                            break;
                        // SPATIAL_NORMAL - 保持默认值
                    }
                    break;
                case MOTION_SUBMODE_VARIABLE: // 变速
                    seekBarIouThreshold.setProgress(30);
                    editTextMaxAge.setText("10");
                    editTextMinHits.setText("2");
                    seekBarKalmanProcessNoise.setProgress(600); // 1e0
                    seekBarKalmanMeasurementNoise.setProgress(600); // 1e0
                    switchEnableMaskTracking.setChecked(false);
                    switchEnableContinuousTracking.setChecked(true);
                    seekBarPredictionThreshold.setProgress(80);
                    seekBarMaxPredictionFrames.setProgress(8);
                    
                    // 根据空间分布模式调整参数
                    switch (spatialDistribution) {
                        case SPATIAL_SPARSE:
                            seekBarIouThreshold.setProgress(25);
                            editTextMaxAge.setText("15");
                            seekBarKalmanProcessNoise.setProgress(550); // 更低的过程噪声
                            seekBarKalmanMeasurementNoise.setProgress(550); // 更低的测量噪声
                            seekBarPredictionThreshold.setProgress(75);
                            seekBarMaxPredictionFrames.setProgress(10);
                            break;
                        case SPATIAL_DENSE:
                            seekBarIouThreshold.setProgress(35);
                            editTextMaxAge.setText("8");
                            editTextMinHits.setText("3");
                            seekBarKalmanProcessNoise.setProgress(650); // 更高的过程噪声
                            seekBarKalmanMeasurementNoise.setProgress(650); // 更高的测量噪声
                            seekBarPredictionThreshold.setProgress(85);
                            seekBarMaxPredictionFrames.setProgress(6);
                            break;
                        case SPATIAL_VERY_DENSE:
                            seekBarIouThreshold.setProgress(40);
                            editTextMaxAge.setText("6");
                            editTextMinHits.setText("4");
                            seekBarKalmanProcessNoise.setProgress(700); // 1e1
                            seekBarKalmanMeasurementNoise.setProgress(700); // 1e1
                            seekBarPredictionThreshold.setProgress(90);
                            seekBarMaxPredictionFrames.setProgress(5);
                            break;
                        // SPATIAL_NORMAL - 保持默认值
                    }
                    break;
                case MOTION_SUBMODE_GENERAL: // 通用 (等同于之前的手持模式默认值)
                default:
                    // Default parameters for Handheld - General (Slightly less strict than Stable-General)
                    seekBarIouThreshold.setProgress(40); // Higher IoU for handheld
                    editTextMaxAge.setText("25");
                    editTextMinHits.setText("3");
                    seekBarKalmanProcessNoise.setProgress(400); // 1e-2 (Higher process noise)
                    seekBarKalmanMeasurementNoise.setProgress(500); // 1e-1 (Higher measurement noise)
                    switchEnableMaskTracking.setChecked(false); // Often disabled for handheld
                    switchEnableContinuousTracking.setChecked(true);
                    seekBarPredictionThreshold.setProgress(70); 
                    seekBarMaxPredictionFrames.setProgress(20);
                    
                    // 根据空间分布模式调整参数
                    switch (spatialDistribution) {
                        case SPATIAL_SPARSE:
                            seekBarIouThreshold.setProgress(35);
                            editTextMaxAge.setText("30");
                            editTextMinHits.setText("2");
                            seekBarKalmanProcessNoise.setProgress(350); // 更低的过程噪声
                            seekBarKalmanMeasurementNoise.setProgress(450); // 更低的测量噪声
                            seekBarPredictionThreshold.setProgress(65);
                            seekBarMaxPredictionFrames.setProgress(25);
                            break;
                        case SPATIAL_DENSE:
                            seekBarIouThreshold.setProgress(45);
                            editTextMaxAge.setText("20");
                            editTextMinHits.setText("4");
                            seekBarKalmanProcessNoise.setProgress(450); // 更高的过程噪声
                            seekBarKalmanMeasurementNoise.setProgress(550); // 更高的测量噪声
                            seekBarPredictionThreshold.setProgress(75);
                            seekBarMaxPredictionFrames.setProgress(15);
                            break;
                        case SPATIAL_VERY_DENSE:
                            seekBarIouThreshold.setProgress(50);
                            editTextMaxAge.setText("15");
                            editTextMinHits.setText("5");
                            seekBarKalmanProcessNoise.setProgress(500); // 1e-1
                            seekBarKalmanMeasurementNoise.setProgress(600); // 1e0
                            seekBarPredictionThreshold.setProgress(80);
                            seekBarMaxPredictionFrames.setProgress(10);
                            break;
                        // SPATIAL_NORMAL - 保持默认值
                    }
                    break;
            }
        }
        
        // 恢复用户对轨迹可视化的设置
        switchEnableTrajectoryViz.setChecked(userTrajectoryVizSetting);
        
        // 更新依赖于SeekBar值的TextViews (因为setProgress不会自动触发listener)
        updateTextViewsAfterPreset();
    }
    
    // 辅助方法：在加载预设后更新所有TextView
    private void updateTextViewsAfterPreset() {
        // 显式计算并显示卡尔曼滤波参数值
        float processNoiseValue = (float) Math.pow(10, (seekBarKalmanProcessNoise.getProgress() / 100.0) - 6.0);
        float measurementNoiseValue = (float) Math.pow(10, (seekBarKalmanMeasurementNoise.getProgress() / 100.0) - 6.0);
        
        // 更新显示值
        textViewIouThresholdValue.setText(String.format(Locale.US, "%.2f", seekBarIouThreshold.getProgress() / 100.0f));
        textViewKalmanProcessNoiseValue.setText(formatScientific(processNoiseValue));
        textViewKalmanMeasurementNoiseValue.setText(formatScientific(measurementNoiseValue));
        textViewTrajectoryLengthValue.setText(String.valueOf(seekBarTrajectoryLength.getProgress()));
        textViewTrajectoryThicknessValue.setText(String.format(Locale.US, "%.1f", 0.5f + (seekBarTrajectoryThickness.getProgress() / 20.0f)));
        String[] colorModeLabels = {"单色", "渐变", "彩虹"};
        textViewTrajectoryColorModeValue.setText(colorModeLabels[seekBarTrajectoryColorMode.getProgress()]);
        textViewPredictionThresholdValue.setText(String.format(Locale.US, "%.2f", seekBarPredictionThreshold.getProgress() / 100.0f));
        textViewMaxPredictionFramesValue.setText(String.valueOf(seekBarMaxPredictionFrames.getProgress()));
        String[] lineTypeLabels = {"虚线", "实线"};
        textViewPredictionLineTypeValue.setText(lineTypeLabels[seekBarPredictionLineType.getProgress()]);
        
        // 记录当前设置的值，以便调试
        Log.d(TAG, "更新UI显示 - 过程噪声: " + formatScientific(processNoiseValue) + 
                  ", 测量噪声: " + formatScientific(measurementNoiseValue));
    }
} 