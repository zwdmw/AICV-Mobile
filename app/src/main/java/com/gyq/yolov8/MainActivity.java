package com.gyq.yolov8;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.os.Handler;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.util.SparseBooleanArray;
import android.view.LayoutInflater;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.AutoCompleteTextView;
import android.content.res.AssetManager;
import android.view.Surface;
import android.widget.PopupMenu;
import android.view.MenuItem;
import android.widget.ImageButton;
import android.widget.ToggleButton;
import android.content.SharedPreferences;
import android.preference.PreferenceManager;
import com.google.android.material.textfield.TextInputEditText;
import android.widget.FrameLayout;
import android.view.Gravity;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.slider.Slider;
import com.google.android.material.textfield.MaterialAutoCompleteTextView;
import com.google.android.material.textfield.TextInputLayout;
import com.google.android.material.switchmaterial.SwitchMaterial;
import com.google.android.material.textview.MaterialTextView;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.io.IOException;
import java.util.stream.Collectors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

import androidx.recyclerview.widget.RecyclerView;
import androidx.recyclerview.widget.LinearLayoutManager;

import android.view.Window;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;

import android.graphics.Color;
import android.widget.RadioGroup;
import android.widget.RadioButton;
import android.widget.Spinner;
import android.widget.ArrayAdapter;
import androidx.annotation.NonNull;
import ylov.colorpicker.ColorPickerDialog;
import com.google.android.material.card.MaterialCardView;

public class MainActivity extends Activity implements SurfaceHolder.Callback {
    public static final int REQUEST_CAMERA = 100; // note: 状态请求码(常量)
    private static final String TAG = MainActivity.class.getSimpleName(); // Add TAG for logging

    private Yolov8Ncnn yolov8ncnn = new Yolov8Ncnn();     // YOLO模型推理引擎，支持多种YOLO模型
    private int facing = 0; // 标志摄像头的前置或者后置

    private AutoCompleteTextView spinnerModel; // 主界面上的下拉菜单，用于切换模型
    private int current_model = 0;
    private String current_model_name = ""; // Store the selected model name
    private AutoCompleteTextView spinnerCPUGPU; // 下拉菜单，用于切换CPU/GPU
    private int current_cpugpu = 1; // 默认使用GPU (1)
    private AutoCompleteTextView spinnerDrawStyle; // Use AutoCompleteTextView for draw style selection
    private int current_detection_style = 9; // 默认使用量子光谱风格 (9)
    private String[] drawStyleNames; // To store style names for menu and spinner

    private SurfaceView cameraView; // SurfaceView 控件，用于显示相机画面
    private TextView textViewCount; // TextView to display detection count
    private Handler handler = new Handler(); // Handler for periodic updates
    private Runnable updateCountRunnable; // Runnable to update the count
    
    // 添加检测阈值相关的控件
    private Slider sliderThreshold;
    private TextView textViewThreshold;
    private float currentThreshold = 0.30f; // 默认检测阈值修改为0.30
    
    // 添加掩码阈值相关的控件
    private Slider sliderMaskThreshold;
    private TextView textViewMaskThreshold;
    private float currentMaskThreshold = 0.30f; // 默认掩码阈值修改为0.30
    private LinearLayout maskThresholdLayout; // Add layout reference
    
    // 当前选中的类别类型
    // private int currentCategoryType = 0; // 0 = 全部
    
    // 存储所有大类的类别ID
    // private int[][] allCategoryClassIds;

    // Define COCO Class Names locally for mapping
    private static final String[] COCO_CLASS_NAMES = new String[] {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    // 添加OIV数据集类别名称数组 - 应该包含完整的OIV类别列表
    // 这个数组必须与C++端的oiv_class_names保持一致，因此需要动态加载
    private String[] OIV_CLASS_NAMES = new String[1]; // Will be populated in runtime from JNI

    /**
     * 获取OIV数据集的类别名称数组
     * @return OIV类别名称数组
     */
    public String[] getOIVClassNames() {
        Log.d(TAG, "getOIVClassNames called, array length: " + (OIV_CLASS_NAMES != null ? OIV_CLASS_NAMES.length : "null"));
        if (OIV_CLASS_NAMES != null) {
            for (int i = 0; i < Math.min(10, OIV_CLASS_NAMES.length); i++) {
                Log.d(TAG, "OIV class " + i + ": " + OIV_CLASS_NAMES[i]);
            }
            if (OIV_CLASS_NAMES.length > 10) {
                Log.d(TAG, "... and " + (OIV_CLASS_NAMES.length - 10) + " more classes");
            }
        }
        return OIV_CLASS_NAMES;
    }

    // 添加掩码跟踪控制变量和控件
    private SwitchMaterial switchMaskTracking;
    private boolean maskTrackingEnabled = true; // 默认启用掩码跟踪
    private LinearLayout maskTrackingLayout; // 添加布局引用

    private Button buttonSwitchCamera;
    private Button buttonSelectClasses; // <-- Add Button for class selection
    private Button buttonSettings; // <-- Add Button for settings
    private ImageButton buttonStyleInfo; // Add ImageButton for style info
    private MaterialButton buttonHelp; // Add Help button

    private SharedPreferences sharedPreferences;
    private ExecutorService backgroundExecutor; // Executor for background tasks

    // Add tracking-related variables
    private float trackingIouThreshold = 0.45f;
    private int trackingMaxAge = 30;
    private int trackingMinHits = 3;
    private float trackingPositionNoiseMultiplier = 1.0f;
    private float trackingMeasurementNoiseMultiplier = 1.0f;
    private boolean trackingEnableMaskTracking = true;
    private boolean isTrackingEnabled = true;

    // Add a new member variable for class filtering
    private SparseBooleanArray selectedClassIds = new SparseBooleanArray();
    private Dialog classSelectionDialog;

    // 添加哈希表用于存储英文到中文的映射
    private Map<String, String> englishToChineseMap = new HashMap<>();

    // Add a map to store style parameters
    private Map<Integer, StyleParameters> styleParametersMap = new HashMap<>();

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        // Set the dark theme BEFORE calling super.onCreate and setContentView
        setTheme(R.style.Theme_OCSort_Tech);
        
        super.onCreate(savedInstanceState);
        // 根据Xml初始化界面布局(R是自动生成的资源类，控制所有的Resources)
        setContentView(R.layout.main);
        // 设置保持屏幕常亮
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // 创建背景线程池用于处理耗时操作
        backgroundExecutor = Executors.newFixedThreadPool(2);
        
        // Initialize the yolov8ncnn instance first
        yolov8ncnn = new Yolov8Ncnn();
        
        // 在后台加载资源
        backgroundExecutor.execute(() -> {
            // Load OIV class names from JNI
            OIV_CLASS_NAMES = yolov8ncnn.getOIVClassNames();
            Log.i(TAG, "Loaded " + OIV_CLASS_NAMES.length + " OIV class names from JNI");
            // Log the first few class names for debugging
            for (int i = 0; i < Math.min(10, OIV_CLASS_NAMES.length); i++) {
                Log.d(TAG, "OIV class " + i + ": " + OIV_CLASS_NAMES[i]);
            }
        });

        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this); // Initialize SharedPreferences

        // 获取布局中的相机画面预览控件 (SurfaceView)
        cameraView = (SurfaceView) findViewById(R.id.cameraview);
        // 设置相机传输过来的数据格式为 RGBA_8888 (具体的可以根据手机的不同而更改，一般是这个)
        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        // TODO: 为 SurfaceView 设置回调接口
        cameraView.getHolder().addCallback(this);
        
        // Get the TextView for detection count display
        textViewCount = (TextView) findViewById(R.id.textViewCount);
        // Make the count text more visible
        textViewCount.setTextColor(Color.WHITE);
        textViewCount.setTextSize(18);
        textViewCount.setShadowLayer(1.0f, 1.0f, 1.0f, Color.BLACK);
        textViewCount.setText("检测物体数量: 0");

        // 获取布局中的掩码阈值控件
        maskThresholdLayout = findViewById(R.id.maskThresholdLayout); // Make sure you have an ID 'maskThresholdLayout' in your main.xml wrapping the mask slider and textview

        // 获取切换前置后置摄像头的Button控件
        buttonSwitchCamera = findViewById(R.id.buttonSwitchCamera);
        // 为点击摄像头的按钮写一个槽函数(事件)，实现切换摄像头功能
        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                // 0->1, 1->0
                int new_facing = 1 - facing;
                yolov8ncnn.closeCamera();
                yolov8ncnn.openCamera(new_facing);
                facing = new_facing;
            }
        });
        
        // 初始化下拉菜单 - Dynamically load models from assets
        spinnerModel = (AutoCompleteTextView) findViewById(R.id.spinnerModel);
        List<String> models = getModelsFromAssets();
        if (models.isEmpty()) {
            Log.e(TAG, "No models found in assets directory!");
            // Handle error appropriately, maybe show a message and disable functionality
            Toast.makeText(this, "错误：在assets中找不到模型文件!", Toast.LENGTH_LONG).show();
            models.add("无模型"); // Add a placeholder
            current_model_name = "无模型";
        } else {
            // 设置默认模型为yolov8n-seg
            current_model_name = "yolov8n-seg";
            
            // 如果模型列表中没有yolov8n-seg，则使用第一个模型
            if (!models.contains(current_model_name)) {
                current_model_name = models.get(0);
                Log.w(TAG, "未找到yolov8n-seg模型，使用默认模型: " + current_model_name);
            } else {
                Log.i(TAG, "使用默认模型: " + current_model_name);
            }
        }
        
        ArrayAdapter<String> modelAdapter = new ArrayAdapter<>(this, R.layout.dropdown_item, models);
        spinnerModel.setAdapter(modelAdapter);
        spinnerModel.setText(current_model_name, false); // 设置默认值，不触发监听器

        // 设置模型选择监听器
        spinnerModel.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                 String selectedModelName = (String) parent.getItemAtPosition(position);
                if (!selectedModelName.equals(current_model_name)) {
                    current_model_name = selectedModelName;
                    // 重新加载(新的)模型
                    reload();
                }
            }
        });

        // 初始化CPU/GPU下拉菜单
        spinnerCPUGPU = (AutoCompleteTextView) findViewById(R.id.spinnerCPUGPU);
        String[] cpugpuOptions = getResources().getStringArray(R.array.cpugpu_array);
        ArrayAdapter<String> cpugpuAdapter = new ArrayAdapter<>(this, R.layout.dropdown_item, cpugpuOptions);
        spinnerCPUGPU.setAdapter(cpugpuAdapter);
        spinnerCPUGPU.setText(cpugpuOptions[1], false); // 设置默认值为GPU，不触发监听器
        
        // 设置CPU/GPU选择监听器
        spinnerCPUGPU.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                if (position != current_cpugpu) {
                    current_cpugpu = position;
                    reload(); // 同样的，切换了设备，模型也需要重载
                }
            }
        });
        
        // 初始化绘制样式下拉菜单
        spinnerDrawStyle = (AutoCompleteTextView) findViewById(R.id.spinnerDrawStyle);
        drawStyleNames = getResources().getStringArray(R.array.draw_style_array);
        ArrayAdapter<String> drawStyleAdapter = new ArrayAdapter<>(this, R.layout.dropdown_item, drawStyleNames);
        spinnerDrawStyle.setAdapter(drawStyleAdapter);
        spinnerDrawStyle.setText(drawStyleNames[9], false); // 默认使用量子光谱风格
        spinnerDrawStyle.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                if (position != current_detection_style) {
                    current_detection_style = position;
                    safeSetDetectionStyle(current_detection_style);
                    
                    // Also apply the current style parameters
                    StyleParameters params = styleParametersMap.get(current_detection_style);
                    if (params != null) {
                        applyStyleParameters(params);
                    }
                    
                    // 获取当前选择的风格描述
                    String[] styleDescriptions = getResources().getStringArray(R.array.draw_style_descriptions);
                    String description = (position < styleDescriptions.length) ? 
                        styleDescriptions[position] : "自定义风格";
                    
                    // 显示带风格描述的提示
                    Toast.makeText(MainActivity.this, 
                        "已设置风格: " + drawStyleNames[current_detection_style] + 
                        "\n" + description, 
                        Toast.LENGTH_SHORT).show();
                }
            }
        });
        
        // Initialize style parameters for all styles
        for (int i = 0; i <= 9; i++) {
            StyleParameters params = new StyleParameters(i);
            params.loadFromPreferences(this);
            styleParametersMap.put(i, params);
        }
        
        // 初始化样式信息按钮
        buttonStyleInfo = findViewById(R.id.buttonStyleInfo);
        buttonStyleInfo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (current_detection_style >= 0 && current_detection_style <= 9) {
                    showStyleParametersDialog(current_detection_style);
                } else {
                    showStyleInfoDialog();
                }
            }
        });
        
        // 初始化帮助按钮
        buttonHelp = findViewById(R.id.buttonHelp);
        buttonHelp.setOnClickListener(v -> showHelpDialog());
        
        // 初始化检测阈值相关控件
        sliderThreshold = findViewById(R.id.sliderThreshold);
        textViewThreshold = findViewById(R.id.textViewThreshold);
        textViewThreshold.setText(String.format("检测阈值: %.2f", currentThreshold));
        sliderThreshold.setValue(currentThreshold);
        sliderThreshold.addOnChangeListener(new Slider.OnChangeListener() {
            @Override
            public void onValueChange(Slider slider, float value, boolean fromUser) {
                currentThreshold = value;
                textViewThreshold.setText(String.format("检测阈值: %.2f", value));
                if (yolov8ncnn != null) {
                    yolov8ncnn.setDetectionThreshold(value);
                }
            }
        });
        
        // 初始化掩码阈值滑块
        sliderMaskThreshold = findViewById(R.id.sliderMaskThreshold);
        textViewMaskThreshold = findViewById(R.id.textViewMaskThreshold);
        textViewMaskThreshold.setText(String.format("掩码阈值: %.2f", currentMaskThreshold));
        sliderMaskThreshold.setValue(currentMaskThreshold);
        sliderMaskThreshold.addOnChangeListener(new Slider.OnChangeListener() {
            @Override
            public void onValueChange(Slider slider, float value, boolean fromUser) {
                currentMaskThreshold = value;
                textViewMaskThreshold.setText(String.format("掩码阈值: %.2f", value));
                if (yolov8ncnn != null) {
                    yolov8ncnn.setMaskThreshold(value);
                }
            }
        });
        
        // Update mask threshold visibility based on initial model
        updateMaskThresholdVisibility();

        // 第一次启动，用默认参数加载模型
        reload();
        
        // 初始化检测阈值
        yolov8ncnn.setDetectionThreshold(currentThreshold);
        
        // Create a Runnable to update the detection count display
        updateCountRunnable = new Runnable() {
            @Override
            public void run() {
                // Get the current detection count from the native code (filtered count)
                int count = yolov8ncnn.getFilteredDetectionCount();
                
                // Get the current FPS from the native code
                int fps = yolov8ncnn.getCurrentFPS();
                
                // Log the count periodically (every 20 updates to avoid log spam)
                if (Math.random() < 0.05) {
                    Log.d(TAG, "Detection count update: " + count + ", FPS: " + fps);
                }
                
                // Update the TextView with the current count
                textViewCount.setText("检测物体数量: " + count);
                
                // Update the FPS TextView
                TextView textViewFPS = findViewById(R.id.textViewFPS);
                if (textViewFPS != null) {
                    textViewFPS.setText("FPS: " + fps);
                }
                
                // Schedule the next update after 100ms for more real-time updates
                handler.postDelayed(this, 100);
            }
        };

        // 初始化掩码阈值控件和布局
        sliderMaskThreshold = findViewById(R.id.sliderMaskThreshold);
        textViewMaskThreshold = findViewById(R.id.textViewMaskThreshold);
        maskThresholdLayout = findViewById(R.id.maskThresholdLayout);
        
        // 初始化掩码跟踪开关和布局
        switchMaskTracking = findViewById(R.id.switchMaskTracking);
        maskTrackingLayout = findViewById(R.id.maskTrackingLayout);
        
        // 设置掩码跟踪开关的初始状态和事件监听
        if (switchMaskTracking != null) {
            switchMaskTracking.setChecked(maskTrackingEnabled);
            switchMaskTracking.setOnCheckedChangeListener((buttonView, isChecked) -> {
                maskTrackingEnabled = isChecked;
                // 更新掩码跟踪状态
                if (yolov8ncnn != null) {
                    yolov8ncnn.setEnableMaskTracking(maskTrackingEnabled);
                    Toast.makeText(MainActivity.this, 
                        "分割掩码跟踪已" + (maskTrackingEnabled ? "启用" : "禁用"), 
                        Toast.LENGTH_SHORT).show();
                }
            });
        } else {
            Log.w(TAG, "switchMaskTracking is null, cannot set listener.");
        }

        // Set OnClickListener for the settings button
        buttonSettings = findViewById(R.id.buttonSettings);
        buttonSettings.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, SettingsActivity.class);
            startActivity(intent);
        });
        
        // Initialize the class selection button
        buttonSelectClasses = findViewById(R.id.buttonSelectClasses);
        buttonSelectClasses.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showClassSelectionDialog();
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == SettingsActivity.REQUEST_CODE_SETTINGS && resultCode == RESULT_OK) {
            Log.d(TAG, "Returned from Settings");
            loadAndApplySettings(); // Reload and apply settings after returning from SettingsActivity
            reload(); // Optionally reload model if settings affect it significantly
        }
    }

    private void loadAndApplySettings() {
        float iouThreshold = sharedPreferences.getFloat(SettingsActivity.PREF_IOU_THRESHOLD, SettingsActivity.DEFAULT_IOU_THRESHOLD);
        int maxAge = sharedPreferences.getInt(SettingsActivity.PREF_MAX_AGE, SettingsActivity.DEFAULT_MAX_AGE);
        int minHits = sharedPreferences.getInt(SettingsActivity.PREF_MIN_HITS, SettingsActivity.DEFAULT_MIN_HITS);
        float processNoise = sharedPreferences.getFloat(SettingsActivity.PREF_KALMAN_PROCESS_NOISE, SettingsActivity.DEFAULT_KALMAN_PROCESS_NOISE);
        float measurementNoise = sharedPreferences.getFloat(SettingsActivity.PREF_KALMAN_MEASUREMENT_NOISE, SettingsActivity.DEFAULT_KALMAN_MEASUREMENT_NOISE);
        boolean enableMaskTracking = sharedPreferences.getBoolean(SettingsActivity.PREF_ENABLE_MASK_TRACKING, SettingsActivity.DEFAULT_ENABLE_MASK_TRACKING);
        // Load trajectory visualization settings
        boolean enableTrajectoryViz = sharedPreferences.getBoolean(SettingsActivity.PREF_ENABLE_TRAJECTORY_VIZ, SettingsActivity.DEFAULT_ENABLE_TRAJECTORY_VIZ);
        int trajectoryLength = sharedPreferences.getInt(SettingsActivity.PREF_TRAJECTORY_LENGTH, SettingsActivity.DEFAULT_TRAJECTORY_LENGTH);
        float trajectoryThickness = sharedPreferences.getFloat(SettingsActivity.PREF_TRAJECTORY_THICKNESS, SettingsActivity.DEFAULT_TRAJECTORY_THICKNESS);
        int trajectoryColorMode = sharedPreferences.getInt(SettingsActivity.PREF_TRAJECTORY_COLOR_MODE, SettingsActivity.DEFAULT_TRAJECTORY_COLOR_MODE);
        // 加载连续跟踪模式设置
        boolean enableContinuousTracking = sharedPreferences.getBoolean(SettingsActivity.PREF_ENABLE_CONTINUOUS_TRACKING, SettingsActivity.DEFAULT_ENABLE_CONTINUOUS_TRACKING);
        float predictionThreshold = sharedPreferences.getFloat(SettingsActivity.PREF_PREDICTION_THRESHOLD, SettingsActivity.DEFAULT_PREDICTION_THRESHOLD);
        int maxPredictionFrames = sharedPreferences.getInt(SettingsActivity.PREF_MAX_PREDICTION_FRAMES, SettingsActivity.DEFAULT_MAX_PREDICTION_FRAMES);
        int predictionLineType = sharedPreferences.getInt(SettingsActivity.PREF_PREDICTION_LINE_TYPE, SettingsActivity.DEFAULT_PREDICTION_LINE_TYPE);
        // 跟踪模式
        int trackingMode = sharedPreferences.getInt(SettingsActivity.PREF_TRACKING_MODE, SettingsActivity.DEFAULT_TRACKING_MODE);
        
        // 获取当前的运动子模式
        int motionSubMode;
        if (trackingMode == SettingsActivity.TRACKING_MODE_STABLE) {
            // 从稳定模式对应的存储键获取
            motionSubMode = sharedPreferences.getInt(SettingsActivity.PREF_MOTION_SUBMODE_STABLE, SettingsActivity.DEFAULT_MOTION_SUBMODE);
        } else if (trackingMode == SettingsActivity.TRACKING_MODE_HANDHELD) {
            // 从手持模式对应的存储键获取
            motionSubMode = sharedPreferences.getInt(SettingsActivity.PREF_MOTION_SUBMODE_HANDHELD, SettingsActivity.DEFAULT_MOTION_SUBMODE);
        } else {
            // 自定义模式，使用默认子模式
            motionSubMode = SettingsActivity.DEFAULT_MOTION_SUBMODE;
        }
        
        // 获取当前的空间分布模式
        int spatialDistribution;
        if (trackingMode == SettingsActivity.TRACKING_MODE_STABLE) {
            spatialDistribution = sharedPreferences.getInt(SettingsActivity.PREF_SPATIAL_DISTRIBUTION_STABLE, SettingsActivity.DEFAULT_SPATIAL_DISTRIBUTION);
        } else if (trackingMode == SettingsActivity.TRACKING_MODE_HANDHELD) {
            spatialDistribution = sharedPreferences.getInt(SettingsActivity.PREF_SPATIAL_DISTRIBUTION_HANDHELD, SettingsActivity.DEFAULT_SPATIAL_DISTRIBUTION);
        } else {
            spatialDistribution = SettingsActivity.DEFAULT_SPATIAL_DISTRIBUTION;
        }

        // 尝试使用新的带空间分布模式的方法
        boolean useNewMethod = true;
        try {
            boolean success = yolov8ncnn.setTrackingParamsWithModes(
                    iouThreshold, maxAge, minHits,
                    processNoise, measurementNoise, enableMaskTracking,
                    enableTrajectoryViz, trajectoryLength, trajectoryThickness, trajectoryColorMode,
                    enableContinuousTracking, predictionThreshold, maxPredictionFrames, predictionLineType,
                    trackingMode, motionSubMode, spatialDistribution);
                    
            if (success) {
                Log.i(TAG, "Applied tracking settings with modes: IoU=" + iouThreshold + 
                      ", TrackingMode=" + trackingMode + 
                      ", MotionSubMode=" + motionSubMode + 
                      ", SpatialDistribution=" + spatialDistribution);
                return;
            }
        } catch (UnsatisfiedLinkError e) {
            useNewMethod = false;
            Log.w(TAG, "setTrackingParamsWithModes not available, falling back to old method", e);
        }

        // 如果新方法不可用，回退到旧方法
        boolean success = yolov8ncnn.setTrackingParamsWithMode(
                iouThreshold, maxAge, minHits,
                processNoise, measurementNoise, enableMaskTracking,
                enableTrajectoryViz, trajectoryLength, trajectoryThickness, trajectoryColorMode,
                enableContinuousTracking, predictionThreshold, maxPredictionFrames, predictionLineType,
                trackingMode, motionSubMode);

        if (success) {
            Log.i(TAG, "Applied tracking settings: IoU=" + iouThreshold + ", MaxAge=" + maxAge + ", MinHits=" + minHits +
                    ", TrackingMode=" + trackingMode + ", MotionSubMode=" + motionSubMode); // Log motionSubMode
        } else {
            Log.w(TAG, "Failed to apply tracking settings via JNI.");
        }
        
        // Also ensure the general tracking enable/disable state is correct
        yolov8ncnn.setEnableTracking(isTrackingEnabled);
    }

    // Re-introduce a simplified reload method
    private void reload() {
        if (yolov8ncnn == null) {
            yolov8ncnn = new Yolov8Ncnn();
        }

        synchronized (this) {
            // 先关闭相机
            yolov8ncnn.closeCamera();
            
            // 短暂延迟以确保相机资源可以正确释放
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Log.e(TAG, "Sleep interrupted", e);
            }
            
            boolean ret_init = yolov8ncnn.loadModel(getAssets(), current_model_name, current_cpugpu);
            if (!ret_init) {
                Log.e(TAG, "yolov8ncnn loadModel failed");
                Toast.makeText(this, "加载模型失败，请检查模型文件", Toast.LENGTH_SHORT).show();
                return;
            }
    
            // 重新打开相机并设置输出窗口
            yolov8ncnn.openCamera(facing);
            if (cameraView.getHolder().getSurface() != null && cameraView.getHolder().getSurface().isValid()) {
                yolov8ncnn.setOutputWindow(cameraView.getHolder().getSurface());
            }
            
            // 设置当前的检测风格
            yolov8ncnn.setDetectionStyle(current_detection_style);
            
            // 根据模型类型设置不同的默认检测阈值
            if (current_model_name != null && current_model_name.contains("oiv")) {
                // OIV模型使用0.1的检测阈值
                currentThreshold = 0.1f;
                sliderThreshold.setValue(currentThreshold);
                textViewThreshold.setText(String.format("检测阈值: %.2f", currentThreshold));
                yolov8ncnn.setDetectionThreshold(currentThreshold);
                Log.i(TAG, "设置OIV模型检测阈值为: " + currentThreshold);
            } else if (currentThreshold != 0.3f) {
                // 非OIV模型恢复默认阈值0.3
                currentThreshold = 0.3f;
                sliderThreshold.setValue(currentThreshold);
                textViewThreshold.setText(String.format("检测阈值: %.2f", currentThreshold));
                yolov8ncnn.setDetectionThreshold(currentThreshold);
                Log.i(TAG, "设置标准模型检测阈值为: " + currentThreshold);
            }
        }
        
        // 加载模型对应的分类名映射，并传递给本地代码以改进显示
        backgroundExecutor.execute(() -> {
            loadClassNamesTranslation(current_model_name);
            // 会在loadClassNamesTranslation方法内部调用passChineseNamesToNative()
        });
        
        // 更新掩码阈值UI可见性
        updateMaskThresholdVisibility();
        
        // 更新数据集类型UI
        updateDatasetTypeUI();
        
        // 应用跟踪设置
        applyTrackingSettings();
        
        // 解决模型切换后掩码跟踪未正确初始化的问题
        // 先关闭再开启掩码跟踪，强制重新初始化跟踪器
        boolean isSegModel = current_model_name != null && current_model_name.contains("seg");
        if (isSegModel && maskTrackingEnabled) {
            Log.i(TAG, "重新初始化掩码跟踪...");
            yolov8ncnn.setEnableMaskTracking(false);
            yolov8ncnn.setEnableMaskTracking(true);
            Log.i(TAG, "掩码跟踪已重新初始化");
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        Log.i(TAG, "Surface changed: format=" + format + ", width=" + width + ", height=" + height);
        // 当 SurfaceView 的大小发生变化时调用此方法，设置输出的 Surface 为 Camera 的预览界面
        if (yolov8ncnn != null && holder.getSurface() != null && holder.getSurface().isValid()) {
            yolov8ncnn.setOutputWindow(holder.getSurface());
            
            // Start the detection count update when surface is ready
            handler.removeCallbacks(updateCountRunnable); // Remove any existing callbacks
            handler.post(updateCountRunnable); // Start the updates
            Log.i(TAG, "Started detection count updates");
        } else {
            Log.e(TAG, "Cannot set output window in surfaceChanged: invalid surface or yolov8ncnn is null");
        }
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        Log.i(TAG, "surfaceCreated");
        
        // Check camera permission
        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
            Log.e(TAG, "Camera permission not granted!");
            return;
        }

        // 确保yolov8ncnn实例有效
        if (yolov8ncnn == null) {
            Log.i(TAG, "创建新的yolov8ncnn实例");
            try {
                yolov8ncnn = new Yolov8Ncnn();
            } catch (Exception e) {
                Log.e(TAG, "创建yolov8ncnn实例失败: " + e.getMessage());
                Toast.makeText(this, "相机初始化失败，请重启应用", Toast.LENGTH_LONG).show();
                return;
            }
        }
        
        if (yolov8ncnn == null) {
            Log.e(TAG, "无法创建yolov8ncnn实例");
            Toast.makeText(this, "无法初始化相机模块", Toast.LENGTH_LONG).show();
            return;
        }

        // 尝试重新加载模型（如果之前已加载过）
        if (!current_model_name.isEmpty() && current_cpugpu >= 0) {
            try {
                Log.i(TAG, "尝试重新加载模型: " + current_model_name);
                boolean loaded = yolov8ncnn.loadModel(getAssets(), current_model_name, current_cpugpu);
                if (loaded) {
                    Log.i(TAG, "模型重新加载成功");
                }
            } catch (Exception e) {
                Log.e(TAG, "重新加载模型失败: " + e.getMessage());
            }
        }
        
        // Make sure to close camera first if it's already open
        try {
            boolean closed = yolov8ncnn.closeCamera();
            Log.d(TAG, "Pre-emptively closed camera: " + closed);
            // 添加短暂延迟，确保资源正确释放
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                Log.e(TAG, "Sleep interrupted", e);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error closing camera: " + e.getMessage());
        }

        // 确保Surface有效
        if (holder == null || holder.getSurface() == null || !holder.getSurface().isValid()) {
            Log.e(TAG, "Surface is null or invalid");
            Toast.makeText(this, "相机预览界面无效", Toast.LENGTH_LONG).show();
            return;
        }

        // 多次尝试打开相机和设置窗口
        int maxTries = 3;
        boolean success = false;
        
        for (int i = 0; i < maxTries && !success; i++) {
            // Open camera with selected facing direction
            try {
                Log.d(TAG, "Opening camera with facing: " + facing + " (attempt " + (i+1) + ")");
                boolean opened = yolov8ncnn.openCamera(facing);
                if (!opened) {
                    Log.e(TAG, "Failed to open camera on attempt " + (i+1));
                    // 如果不是最后一次尝试，等待一会再试
                    if (i < maxTries - 1) {
                        try {
                            Thread.sleep(300);
                        } catch (InterruptedException e) {
                            Log.e(TAG, "Sleep interrupted", e);
                        }
                        
                        // 如果超过一次尝试还不成功，尝试重建yolov8ncnn实例
                        if (i >= 1) {
                            try {
                                Log.i(TAG, "尝试重新创建yolov8ncnn实例（尝试" + (i+1) + "）");
                                yolov8ncnn = new Yolov8Ncnn();
                                
                                // 如果之前有模型，尝试重新加载
                                if (!current_model_name.isEmpty() && current_cpugpu >= 0) {
                                    yolov8ncnn.loadModel(getAssets(), current_model_name, current_cpugpu);
                                }
                            } catch (Exception e) {
                                Log.e(TAG, "重新创建yolov8ncnn失败: " + e.getMessage());
                            }
                        }
                        
                        continue;
                    }
                    Toast.makeText(this, "无法打开相机，请检查权限或重启应用", Toast.LENGTH_LONG).show();
                    return;
                }
                Log.i(TAG, "Camera opened successfully");
                
                // 短暂延迟确保相机完全准备好
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Log.e(TAG, "Sleep interrupted", e);
                }
                
                // Set output window
                boolean setWindow = yolov8ncnn.setOutputWindow(holder.getSurface());
                if (!setWindow) {
                    Log.e(TAG, "Failed to set output window on attempt " + (i+1));
                    yolov8ncnn.closeCamera();
                    
                    // 如果不是最后一次尝试，等待一会再试
                    if (i < maxTries - 1) {
                        try {
                            Thread.sleep(300);
                        } catch (InterruptedException e) {
                            Log.e(TAG, "Sleep interrupted", e);
                        }
                        continue;
                    }
                    
                    Toast.makeText(this, "设置相机输出窗口失败", Toast.LENGTH_LONG).show();
                    return;
                }
                
                // 如果到这里，说明成功了
                success = true;
                Log.i(TAG, "Output window set successfully on attempt " + (i+1));
                
            } catch (Exception e) {
                Log.e(TAG, "Exception in camera setup: " + e.getMessage());
                if (i == maxTries - 1) {
                    Toast.makeText(this, "相机设置异常: " + e.getMessage(), Toast.LENGTH_LONG).show();
                }
            }
        }
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i(TAG, "Surface destroyed");
        // 先关闭相机再释放Surface
        if (yolov8ncnn != null) {
            synchronized (this) {
                yolov8ncnn.closeCamera();
                // 移除setOutputWindow(null)调用，这行代码会导致JNI崩溃
                // yolov8ncnn.setOutputWindow(null);
            }
        }
    }

    // note: 从后台重新回到该程序时应该做的
    @Override
    public void onResume() {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        // 确保SurfaceView已准备好再打开相机
        if (cameraView.getHolder().getSurface() != null && cameraView.getHolder().getSurface().isValid()) {
            synchronized (this) {
                yolov8ncnn.openCamera(facing);
                yolov8ncnn.setOutputWindow(cameraView.getHolder().getSurface());
            }
            
            // Start periodic updates of the detection count
            handler.post(updateCountRunnable);
            applyTrackingSettings(); // Apply settings when the activity resumes
        } else {
            Log.w(TAG, "Surface不可用，稍后将通过surfaceChanged设置");
        }
    }

    // note: 程序到后台时，关闭摄像头
    @Override
    public void onPause() {
        super.onPause();

        // 在暂停前关闭相机并移除Surface
        if (yolov8ncnn != null) {
            synchronized (this) {
                yolov8ncnn.closeCamera();
            }
        }

        // Stop the periodic updates when app is paused
        handler.removeCallbacks(updateCountRunnable);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        
        // 优雅地关闭线程池
        if (backgroundExecutor != null) {
            backgroundExecutor.shutdown();
            try {
                if (!backgroundExecutor.awaitTermination(500, java.util.concurrent.TimeUnit.MILLISECONDS)) {
                    backgroundExecutor.shutdownNow();
                }
            } catch (InterruptedException e) {
                backgroundExecutor.shutdownNow();
            }
        }

        // 清理yolov8ncnn资源
        if (yolov8ncnn != null) {
            synchronized (this) {
                yolov8ncnn.closeCamera();
                yolov8ncnn = null;
            }
        }
    }

    // Add this method to get model names from assets
    private List<String> getModelsFromAssets() {
        AssetManager assetManager = getAssets();
        List<String> modelFiles = new ArrayList<>();
        try {
            String[] files = assetManager.list(""); // List files in the root of assets
            if (files != null) {
                for (String file : files) {
                    if (file.endsWith(".param")) {
                        // Extract model name without extension
                        String modelName = file.substring(0, file.lastIndexOf('.'));
                        // Check if the corresponding .bin file exists
                        boolean binExists = false;
                        for(String f : files) {
                           if (f.equals(modelName + ".bin")) {
                               binExists = true;
                               break;
                           }
                        }
                        if(binExists) {
                            modelFiles.add(modelName);
                        } else {
                           Log.w(TAG, "Found " + file + " but missing corresponding .bin file.");
                        }
                    }
                }
            }
            // Sort models alphabetically for consistency
            java.util.Collections.sort(modelFiles);
        } catch (IOException e) {
            Log.e(TAG, "Error listing assets: " + e.getMessage());
        }
        return modelFiles;
    }
    
    // Add this method to update mask threshold visibility
    private void updateMaskThresholdVisibility() {
        // 如果模型名中不包含"seg"，就是普通检测模型，隐藏掩码阈值控件
        boolean isSegmentationModel = current_model_name != null && current_model_name.contains("seg");
        if (maskThresholdLayout != null) {
             maskThresholdLayout.setVisibility(isSegmentationModel ? View.VISIBLE : View.GONE);
        } else {
            Log.w(TAG, "maskThresholdLayout is null, cannot update visibility.");
        }
        
        // 更新掩码跟踪布局可见性
        if (maskTrackingLayout != null) {
            maskTrackingLayout.setVisibility(isSegmentationModel ? View.VISIBLE : View.GONE);
        } else {
            Log.w(TAG, "maskTrackingLayout is null, cannot update visibility.");
        }
    }

    // 显示所有检测风格的详情对话框
    private void showStyleInfoDialog() {
        // 创建并初始化对话框
        Dialog dialog = new Dialog(this, R.style.ThemeOverlay_OCSort_Tech_MaterialAlertDialog);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.dialog_style_info);
        
        // 设置对话框宽度为屏幕宽度的90%
        Window window = dialog.getWindow();
        if (window != null) {
            WindowManager.LayoutParams layoutParams = new WindowManager.LayoutParams();
            layoutParams.copyFrom(window.getAttributes());
            layoutParams.width = (int) (getResources().getDisplayMetrics().widthPixels * 0.9);
            window.setAttributes(layoutParams);
        }
        
        // 获取风格名称和描述
        String[] styleNames = getResources().getStringArray(R.array.draw_style_array);
        String[] styleDescriptions = getResources().getStringArray(R.array.draw_style_descriptions);
        
        // 获取风格列表容器
        LinearLayout styleInfoContainer = dialog.findViewById(R.id.styleInfoContainer);
        
        // 为每个风格添加一个列表项
        for (int i = 0; i < styleNames.length && i < styleDescriptions.length; i++) {
            // 填充自定义风格项布局
            View itemView = getLayoutInflater().inflate(R.layout.item_style_info, styleInfoContainer, false);
            
            // 设置风格名称和描述
            TextView textStyleName = itemView.findViewById(R.id.textStyleName);
            TextView textStyleDescription = itemView.findViewById(R.id.textStyleDescription);
            View styleIndicator = itemView.findViewById(R.id.styleIndicator);
            
            textStyleName.setText(styleNames[i]);
            textStyleDescription.setText(styleDescriptions[i]);
            
            // 当前选中的风格高亮显示
            if (i == current_detection_style) {
                textStyleName.setTextColor(getResources().getColor(R.color.tech_accent));
                textStyleDescription.setTextColor(getResources().getColor(R.color.white));
                styleIndicator.setBackgroundResource(R.drawable.style_indicator_dot);
            } else {
                styleIndicator.setBackgroundColor(Color.TRANSPARENT);
            }
            
            // 设置点击事件，点击时选择该风格并关闭对话框
            final int position = i;
            itemView.setOnClickListener(v -> {
                if (position != current_detection_style) {
                    current_detection_style = position;
                    safeSetDetectionStyle(current_detection_style);
                    
                    // Also apply the current style parameters
                    StyleParameters params = styleParametersMap.get(current_detection_style);
                    if (params != null) {
                        applyStyleParameters(params);
                    }
                    
                    // 更新Spinner显示
                    spinnerDrawStyle.setText(styleNames[position]);
                    
                    // 显示提示
                    Toast.makeText(MainActivity.this, 
                        "已设置风格: " + styleNames[current_detection_style], 
                        Toast.LENGTH_SHORT).show();
                }
                dialog.dismiss();
            });
            
            // 添加到容器
            styleInfoContainer.addView(itemView);
        }
        
        // 设置关闭按钮点击事件
        Button btnClose = dialog.findViewById(R.id.btnClose);
        btnClose.setOnClickListener(v -> dialog.dismiss());
        
        // 显示对话框
        dialog.show();
    }

    // 显示应用帮助指南对话框
    private void showHelpDialog() {
        // 创建并初始化对话框
        Dialog dialog = new Dialog(this, R.style.ThemeOverlay_OCSort_Tech_MaterialAlertDialog);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.dialog_app_help);
        
        // 设置对话框宽度为屏幕宽度的90%
        Window window = dialog.getWindow();
        if (window != null) {
            WindowManager.LayoutParams layoutParams = new WindowManager.LayoutParams();
            layoutParams.copyFrom(window.getAttributes());
            layoutParams.width = (int) (getResources().getDisplayMetrics().widthPixels * 0.9);
            window.setAttributes(layoutParams);
        }
        
        // 获取当前风格的颜色
        int styleColor;
        try {
            StyleParameters params = styleParametersMap.get(current_detection_style);
            if (params != null && params.getBoxColor() != 0) {
                styleColor = params.getBoxColor();
            } else {
                styleColor = getResources().getColor(R.color.tech_accent);
            }
            
            // 应用颜色到标题和指示器
            TextView textViewHelpTitle = dialog.findViewById(R.id.textViewHelpTitle);
            View titleIndicator = dialog.findViewById(R.id.titleIndicator);
            
            if (textViewHelpTitle != null) {
                textViewHelpTitle.setTextColor(styleColor);
            }
            
            if (titleIndicator != null) {
                titleIndicator.setBackgroundColor(styleColor);
            }
            
            // 设置章节标题颜色
            setHeadingTextColors(dialog, styleColor);
            
            // 设置关闭按钮颜色
            MaterialButton btnCloseHelp = dialog.findViewById(R.id.btnCloseHelp);
            if (btnCloseHelp != null) {
                btnCloseHelp.setOnClickListener(v -> dialog.dismiss());
                btnCloseHelp.setBackgroundTintList(android.content.res.ColorStateList.valueOf(styleColor));
            }
            
            // 设置卡片边框颜色
            MaterialCardView cardView = dialog.findViewById(R.id.helpCardView);
            if (cardView != null) {
                cardView.setStrokeColor(styleColor);
            }
            
        } catch (Exception e) {
            Log.e(TAG, "Error applying style color to help dialog: " + e.getMessage());
        }
        
        // 显示对话框
        dialog.show();
    }
    
    // 设置所有章节标题的颜色和图标颜色
    private void setHeadingTextColors(Dialog dialog, int styleColor) {
        try {
            LinearLayout contentLayout = dialog.findViewById(R.id.helpContentLayout);
            if (contentLayout != null) {
                for (int i = 0; i < contentLayout.getChildCount(); i++) {
                    View child = contentLayout.getChildAt(i);
                    if (child instanceof TextView) {
                        TextView textView = (TextView) child;
                        // 检查是否是标题（通过 drawableStart 和 textStyle 判断）
                        if (textView.getTypeface() != null && textView.getTypeface().isBold() && 
                            textView.getCompoundDrawables()[0] != null) { // Headers have a drawableStart
                            
                            // 设置文本颜色
                            textView.setTextColor(styleColor);
                            
                            // 设置图标颜色
                            textView.getCompoundDrawables()[0].setColorFilter(styleColor, android.graphics.PorterDuff.Mode.SRC_IN);
                        }
                    }
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Error setting heading colors: " + e.getMessage());
        }
    }

    // Method to load settings from SharedPreferences and apply them via JNI
    private void applyTrackingSettings() {
        // 从SharedPreferences加载跟踪设置
        float iouThreshold = sharedPreferences.getFloat(SettingsActivity.PREF_IOU_THRESHOLD, SettingsActivity.DEFAULT_IOU_THRESHOLD);
        int maxAge = sharedPreferences.getInt(SettingsActivity.PREF_MAX_AGE, SettingsActivity.DEFAULT_MAX_AGE);
        int minHits = sharedPreferences.getInt(SettingsActivity.PREF_MIN_HITS, SettingsActivity.DEFAULT_MIN_HITS);
        float processNoise = sharedPreferences.getFloat(SettingsActivity.PREF_KALMAN_PROCESS_NOISE, SettingsActivity.DEFAULT_KALMAN_PROCESS_NOISE);
        float measurementNoise = sharedPreferences.getFloat(SettingsActivity.PREF_KALMAN_MEASUREMENT_NOISE, SettingsActivity.DEFAULT_KALMAN_MEASUREMENT_NOISE);
        boolean enableMaskTracking = sharedPreferences.getBoolean(SettingsActivity.PREF_ENABLE_MASK_TRACKING, SettingsActivity.DEFAULT_ENABLE_MASK_TRACKING);
        // 轨迹可视化参数
        boolean enableTrajectoryViz = sharedPreferences.getBoolean(SettingsActivity.PREF_ENABLE_TRAJECTORY_VIZ, SettingsActivity.DEFAULT_ENABLE_TRAJECTORY_VIZ);
        int trajectoryLength = sharedPreferences.getInt(SettingsActivity.PREF_TRAJECTORY_LENGTH, SettingsActivity.DEFAULT_TRAJECTORY_LENGTH);
        float trajectoryThickness = sharedPreferences.getFloat(SettingsActivity.PREF_TRAJECTORY_THICKNESS, SettingsActivity.DEFAULT_TRAJECTORY_THICKNESS);
        int trajectoryColorMode = sharedPreferences.getInt(SettingsActivity.PREF_TRAJECTORY_COLOR_MODE, SettingsActivity.DEFAULT_TRAJECTORY_COLOR_MODE);
        // 连续跟踪参数（也用于控制运动补偿）
        boolean enableContinuousTracking = sharedPreferences.getBoolean(SettingsActivity.PREF_ENABLE_CONTINUOUS_TRACKING, SettingsActivity.DEFAULT_ENABLE_CONTINUOUS_TRACKING);
        float predictionThreshold = sharedPreferences.getFloat(SettingsActivity.PREF_PREDICTION_THRESHOLD, SettingsActivity.DEFAULT_PREDICTION_THRESHOLD);
        int maxPredictionFrames = sharedPreferences.getInt(SettingsActivity.PREF_MAX_PREDICTION_FRAMES, SettingsActivity.DEFAULT_MAX_PREDICTION_FRAMES);
        int predictionLineType = sharedPreferences.getInt(SettingsActivity.PREF_PREDICTION_LINE_TYPE, SettingsActivity.DEFAULT_PREDICTION_LINE_TYPE);
        
        // 跟踪模式
        int trackingMode = sharedPreferences.getInt(SettingsActivity.PREF_TRACKING_MODE, SettingsActivity.DEFAULT_TRACKING_MODE);
        
        // 获取当前的运动子模式
        int motionSubMode;
        if (trackingMode == SettingsActivity.TRACKING_MODE_STABLE) {
            // 从稳定模式对应的存储键获取
            motionSubMode = sharedPreferences.getInt(SettingsActivity.PREF_MOTION_SUBMODE_STABLE, SettingsActivity.DEFAULT_MOTION_SUBMODE);
        } else if (trackingMode == SettingsActivity.TRACKING_MODE_HANDHELD) {
            // 从手持模式对应的存储键获取
            motionSubMode = sharedPreferences.getInt(SettingsActivity.PREF_MOTION_SUBMODE_HANDHELD, SettingsActivity.DEFAULT_MOTION_SUBMODE);
        } else {
            // 自定义模式，使用默认子模式
            motionSubMode = SettingsActivity.DEFAULT_MOTION_SUBMODE;
        }
        
        // 获取当前的空间分布模式
        int spatialDistribution;
        if (trackingMode == SettingsActivity.TRACKING_MODE_STABLE) {
            spatialDistribution = sharedPreferences.getInt(SettingsActivity.PREF_SPATIAL_DISTRIBUTION_STABLE, SettingsActivity.DEFAULT_SPATIAL_DISTRIBUTION);
        } else if (trackingMode == SettingsActivity.TRACKING_MODE_HANDHELD) {
            spatialDistribution = sharedPreferences.getInt(SettingsActivity.PREF_SPATIAL_DISTRIBUTION_HANDHELD, SettingsActivity.DEFAULT_SPATIAL_DISTRIBUTION);
        } else {
            spatialDistribution = SettingsActivity.DEFAULT_SPATIAL_DISTRIBUTION;
        }
        
        // 保存一份成员变量备用
        this.trackingIouThreshold = iouThreshold;
        this.trackingMaxAge = maxAge;
        this.trackingMinHits = minHits;
        this.trackingPositionNoiseMultiplier = processNoise / SettingsActivity.DEFAULT_KALMAN_PROCESS_NOISE;
        this.trackingMeasurementNoiseMultiplier = measurementNoise / SettingsActivity.DEFAULT_KALMAN_MEASUREMENT_NOISE;
        this.trackingEnableMaskTracking = enableMaskTracking;
        
        Log.i(TAG, "Applying tracking settings: IoU=" + iouThreshold + ", MaxAge=" + maxAge + ", MinHits=" + minHits + 
              ", PNoise=" + processNoise + ", MNoise=" + measurementNoise + ", MaskTrack=" + enableMaskTracking +
              ", ContinuousTracking=" + enableContinuousTracking + ", TrackingMode=" + trackingMode + 
              ", MotionSubMode=" + motionSubMode + ", SpatialDistribution=" + spatialDistribution);
        
        // 检查是否支持新的带空间分布模式的方法
        boolean useNewMethod = true;
        try {
            // 尝试使用新方法
            boolean success = yolov8ncnn.setTrackingParamsWithModes(
                    iouThreshold, maxAge, minHits,
                    processNoise, measurementNoise, enableMaskTracking,
                    enableTrajectoryViz, trajectoryLength, trajectoryThickness, trajectoryColorMode,
                    enableContinuousTracking, predictionThreshold, maxPredictionFrames, predictionLineType,
                    trackingMode, motionSubMode, spatialDistribution);
            
            if (success) {
                Log.i(TAG, "Successfully applied tracking settings with all modes");
                return;
            }
        } catch (UnsatisfiedLinkError e) {
            // 新方法可能尚未实现，回退到旧方法
            useNewMethod = false;
            Log.w(TAG, "New tracking method not implemented, falling back to old method", e);
        }
        
        // 如果新方法不可用或失败，使用旧方法
        if (!useNewMethod) {
            boolean success = yolov8ncnn.setTrackingParamsWithMode(
                    iouThreshold, maxAge, minHits,
                    processNoise, measurementNoise, enableMaskTracking,
                    enableTrajectoryViz, trajectoryLength, trajectoryThickness, trajectoryColorMode,
                    enableContinuousTracking, predictionThreshold, maxPredictionFrames, predictionLineType,
                    trackingMode, motionSubMode);
            
            if (success) {
                Log.i(TAG, "Successfully applied tracking settings with mode (old method)");
            } else {
                Log.e(TAG, "Failed to apply tracking settings via JNI");
            }
        }
    }

    // 为COCO数据集设置中英文映射
    private void setupCOCOChineseToEnglishMap(Map<String, String> map) {
        loadChineseToEnglishMapFromAssets(map, "coco_zh_en.txt");
    }

    // 为OIV数据集设置中英文映射
    private void setupOIVChineseToEnglishMap(Map<String, String> map) {
        loadChineseToEnglishMapFromAssets(map, "oiv_zh_en.txt");
    }
    
    // 从assets目录中的txt文件加载中英文映射
    private void loadChineseToEnglishMapFromAssets(Map<String, String> map, String fileName) {
        try {
            InputStream is = getAssets().open(fileName);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            String line;
            
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                
                // 移除可能的序号前缀（如 "19. "）
                if (line.matches("\\d+\\..*")) {
                    line = line.replaceFirst("\\d+\\.", "").trim();
                }
                
                // 按 " - " 分割获取中文和英文名
                String[] parts = line.split(" - ");
                if (parts.length >= 2) {
                    String chineseName = parts[0].trim();
                    String englishName = parts[1].trim();
                    map.put(chineseName, englishName);
                    Log.d(TAG, "Loaded mapping: " + chineseName + " -> " + englishName);
                }
            }
            
            reader.close();
            is.close();
            
            Log.i(TAG, "Loaded " + map.size() + " Chinese-English mappings from " + fileName);
        } catch (IOException e) {
            Log.w(TAG, "Failed to load Chinese-English mappings from " + fileName + ": " + e.getMessage());
        }
    }

    // 从txt文件加载类名翻译
    // 注意：所有类别的中英文名称现在从assets目录下的txt文件读取，不再使用strings.xml中的资源
    // 每个模型对应一个同名的txt文件（如yolov8n.txt），文件格式为"数字. 英文名 - 中文名"
    // 如果找不到特定模型的txt文件，会尝试加载默认的yolov8n.txt或yolov8n-oiv7.txt（根据数据集类型）
    private void loadClassNamesTranslation(String modelName) {
        Log.i(TAG, "Loading class name translations for " + modelName);
        englishToChineseMap.clear();
        
        try {
            // 尝试加载与模型同名的txt文件（如yolov8s.txt）
            String fileName = modelName + ".txt";
            InputStream is = getAssets().open(fileName);
            Log.i(TAG, "Successfully opened translation file: " + fileName); // Add this log
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            String line;
            
            while ((line = reader.readLine()) != null) {
                // 解析格式为 "数字. 英文名 - 中文名" 或 "英文名 - 中文名"
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                
                // 移除可能的序号前缀（如 "19. "）
                if (line.matches("\\d+\\..*")) {
                    line = line.replaceFirst("\\d+\\.", "").trim();
                }
                
                // 按 " - " 分割获取英文和中文名
                String[] parts = line.split(" - ");
                if (parts.length >= 2) {
                    String englishName = parts[0].trim();
                    String chineseName = parts[1].trim();
                    englishToChineseMap.put(englishName.toLowerCase(), chineseName);
                    Log.d(TAG, "Loaded translation: " + englishName + " -> " + chineseName);
                }
            }
            
            reader.close();
            is.close();
            
            Log.i(TAG, "Loaded " + englishToChineseMap.size() + " translations from " + fileName);
            
            // 将中文名称传递给本地代码以提高检测显示效果
            passChineseNamesToNative();
        } catch (IOException e) {
            Log.w(TAG, "Failed to load class translations from txt: " + e.getMessage());
            
            // 如果无法加载指定模型的txt文件，尝试加载通用文件
            try {
                // 根据数据集类型选择加载不同的文件
                boolean isOIV = modelName.contains("oiv");
                String fallbackFile = isOIV ? "yolov8n-oiv7.txt" : "yolov8n.txt";
                
                Log.i(TAG, "Trying fallback translation file: " + fallbackFile);
                InputStream is = getAssets().open(fallbackFile);
                BufferedReader reader = new BufferedReader(new InputStreamReader(is));
                String line;
                
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty() || line.startsWith("#")) continue;
                    
                    if (line.matches("\\d+\\..*")) {
                        line = line.replaceFirst("\\d+\\.", "").trim();
                    }
                    
                    String[] parts = line.split(" - ");
                    if (parts.length >= 2) {
                        String englishName = parts[0].trim();
                        String chineseName = parts[1].trim();
                        englishToChineseMap.put(englishName.toLowerCase(), chineseName);
                    }
                }
                
                reader.close();
                is.close();
                
                Log.i(TAG, "Loaded " + englishToChineseMap.size() + " translations from fallback file " + fallbackFile);
                
                // 从后备文件加载后也要传递中文名称
                passChineseNamesToNative();
            } catch (IOException ex) {
                Log.e(TAG, "Failed to load fallback translations: " + ex.getMessage());
            }
        }
    }

    // 修改getChineseNameForClass方法，优先使用txt加载的翻译
    private String getChineseNameForClass(String englishName, int datasetType) {
        Log.d(TAG, "getChineseNameForClass called for: " + englishName + ", datasetType: " + datasetType);
        
        // 从加载的映射中查找
        if (englishName != null && !englishToChineseMap.isEmpty()) {
            String chineseName = englishToChineseMap.get(englishName.toLowerCase());
            if (chineseName != null) {
                Log.d(TAG, "Found translation in map for " + englishName + " -> " + chineseName);
                return chineseName;
            }
        }
        
        // 如果映射中没有，则直接返回英文名称
        // 不再从字符串资源中查找
        Log.d(TAG, "No translation found for " + englishName + ", returning original name");
        return englishName;
    }

    // 获取类别的双语名称（中文/英文）
    private String getBilingualClassName(String englishName, int datasetType) {
        Log.d(TAG, "getBilingualClassName called for: " + englishName + ", datasetType: " + datasetType);
        if (englishName == null) {
            Log.e(TAG, "getBilingualClassName received null englishName");
            return "unknown";
        }
        
        String chineseName = getChineseNameForClass(englishName, datasetType);
        if (chineseName != null && !chineseName.equals(englishName)) {
            String result = chineseName + " / " + englishName;
            Log.d(TAG, "Created bilingual name: " + result);
            return result;
        }
        
        Log.d(TAG, "Using English name only: " + englishName);
        return englishName;
    }
    
    // 使用双语名称更新类别绘制逻辑

    static {
        // ... existing code ...
    }

    private void updateDatasetTypeUI() {
        if (yolov8ncnn == null) return;
        
        int datasetType = yolov8ncnn.getDatasetType();
        String datasetName = datasetType == Yolov8Ncnn.DATASET_OIV ? "OIV数据集" : "COCO数据集";
        Log.i(TAG, "当前模型使用的数据集: " + datasetName);
        
        // 如果需要，可以在UI中显示当前数据集类型
        Toast.makeText(this, "模型使用的数据集: " + datasetName, Toast.LENGTH_SHORT).show();
    }

    // 显示类别选择对话框
    private void showClassSelectionDialog() {
        if (classSelectionDialog != null && classSelectionDialog.isShowing()) {
            classSelectionDialog.dismiss();
        }
        
        // 创建对话框
        Dialog classSelectionDialog = new Dialog(this);
        classSelectionDialog.setContentView(R.layout.class_selection_dialog);
        classSelectionDialog.getWindow().setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        classSelectionDialog.getWindow().setBackgroundDrawableResource(android.R.color.transparent);
        this.classSelectionDialog = classSelectionDialog;
        
        // 初始化对话框中的控件
        RecyclerView recyclerView = classSelectionDialog.findViewById(R.id.recyclerViewClasses);
        Button btnSelectAll = classSelectionDialog.findViewById(R.id.btnSelectAll);
        Button btnClearAll = classSelectionDialog.findViewById(R.id.btnClearAll);
        Button btnCancel = classSelectionDialog.findViewById(R.id.btnCancel);
        Button btnApply = classSelectionDialog.findViewById(R.id.btnApply);
        TextInputEditText searchEditText = classSelectionDialog.findViewById(R.id.editTextSearch);
        
        // 获取数据集类型
        int datasetType = yolov8ncnn.getDatasetType();
        Log.d(TAG, "Dataset type: " + datasetType);
        
        // 根据数据集类型选择类别名称数组
        String[] classNames;
        if (datasetType == Yolov8Ncnn.DATASET_OIV) {
            classNames = OIV_CLASS_NAMES;
            Log.d(TAG, "Using OIV class names, length: " + (classNames != null ? classNames.length : "null"));
        } else {
            classNames = COCO_CLASS_NAMES;
            Log.d(TAG, "Using COCO class names, length: " + (classNames != null ? classNames.length : "null"));
        }
        
        if (classNames == null) {
            Log.e(TAG, "Class names array is null!");
            Toast.makeText(this, "错误：类别名称数组为空", Toast.LENGTH_SHORT).show();
            classSelectionDialog.dismiss();
            return;
        }
        
        // Prepare data for the adapter
        List<ClassAdapter.ClassItem> classItems = new ArrayList<>();
        for (int i = 0; i < classNames.length; i++) {
            String currentClassName = classNames[i];
            if (currentClassName == null) {
                Log.e(TAG, "Null class name at index " + i);
                currentClassName = "unknown_" + i;
            }
            
            // Get bilingual name once
            String displayName = getBilingualClassName(currentClassName, datasetType);
            classItems.add(new ClassAdapter.ClassItem(i, currentClassName, displayName));
        }
        
        // Create and set adapter
        ClassAdapter adapter = new ClassAdapter(classItems, selectedClassIds);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(adapter);
        
        // Set up search functionality with optimized implementation
        searchEditText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            
            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {}
            
            @Override
            public void afterTextChanged(Editable s) {
                adapter.getFilter().filter(s.toString());
            }
        });
        
        // Set up select all button
        btnSelectAll.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                adapter.selectAll();
            }
        });
        
        // Set up clear all button
        btnClearAll.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                adapter.clearAll();
            }
        });
        
        // Set up cancel button
        btnCancel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                classSelectionDialog.dismiss();
            }
        });
        
        // Set up apply button
        btnApply.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                applyClassSelection();
                classSelectionDialog.dismiss();
            }
        });
        
        // Show dialog
        classSelectionDialog.show();
    }
    
    // Method to apply class selection
    private void applyClassSelection() {
        Log.d(TAG, "applyClassSelection called");
        // Count selected classes
        int selectedCount = 0;
        for (int i = 0; i < selectedClassIds.size(); i++) {
            if (selectedClassIds.valueAt(i)) {
                selectedCount++;
            }
        }
        
        Log.d(TAG, "Selected class count: " + selectedCount);
        
        // If no classes selected, clear filter
        if (selectedCount == 0) {
            Log.d(TAG, "No classes selected, clearing filter");
            yolov8ncnn.clearClassFilter();
            Toast.makeText(this, "已清除类别过滤，将显示所有类别", Toast.LENGTH_SHORT).show();
            return;
        }
        
        // Create array of selected class IDs
        int[] enabledClassIds = new int[selectedCount];
        int index = 0;
        for (int i = 0; i < selectedClassIds.size(); i++) {
            if (selectedClassIds.valueAt(i)) {
                int classId = selectedClassIds.keyAt(i);
                enabledClassIds[index++] = classId;
                Log.d(TAG, "Added class ID " + classId + " to enabled classes");
            }
        }
        
        // 新增日志用于排查问题
        Log.i(TAG, "===== 开始设置类别过滤 =====");
        Log.i(TAG, "总选中类别数: " + selectedCount);
        Log.i(TAG, "第一个类别ID: " + (selectedCount > 0 ? enabledClassIds[0] : "无"));
        Log.i(TAG, "最后一个类别ID: " + (selectedCount > 0 ? enabledClassIds[selectedCount-1] : "无"));
        
        // 启用类别过滤功能
        boolean useFilter = true; // 设置为true启用过滤
        
        if (useFilter) {
            // Apply filter
            boolean success = yolov8ncnn.setEnabledClassIds(enabledClassIds);
            Log.i(TAG, "应用类别过滤结果: " + (success ? "成功" : "失败"));
            
            if (success) {
                Toast.makeText(this, "已设置过滤，仅显示选中的 " + selectedCount + " 个类别", Toast.LENGTH_SHORT).show();
            } else {
                Log.e(TAG, "Failed to set class filter");
                Toast.makeText(this, "设置类别过滤失败", Toast.LENGTH_SHORT).show();
            }
        } else {
            // 临时禁用过滤
            Log.i(TAG, "临时禁用类别过滤，显示所有检测结果");
            yolov8ncnn.clearClassFilter();
            Toast.makeText(this, "已禁用类别过滤，将显示所有检测结果", Toast.LENGTH_SHORT).show();
        }
    }

    // 将中文名称传递给本地代码以提高检测显示效果
    private void passChineseNamesToNative() {
        try {
            int datasetType = yolov8ncnn.getDatasetType();
            String[] classNames;
            
            // 获取当前数据集的英文类名
            if (datasetType == Yolov8Ncnn.DATASET_OIV) {
                classNames = OIV_CLASS_NAMES;
                Log.d(TAG, "Using OIV class names for Chinese mapping");
            } else {
                classNames = COCO_CLASS_NAMES;
                Log.d(TAG, "Using COCO class names for Chinese mapping");
            }
            
            if (classNames == null || classNames.length == 0) {
                Log.e(TAG, "Cannot pass Chinese names: class names array is null or empty");
                return;
            }
            
            // 为每个英文类名查找对应的中文名称
            String[] chineseNames = new String[classNames.length];
            for (int i = 0; i < classNames.length; i++) {
                String englishName = classNames[i];
                if (englishName == null) {
                    Log.w(TAG, "Null English class name at index " + i);
                    chineseNames[i] = "";
                    continue;
                }
                
                String chineseName = getChineseNameForClass(englishName, datasetType);
                chineseNames[i] = chineseName;
                Log.d(TAG, "Chinese mapping for index " + i + ": " + englishName + " -> " + chineseName);
            }
            
            // 传递中文名称数组到本地代码
            boolean result = yolov8ncnn.setChineseClassNames(chineseNames);
            Log.i(TAG, "Passed " + chineseNames.length + " Chinese class names to native code, result: " + result);
        } catch (Exception e) {
            Log.e(TAG, "Error passing Chinese names to native code: " + e.getMessage(), e);
        }
    }

    // Update showStyleParametersDialog
    private void showStyleParametersDialog(int styleId) {
        // Reload style parameters to ensure we have the latest values
        StyleParameters existingParams = styleParametersMap.get(styleId);
        if (existingParams == null) {
            // Create a new one if not found
            existingParams = new StyleParameters(styleId);
            existingParams.loadFromPreferences(this);
            styleParametersMap.put(styleId, existingParams);
        } else {
            // Reload from preferences to ensure latest values
            existingParams.loadFromPreferences(this);
        }
        
        // Use a local reference to the parameters to ensure we're modifying the right object
        final StyleParameters params = existingParams;
        
        if (params == null) {
            Log.e(TAG, "No style parameters found for style ID " + styleId);
            return;
        }
        
        // Create dialog
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        View dialogView = getLayoutInflater().inflate(R.layout.style_parameters_dialog, null);
        builder.setView(dialogView);
        
        // Set dialog title
        TextView titleTextView = dialogView.findViewById(R.id.style_param_title);
        titleTextView.setText("风格 " + styleId + " 参数设置");

        Log.d(TAG, "Setting up dialog for style " + styleId + 
             " - Line thickness: " + params.getLineThickness() +
             ", Box color: " + params.getBoxColor());
        
        // Setup line thickness slider
        final Slider lineThicknessSlider = dialogView.findViewById(R.id.slider_line_thickness);
        final TextView lineThicknessText = dialogView.findViewById(R.id.text_line_thickness);
        lineThicknessSlider.setValue(params.getLineThickness());
        lineThicknessText.setText(String.valueOf(params.getLineThickness()));
        lineThicknessSlider.addOnChangeListener(new Slider.OnChangeListener() {
            @Override
            public void onValueChange(@NonNull Slider slider, float value, boolean fromUser) {
                int intValue = (int) value;
                lineThicknessText.setText(String.valueOf(intValue));
                params.setLineThickness(intValue);
            }
        });
        
        // Setup line type radio group
        RadioGroup lineTypeGroup = dialogView.findViewById(R.id.radio_group_line_type);
        if (params.getLineType() == StyleParameters.LINE_TYPE_SOLID) {
            lineTypeGroup.check(R.id.radio_line_solid);
        } else if (params.getLineType() == StyleParameters.LINE_TYPE_DASHED) {
            lineTypeGroup.check(R.id.radio_line_dashed);
        } else if (params.getLineType() == StyleParameters.LINE_TYPE_CORNER) {
            lineTypeGroup.check(R.id.radio_line_corner);
        }
        
        lineTypeGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                if (checkedId == R.id.radio_line_solid) {
                    params.setLineType(StyleParameters.LINE_TYPE_SOLID);
                } else if (checkedId == R.id.radio_line_dashed) {
                    params.setLineType(StyleParameters.LINE_TYPE_DASHED);
                } else if (checkedId == R.id.radio_line_corner) {
                    params.setLineType(StyleParameters.LINE_TYPE_CORNER);
                }
            }
        });
        
        // Setup box alpha slider
        final Slider boxAlphaSlider = dialogView.findViewById(R.id.slider_box_alpha);
        final TextView boxAlphaText = dialogView.findViewById(R.id.text_box_alpha);
        boxAlphaSlider.setValue(params.getBoxAlpha());
        boxAlphaText.setText(String.format("%.1f", params.getBoxAlpha()));
        boxAlphaSlider.addOnChangeListener(new Slider.OnChangeListener() {
            @Override
            public void onValueChange(@NonNull Slider slider, float value, boolean fromUser) {
                boxAlphaText.setText(String.format("%.1f", value));
                params.setBoxAlpha(value);
            }
        });
        
        // Setup font size slider
        final Slider fontSizeSlider = dialogView.findViewById(R.id.slider_font_size);
        final TextView fontSizeText = dialogView.findViewById(R.id.text_font_size);
        fontSizeSlider.setValue(params.getFontSize());
        fontSizeText.setText(String.format("%.2f", params.getFontSize()));
        fontSizeSlider.addOnChangeListener(new Slider.OnChangeListener() {
            @Override
            public void onValueChange(@NonNull Slider slider, float value, boolean fromUser) {
                fontSizeText.setText(String.format("%.2f", value));
                params.setFontSize(value);
            }
        });
        
        // Setup text style radio group
        RadioGroup textStyleGroup = dialogView.findViewById(R.id.radio_group_text_style);
        switch (params.getTextStyle()) {
            case StyleParameters.TEXT_STYLE_TECH:
                textStyleGroup.check(R.id.radio_text_style_tech);
                break;
            case StyleParameters.TEXT_STYLE_FUTURE:
                textStyleGroup.check(R.id.radio_text_style_future);
                break;
            case StyleParameters.TEXT_STYLE_NEON:
                textStyleGroup.check(R.id.radio_text_style_neon);
                break;
            case StyleParameters.TEXT_STYLE_MILITARY:
                textStyleGroup.check(R.id.radio_text_style_military);
                break;
            case StyleParameters.TEXT_STYLE_MINIMAL:
                textStyleGroup.check(R.id.radio_text_style_minimal);
                break;
        }
        
        textStyleGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                if (checkedId == R.id.radio_text_style_tech) {
                    params.setTextStyle(StyleParameters.TEXT_STYLE_TECH);
                } else if (checkedId == R.id.radio_text_style_future) {
                    params.setTextStyle(StyleParameters.TEXT_STYLE_FUTURE);
                } else if (checkedId == R.id.radio_text_style_neon) {
                    params.setTextStyle(StyleParameters.TEXT_STYLE_NEON);
                } else if (checkedId == R.id.radio_text_style_military) {
                    params.setTextStyle(StyleParameters.TEXT_STYLE_MILITARY);
                } else if (checkedId == R.id.radio_text_style_minimal) {
                    params.setTextStyle(StyleParameters.TEXT_STYLE_MINIMAL);
                }
            }
        });
        
        // Setup mask alpha slider
        final Slider maskAlphaSlider = dialogView.findViewById(R.id.slider_mask_alpha);
        final TextView maskAlphaText = dialogView.findViewById(R.id.text_mask_alpha);
        maskAlphaSlider.setValue(params.getMaskAlpha());
        maskAlphaText.setText(String.format("%.1f", params.getMaskAlpha()));
        maskAlphaSlider.addOnChangeListener(new Slider.OnChangeListener() {
            @Override
            public void onValueChange(@NonNull Slider slider, float value, boolean fromUser) {
                maskAlphaText.setText(String.format("%.1f", value));
                params.setMaskAlpha(value);
            }
        });
        
        // Setup mask contrast slider
        final Slider maskContrastSlider = dialogView.findViewById(R.id.slider_mask_contrast);
        final TextView maskContrastText = dialogView.findViewById(R.id.text_mask_contrast);
        maskContrastSlider.setValue(params.getMaskContrast());
        maskContrastText.setText(String.format("%.1f", params.getMaskContrast()));
        maskContrastSlider.addOnChangeListener(new Slider.OnChangeListener() {
            @Override
            public void onValueChange(@NonNull Slider slider, float value, boolean fromUser) {
                maskContrastText.setText(String.format("%.1f", value));
                params.setMaskContrast(value);
            }
        });
        
        // Setup mask edge thickness slider
        final Slider maskEdgeThicknessSlider = dialogView.findViewById(R.id.slider_mask_edge_thickness);
        final TextView maskEdgeThicknessText = dialogView.findViewById(R.id.text_mask_edge_thickness);
        maskEdgeThicknessSlider.setValue(params.getMaskEdgeThickness());
        maskEdgeThicknessText.setText(String.valueOf(params.getMaskEdgeThickness()));
        maskEdgeThicknessSlider.addOnChangeListener(new Slider.OnChangeListener() {
            @Override
            public void onValueChange(@NonNull Slider slider, float value, boolean fromUser) {
                int intValue = (int) value;
                maskEdgeThicknessText.setText(String.valueOf(intValue));
                params.setMaskEdgeThickness(intValue);
            }
        });
        
        // Setup mask edge type radio group
        final RadioGroup maskEdgeTypeGroup = dialogView.findViewById(R.id.radio_group_mask_edge_type);
        if (params.getMaskEdgeType() == StyleParameters.MASK_EDGE_TYPE_SOLID) {
            maskEdgeTypeGroup.check(R.id.radio_mask_edge_solid);
        } else if (params.getMaskEdgeType() == StyleParameters.MASK_EDGE_TYPE_DASHED) {
            maskEdgeTypeGroup.check(R.id.radio_mask_edge_dashed);
        } else if (params.getMaskEdgeType() == StyleParameters.MASK_EDGE_TYPE_DOTTED) {
            maskEdgeTypeGroup.check(R.id.radio_mask_edge_dotted);
        }
        
        maskEdgeTypeGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                if (checkedId == R.id.radio_mask_edge_solid) {
                    params.setMaskEdgeType(StyleParameters.MASK_EDGE_TYPE_SOLID);
                } else if (checkedId == R.id.radio_mask_edge_dashed) {
                    params.setMaskEdgeType(StyleParameters.MASK_EDGE_TYPE_DASHED);
                } else if (checkedId == R.id.radio_mask_edge_dotted) {
                    params.setMaskEdgeType(StyleParameters.MASK_EDGE_TYPE_DOTTED);
                }
            }
        });
        
        // Setup mask edge color button
        final Button maskEdgeColorButton = dialogView.findViewById(R.id.button_mask_edge_color);
        maskEdgeColorButton.setBackgroundColor(params.getMaskEdgeColor());
        maskEdgeColorButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Implement color picker logic here
                // For now, just cycle through some predefined colors
                int[] colors = {
                    0xFFFF0000, // Red
                    0xFF00FF00, // Green
                    0xFF0000FF, // Blue
                    0xFFFFFF00, // Yellow
                    0xFF00FFFF, // Cyan
                    0xFFFF00FF, // Magenta
                    0xFFFFFFFF, // White
                    0xFF000000  // Black
                };
                
                int currentColor = params.getMaskEdgeColor();
                int nextColorIndex = 0;
                
                // Find the current color in the array or choose the first one
                for (int i = 0; i < colors.length; i++) {
                    if (colors[i] == currentColor) {
                        nextColorIndex = (i + 1) % colors.length;
                        break;
                    }
                }
                
                // Set the new color and update the button background
                params.setMaskEdgeColor(colors[nextColorIndex]);
                v.setBackgroundColor(colors[nextColorIndex]);
                
                // Show a toast with the color name
                String[] colorNames = {"Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White", "Black"};
                Toast.makeText(MainActivity.this, "Mask edge color: " + colorNames[nextColorIndex], Toast.LENGTH_SHORT).show();
            }
        });
        
        // Setup reset and apply buttons
        Button resetButton = dialogView.findViewById(R.id.button_reset);
        resetButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // 只在本地重置参数，不直接调用native方法
                params.resetToDefaults();
                updateStyleParametersUI();
                Toast.makeText(MainActivity.this, "已重置设置，点击应用使其生效", Toast.LENGTH_SHORT).show();
            }
            
            // 更新UI以反映重置的值
            private void updateStyleParametersUI() {
                lineThicknessSlider.setValue(params.getLineThickness());
                lineThicknessText.setText(String.valueOf(params.getLineThickness()));
                
                if (params.getLineType() == StyleParameters.LINE_TYPE_SOLID) {
                    lineTypeGroup.check(R.id.radio_line_solid);
                } else if (params.getLineType() == StyleParameters.LINE_TYPE_DASHED) {
                    lineTypeGroup.check(R.id.radio_line_dashed);
                } else if (params.getLineType() == StyleParameters.LINE_TYPE_CORNER) {
                    lineTypeGroup.check(R.id.radio_line_corner);
                }
                
                boxAlphaSlider.setValue(params.getBoxAlpha());
                boxAlphaText.setText(String.format("%.1f", params.getBoxAlpha()));
                
                fontSizeSlider.setValue(params.getFontSize());
                fontSizeText.setText(String.format("%.2f", params.getFontSize()));
                
                // 更新文字风格单选按钮
                switch (params.getTextStyle()) {
                    case StyleParameters.TEXT_STYLE_TECH:
                        textStyleGroup.check(R.id.radio_text_style_tech);
                        break;
                    case StyleParameters.TEXT_STYLE_FUTURE:
                        textStyleGroup.check(R.id.radio_text_style_future);
                        break;
                    case StyleParameters.TEXT_STYLE_NEON:
                        textStyleGroup.check(R.id.radio_text_style_neon);
                        break;
                    case StyleParameters.TEXT_STYLE_MILITARY:
                        textStyleGroup.check(R.id.radio_text_style_military);
                        break;
                    case StyleParameters.TEXT_STYLE_MINIMAL:
                        textStyleGroup.check(R.id.radio_text_style_minimal);
                        break;
                }
                
                maskAlphaSlider.setValue(params.getMaskAlpha());
                maskAlphaText.setText(String.format("%.1f", params.getMaskAlpha()));
                
                maskContrastSlider.setValue(params.getMaskContrast());
                maskContrastText.setText(String.format("%.1f", params.getMaskContrast()));
                
                // Update mask edge thickness
                maskEdgeThicknessSlider.setValue(params.getMaskEdgeThickness());
                maskEdgeThicknessText.setText(String.valueOf(params.getMaskEdgeThickness()));
                
                // Update mask edge type
                if (params.getMaskEdgeType() == StyleParameters.MASK_EDGE_TYPE_SOLID) {
                    maskEdgeTypeGroup.check(R.id.radio_mask_edge_solid);
                } else if (params.getMaskEdgeType() == StyleParameters.MASK_EDGE_TYPE_DASHED) {
                    maskEdgeTypeGroup.check(R.id.radio_mask_edge_dashed);
                } else if (params.getMaskEdgeType() == StyleParameters.MASK_EDGE_TYPE_DOTTED) {
                    maskEdgeTypeGroup.check(R.id.radio_mask_edge_dotted);
                }
                
                // Update mask edge color
                maskEdgeColorButton.setBackgroundColor(params.getMaskEdgeColor());
            }
        });
        
        Button applyButton = dialogView.findViewById(R.id.button_apply);
        
        final AlertDialog dialog = builder.create();
        
        applyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Save parameters and apply to model
                params.saveToPreferences(MainActivity.this);
                applyStyleParameters(params);
                dialog.dismiss();
            }
        });
        
        dialog.show();
    }
    
    // Apply style parameters to the native model
    private void applyStyleParameters(StyleParameters params) {
        if (yolov8ncnn != null) {
            try {
                // First send parameters to native code using the new method with mask edge support
                boolean result = yolov8ncnn.setStyleParametersWithMaskEdge(
                        params.getStyleId(),
                        params.getLineThickness(),
                        params.getLineType(),
                        params.getBoxAlpha(),
                        params.getBoxColor(),
                        params.getFontSize(),
                        params.getFontType(),
                        params.getTextColor(),
                        params.getTextStyle(),
                        params.isFullTextBackground(),
                        params.getMaskAlpha(),
                        params.getMaskContrast(),
                        params.getMaskEdgeThickness(),
                        params.getMaskEdgeType(),
                        params.getMaskEdgeColor()
                );
                
                if (result) {
                    Log.i(TAG, "Style parameters applied successfully - Style: " + params.getStyleId());
                    Toast.makeText(this, "风格参数已应用", Toast.LENGTH_SHORT).show();
                    // Apply the current detection style again to ensure rendering updates
                    safeSetDetectionStyle(current_detection_style);
                } else {
                    Log.e(TAG, "Failed to apply style parameters for style: " + params.getStyleId());
                    Toast.makeText(this, "风格参数应用失败", Toast.LENGTH_SHORT).show();
                }
            } catch (UnsatisfiedLinkError e) {
                // Native method not implemented
                Log.e(TAG, "Native setStyleParametersWithMaskEdge method not implemented", e);
                Toast.makeText(this, "风格参数功能尚未实现，参数已保存但未应用", Toast.LENGTH_LONG).show();
                
                // Still save the parameters for future use when the native implementation is ready
                params.saveToPreferences(this);
            }
        }
    }

    // Add a helper method to safely call setDetectionStyle
    private void safeSetDetectionStyle(int style) {
        try {
            yolov8ncnn.setDetectionStyle(style);
        } catch (UnsatisfiedLinkError e) {
            // Native method not implemented
            Log.e(TAG, "Native setDetectionStyle method not implemented", e);
            Toast.makeText(this, "风格设置功能尚未实现", Toast.LENGTH_SHORT).show();
        }
    }

    // Add reloadStyleParameters method after the onCreate method
    private void reloadStyleParameters() {
        // Initialize style parameters for all styles
        for (int i = 0; i <= 9; i++) {
            StyleParameters params = new StyleParameters(i);
            params.loadFromPreferences(this);
            styleParametersMap.put(i, params);
        }
        Log.i(TAG, "Reloaded style parameters from preferences");
    }
}
