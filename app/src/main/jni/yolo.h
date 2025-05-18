#pragma once

#include <opencv2/core/core.hpp>
#include "Object.h"
#include <net.h>
#include "TrackingParams.h"
#include <set>
#include <mutex>

// 添加数据集类型枚举
enum DatasetType {
    DATASET_COCO = 0,
    DATASET_OIV = 1
};

// 当前使用的数据集类型(默认为COCO)
static DatasetType current_dataset_type = DATASET_COCO;

// COCO数据集类别名称
static const char *coco_class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
        "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear",
        "hair drier", "toothbrush"
};

// OIV数据集类别名称 (Open Images V7完整版)
static const char *oiv_class_names[] = {
    "Accordion", "Adhesive tape", "Aircraft", "Alarm clock", "Alpaca", "Ambulance", 
    "Animal", "Ant", "Antelope", "Apple", "Armadillo", "Artichoke", "Auto part", "Axe", 
    "Backpack", "Bagel", "Baked goods", "Balance beam", "Ball (Object)", "Balloon", "Banana", 
    "Band-aid", "Banjo", "Barge", "Barrel", "Baseball bat", "Baseball glove", "Bat (Animal)", 
    "Bathroom accessory", "Bathroom cabinet", "Bathtub", "Beaker", "Bear", "Beard", "Bed", 
    "Bee", "Beehive", "Beer", "Beetle", "Bell pepper", "Belt", "Bench", "Bicycle", 
    "Bicycle helmet", "Bicycle wheel", "Bidet", "Billboard", "Billiard table", "Binoculars", 
    "Bird", "Blender", "Blue jay", "Boat", "Bomb", "Book", "Bookcase", "Boot", "Bottle", 
    "Bottle opener", "Bow and arrow", "Bowl", "Bowling equipment", "Box", "Boy", "Brassiere", 
    "Bread", "Briefcase", "Broccoli", "Bronze sculpture", "Brown bear", "Building", "Bull", 
    "Burrito", "Bus", "Bust", "Butterfly", "Cabbage", "Cabinetry", "Cake", "Cake stand", 
    "Calculator", "Camel", "Camera", "Can opener", "Canary", "Candle", "Candy", "Cannon", 
    "Canoe", "Cantaloupe", "Car", "Carnivore", "Carrot", "Cart", "Cassette deck", "Castle", 
    "Cat", "Cat furniture", "Caterpillar", "Cattle", "Ceiling fan", "Cello", "Centipede", 
    "Chainsaw", "Chair", "Cheese", "Cheetah", "Chest of drawers", "Chicken", "Chime", 
    "Chisel", "Chopsticks", "Christmas tree", "Clock", "Closet", "Clothing", "Coat", "Cocktail", 
    "Cocktail shaker", "Coconut", "Coffee (drink)", "Coffee cup", "Coffee table", "Coffeemaker", 
    "Coin", "Common fig", "Common sunflower", "Computer keyboard", "Computer monitor", 
    "Computer mouse", "Container", "Convenience store", "Cookie", "Cooking spray", "Corded phone", 
    "Cosmetics", "Couch", "Countertop", "Cowboy hat", "Crab", "Cream", "Cricket ball", 
    "Crocodile", "Croissant", "Crown", "Crutch", "Cucumber", "Cupboard", "Curtain", 
    "Cutting board", "Dagger", "Dairy Product", "Deer", "Desk", "Dessert", "Diaper", "Dice", 
    "Digital clock", "Dinosaur", "Dishwasher", "Dog", "Dog bed", "Doll", "Dolphin", "Door", 
    "Door handle", "Doughnut", "Dragonfly", "Drawer", "Dress", "Drill (Tool)", "Drink", 
    "Drinking straw", "Drum", "Duck", "Dumbbell", "Eagle", "Earring", "Egg", "Elephant", 
    "Envelope", "Eraser", "Face powder", "Facial tissue holder", "Falcon", "Fashion accessory", 
    "Fast food", "Fax", "Fedora", "Filing cabinet", "Fire hydrant", "Fireplace", "Fish", 
    "Fixed-wing aircraft", "Flag", "Flashlight", "Flower", "Flowerpot", "Flute", "Flying disc", 
    "Food", "Food processor", "Football", "Football helmet", "Footwear", "Fork", "Fountain", 
    "Fox", "French fries", "French horn", "Frog", "Fruit", "Frying pan", "Furniture", 
    "Garden Asparagus", "Gas stove", "Glasses", "Giraffe", "Girl" "Glove", "Goat", "Goggles", 
    "Goldfish", "Golf ball", "Golf cart", "Gondola", "Goose", "Grape", "Grapefruit", "Grinder", 
    "Guacamole", "Guitar", "Hair dryer", "Hair spray", "Hamburger", "Hammer", "Hamster", 
    "Hand dryer", "Handbag", "Handgun", "Harbor seal", "Harmonica", "Harp", "Harpsichord", 
    "Hat", "Headphones", "Heater", "Hedgehog", "Helicopter", "Helmet", "High heels", 
    "Hiking equipment", "Hippopotamus", "Home appliance", "Honeycomb", "Horizontal bar", 
    "Horse", "Hot dog", "House", "Houseplant", "Human arm", "Human body", "Human ear", 
    "Human eye", "Human face", "Human foot", "Human hair", "Human hand", "Human head", 
    "Human leg", "Human mouth", "Human nose", "Humidifier", "Ice cream", "Indoor rower", 
    "Infant bed", "Insect", "Invertebrate", "Ipod", "Isopod", "Jacket", "Jacuzzi", 
    "Jaguar (Animal)", "Jeans", "Jellyfish", "Jet ski", "Jug", "Juice", "Kangaroo", "Kettle", 
    "Kitchen & dining room table", "Kitchen appliance", "Kitchen knife", "Kitchen utensil", 
    "Kitchenware", "Kite", "Knife", "Koala", "Ladder", "Ladle", "Ladybug", "Lamp", 
    "Land vehicle", "Lantern", "Laptop", "Lavender (Plant)", "Lemon (plant)", "Leopard", 
    "Light bulb", "Light switch", "Lighthouse", "Lily", "Limousine", "Lion", "Lipstick", 
    "Lizard", "Lobster", "Loveseat", "Luggage and bags", "Lynx", "Magpie", "Mammal", "Man", 
    "Mango", "Maple", "Maraca", "Marine invertebrates", "Marine mammal", "Measuring cup", 
    "Mechanical fan", "Medical equipment", "Microphone", "Microwave oven", "Milk", "Miniskirt", 
    "Mirror", "Missile", "Mixer", "Mixing bowl", "Mobile phone", "Monkey", "Moths and butterflies", 
    "Motorcycle", "Mouse", "Muffin", "Mug", "Mule", "Mushroom", "Musical instrument", 
    "Musical keyboard", "Nail (Construction)", "Necklace", "Nightstand", "Oboe", "Office building", 
    "Office supplies", "Orange (fruit)", "Organ (Musical Instrument)", "Ostrich", "Otter", "Oven", 
    "Owl", "Oyster", "Paddle", "Palm tree", "Pancake", "Panda", "Paper cutter", "Paper towel", 
    "Parachute", "Parking meter", "Parrot", "Pasta", "Pastry", "Peach", "Pear", "Pen", 
    "Pencil case", "Pencil sharpener", "Penguin", "Perfume", "Person", "Personal care", 
    "Personal flotation device", "Piano", "Picnic basket", "Picture frame", "Pig", "Pillow", 
    "Pineapple", "Pitcher (Container)", "Pizza", "Pizza cutter", "Plant", "Plastic bag", "Plate", 
    "Platter", "Plumbing fixture", "Polar bear", "Pomegranate", "Popcorn", "Porch", "Porcupine", 
    "Poster", "Potato", "Power plugs and sockets", "Pressure cooker", "Pretzel", "Printer", 
    "Pumpkin", "Punching bag", "Rabbit", "Raccoon", "Racket", "Radish", "Ratchet (Device)", 
    "Raven", "Rays and skates", "Red panda", "Refrigerator", "Remote control", "Reptile", 
    "Rhinoceros", "Rifle", "Ring binder", "Rocket", "Roller skates", "Rose", "Rugby ball", 
    "Ruler", "Salad", "Salt and pepper shakers", "Sandal", "Sandwich", "Saucer", "Saxophone", 
    "Scale", "Scarf", "Scissors", "Scoreboard", "Scorpion", "Screwdriver", "Sculpture", 
    "Sea lion", "Sea turtle", "Seafood", "Seahorse", "Seat belt", "Segway", "Serving tray", 
    "Sewing machine", "Shark", "Sheep", "Shelf", "Shellfish", "Shirt", "Shorts", "Shotgun", 
    "Shower", "Shrimp", "Sink", "Skateboard", "Ski", "Skirt", "Skull", "Skunk", "Skyscraper", 
    "Slow cooker", "Snack", "Snail", "Snake", "Snowboard", "Snowman", "Snowmobile", "Snowplow", 
    "Soap dispenser", "Sock", "Sofa bed", "Sombrero", "Sparrow", "Spatula", "Spice rack", 
    "Spider", "Spoon", "Sports equipment", "Sports uniform", "Squash (Plant)", "Squid", 
    "Squirrel", "Stairs", "Stapler", "Starfish", "Stationary bicycle", "Stethoscope", "Stool", 
    "Stop sign", "Strawberry", "Street light", "Stretcher", "Studio couch", "Submarine", 
    "Submarine sandwich", "Suit", "Suitcase", "Sun hat", "Sunglasses", "Surfboard", "Sushi", 
    "Swan", "Swim cap", "Swimming pool", "Swimwear", "Sword", "Syringe", "Table", 
    "Table tennis racket", "Tablet computer", "Tableware", "Taco", "Tank", "Tap", "Tart", 
    "Taxi", "Tea", "Teapot", "Teddy bear", "Telephone", "Television", "Tennis ball", 
    "Tennis racket", "Tent", "Tiara", "Tick", "Tie", "Tiger", "Tin can", "Tire", "Toaster", 
    "Toilet", "Toilet paper", "Tomato", "Tool", "Toothbrush", "Torch", "Tortoise", "Towel", 
    "Tower", "Toy", "Traffic light", "Traffic sign", "Train", "Training bench", "Treadmill", 
    "Tree", "Tree house", "Tripod", "Trombone", "Trousers", "Truck", "Trumpet", "Turkey", 
    "Turtle", "Umbrella", "Unicycle", "Van", "Vase", "Vegetable", "Vehicle", 
    "Vehicle registration plate", "Violin", "Volleyball (Ball)", "Waffle", "Waffle iron", 
    "Wall clock", "Wardrobe", "Washing machine", "Waste container", "Watch", "Watercraft", 
    "Watermelon", "Weapon", "Whale", "Wheel", "Wheelchair", "Whisk", "Whiteboard", "Willow", 
    "Window", "Window blind", "Wine", "Wine glass", "Wine rack", "Winter melon", "Wok", 
    "Woman", "Wood-burning stove", "Woodpecker", "Worm", "Wrench", "Zebra", "Zucchini"
};

// 不管使用哪个数据集，都用这个指针指向当前使用的类别名称
static const char **class_names = coco_class_names;

static const unsigned char colors[19][3] = {
        {54,  67,  244},
        {99,  30,  233},
        {176, 39,  156},
        {183, 58,  103},
        {181, 81,  63},
        {243, 150, 33},
        {244, 169, 3},
        {212, 188, 0},
        {136, 150, 0},
        {80,  175, 76},
        {74,  195, 139},
        {57,  220, 205},
        {59,  235, 255},
        {7,   193, 255},
        {0,   152, 255},
        {34,  87,  255},
        {72,  85,  121},
        {158, 158, 158},
        {139, 125, 96}
};

struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};

// 模型配置结构，用于保存不同模型的特定配置
struct ModelConfig {
    std::string name;              // 模型名称
    std::string input_name;        // 输入层名称
    std::string detection_output;  // 检测输出层名称
    std::string seg_output;        // 分割输出层名称 (如果有)
    std::string coeff_output;      // 系数输出层名称 (如果有)
    int input_size;                // 输入尺寸
    bool transpose_output;         // 是否需要转置输出
    int num_classes;               // 类别数
    int num_mask_coeffs;           // 掩码系数数量 (分割模型)
    bool is_segmentation;          // 是否为分割模型
    
    // 默认构造函数，设置默认值
    ModelConfig() : input_size(640), transpose_output(false), 
                   num_classes(80), num_mask_coeffs(32), 
                   is_segmentation(false) {
        input_name = "images";
        detection_output = "output";
        seg_output = "seg";
        coeff_output = "";
    }
};

class Yolo {
public:
    Yolo();
    ~Yolo();

    int load(AAssetManager *mgr, const char *modeltype, int target_size, const float *mean_vals,
             const float *norm_vals, bool use_gpu = false);
    
    // 添加新的配置加载功能
    int loadWithConfig(AAssetManager *mgr, const char *modeltype, const ModelConfig& config, bool use_gpu = false);
    
    // 根据模型名称自动检测配置
    bool detectModelConfig(const char *modeltype, ModelConfig& outConfig);
    
    // 模型验证工具
    int validateModel(AAssetManager* mgr, const char* model_path);

    int detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold = 0.4f,
               float nms_threshold = 0.5f);

    int draw(cv::Mat &rgb, const std::vector<Object> &objects);

    void setDetectionStyle(int style);
    
    bool setStyleParameters(int styleId, int lineThickness, int lineType,
                           float boxAlpha, int boxColor, float fontSize,
                           int fontType, int textColor, int textStyle,
                           bool fullTextBackground,
                           float maskAlpha, float maskContrast,
                           int maskEdgeThickness, int maskEdgeType, int maskEdgeColor);
    
    // Get current style parameters for a given style ID
    bool getStyleParameters(int styleId, int* lineThickness, int* lineType,
                           float* boxAlpha, int* boxColor, float* fontSize,
                           int* fontType, int* textColor,
                           float* maskAlpha, float* maskContrast,
                           int* maskEdgeThickness, int* maskEdgeType, int* maskEdgeColor);
    
    // 新增：重置指定风格ID的样式参数为默认值
    bool resetStyleParameters(int styleId);
    
    void setMaskThreshold(float threshold);
    
    void setTrackingEnabled(bool enable);
    bool isTrackingEnabled() const;
    
    void setEnableMaskTracking(bool enable);
    bool isMaskTrackingEnabled() const;

    void setTrackingParams(const TrackingParams& params);

    const std::vector<const char*>& getClassNames() const;
    
    // 获取当前数据集类型
    DatasetType getDatasetType() const;
    
    // 新增：从指定的模型名称读取类别文件
    int loadClassesFromFile(AAssetManager *mgr, const char *modelName);
    
    // 新增：设置启用的类别ID
    void setEnabledClassIds(const std::set<int>& classIds);
    
    // 新增：清除类别过滤
    void clearClassFilter();
    
    // 新增：获取类别过滤状态
    bool isClassFilteringEnabled() const;
    
    // 新增：设置中文类别名称
    void setChineseClassNames(const char** names, int count);

private:
    ncnn::Net yolo;
    std::string model_type;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::UnlockedPoolAllocator workspace_pool_allocator;
    int detection_style;

    bool is_segmentation_model;
    int num_mask_coeffs;
    std::string det_output_name;
    std::string seg_output_name;
    float mask_threshold;
    ncnn::Mat mask_protos;
    
    bool enable_tracking;
    TrackingParams current_tracking_params;
    
    // 存储当前数据集类型
    DatasetType dataset_type;
    
    // 新增：存储类别筛选相关数据
    std::set<int> enabled_class_ids;
    bool class_filtering_enabled;
    
    // 新增：存储中文类别名称
    std::vector<std::string> chinese_class_names;

    // Style parameters for customizing detection visualization
    int style_line_thickness = 2;
    int style_line_type = 0; // 0=solid, 1=dashed, 2=corner brackets
    float style_box_alpha = 1.0f;
    int style_box_color = 0xFF00FF00; // Default green in ARGB format
    float style_font_size = 0.65f;
    int style_font_type = 0; // Default font
    int style_text_color = 0xFFFFFFFF; // Default white in ARGB format
    int style_text_style = 0; // 新增：文字风格类型
    bool style_full_text_background = true; // 新增：文字背景完全覆盖
    float style_mask_alpha = 0.5f;
    float style_mask_contrast = 1.0f;
    int style_mask_edge_thickness = 2; // 掩码边缘线条粗细
    int style_mask_edge_type = 0; // 0=solid, 1=dashed, 2=dotted
    int style_mask_edge_color = 0xFFFFFFFF; // 掩码边缘颜色，默认白色
    bool style_custom_parameters_set = false;
    
    // Style constants
    static const int TEXT_STYLE_TECH = 0;      // 科技风格 - 蓝底白字
    static const int TEXT_STYLE_FUTURE = 1;    // 未来风格 - 深蓝渐变底色
    static const int TEXT_STYLE_NEON = 2;      // 霓虹风格 - 黑底亮色描边
    static const int TEXT_STYLE_MILITARY = 3;  // 军事风格 - 绿底黄字
    static const int TEXT_STYLE_MINIMAL = 4;   // 简约风格 - 半透明底色
    
    // Mask edge type constants
    static const int MASK_EDGE_TYPE_SOLID = 0;   // 实线边缘
    static const int MASK_EDGE_TYPE_DASHED = 1;  // 虚线边缘
    static const int MASK_EDGE_TYPE_DOTTED = 2;  // 点状边缘

    // 存储模型配置信息
    ModelConfig model_config;

    std::mutex tracking_mutex;
    std::mutex draw_mutex;
};

