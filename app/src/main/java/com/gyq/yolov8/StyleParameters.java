package com.gyq.yolov8;

import android.content.Context;
import android.content.SharedPreferences;
import android.preference.PreferenceManager;
import android.graphics.Color;

/**
 * Class for managing detection style parameters
 */
public class StyleParameters {
    // Line parameters
    private int lineThickness;
    private int lineType; // 0=solid, 1=dashed, 2=corner
    
    // Box parameters
    private float boxAlpha;
    private int boxColor;
    
    // Text parameters
    private float fontSize;
    private int fontType; // 0=default, 1=monospace, 2=serif
    private int textColor;
    private int textStyle; // 新增：文字风格预设
    private boolean fullTextBackground; // 新增：文字背景完全覆盖
    
    // Mask parameters
    private float maskAlpha;
    private float maskContrast;
    // 新增：掩码边缘参数
    private int maskEdgeThickness;
    private int maskEdgeType; // 0=solid, 1=dashed, 2=dotted
    private int maskEdgeColor;
    
    // Style ID this parameter set belongs to
    private int styleId;
    
    // Default values
    public static final int LINE_TYPE_SOLID = 0;
    public static final int LINE_TYPE_DASHED = 1;
    public static final int LINE_TYPE_CORNER = 2;
    
    public static final int FONT_TYPE_DEFAULT = 0;
    public static final int FONT_TYPE_MONOSPACE = 1;
    public static final int FONT_TYPE_SERIF = 2;
    
    // 新增掩码边缘线型常量
    public static final int MASK_EDGE_TYPE_SOLID = 0;
    public static final int MASK_EDGE_TYPE_DASHED = 1;
    public static final int MASK_EDGE_TYPE_DOTTED = 2;
    
    // 新增文字风格常量
    public static final int TEXT_STYLE_TECH = 0;      // 科技风格 - 蓝底白字
    public static final int TEXT_STYLE_FUTURE = 1;    // 未来风格 - 深蓝渐变底色
    public static final int TEXT_STYLE_NEON = 2;      // 霓虹风格 - 黑底亮色描边
    public static final int TEXT_STYLE_MILITARY = 3;  // 军事风格 - 绿底黄字
    public static final int TEXT_STYLE_MINIMAL = 4;   // 简约风格 - 半透明底色
    
    // Constructor with default values
    public StyleParameters(int styleId) {
        this.styleId = styleId;
        // First reset to base defaults
        resetToDefaults();
    }
    
    // Reset parameters to defaults for the current style
    public void resetToDefaults() {
        // Set base defaults first
        lineThickness = 2;
        lineType = LINE_TYPE_SOLID;
        boxAlpha = 1.0f;
        fontSize = 0.65f;
        fontType = FONT_TYPE_DEFAULT;
        textColor = 0xFFFFFFFF; // White
        textStyle = TEXT_STYLE_TECH; // 默认科技风格
        fullTextBackground = true; // 默认完全覆盖文字区域
        maskAlpha = 0.5f;
        maskContrast = 1.0f;
        // 掩码边缘默认值
        maskEdgeThickness = 1;
        maskEdgeType = MASK_EDGE_TYPE_SOLID;
        maskEdgeColor = 0xFFFFFFFF; // White
        
        // Set default colors based on style ID
        switch (styleId) {
            case 0: // Cyber Grid
                boxColor = 0xFF00FF00; // Green
                textStyle = TEXT_STYLE_TECH;
                maskEdgeColor = 0xFF00FF00; // Match box color
                break;
            case 1: // Holographic
                lineThickness = 3;
                lineType = LINE_TYPE_SOLID;
                boxAlpha = 0.8f;
                boxColor = 0xFF00FFFF; // Cyan
                textStyle = TEXT_STYLE_FUTURE;
                maskEdgeColor = 0xFF00FFFF; // Match box color
                maskEdgeThickness = 2;
                break;
            case 2: // Pulse
                lineType = LINE_TYPE_DASHED;
                boxAlpha = 0.9f;
                boxColor = 0xFFFF00FF; // Magenta
                textStyle = TEXT_STYLE_NEON;
                maskEdgeColor = 0xFFFF00FF; // Match box color
                maskEdgeType = MASK_EDGE_TYPE_DASHED;
                break;
            case 3: // Plasma
                lineThickness = 2;
                lineType = LINE_TYPE_CORNER;
                maskAlpha = 0.6f;
                boxColor = 0xFFFF5500; // Orange
                textStyle = TEXT_STYLE_FUTURE;
                maskEdgeColor = 0xFFFF5500; // Match box color
                maskEdgeType = MASK_EDGE_TYPE_DOTTED;
                break;
            case 4: // Data
                fontSize = 0.7f;
                maskContrast = 1.2f;
                boxColor = 0xFF5555FF; // Blue
                textStyle = TEXT_STYLE_TECH;
                maskEdgeColor = 0xFF5555FF; // Match box color
                maskEdgeThickness = 2;
                break;
            case 5: // Target
                lineThickness = 1;
                lineType = LINE_TYPE_CORNER;
                fontSize = 0.6f;
                boxColor = 0xFFFF0000; // Red
                textStyle = TEXT_STYLE_MILITARY;
                maskEdgeColor = 0xFFFF0000; // Match box color
                break;
            case 6: // Classic
                boxColor = 0xFFFFFF00; // Yellow
                textStyle = TEXT_STYLE_MINIMAL;
                maskEdgeColor = 0xFFFFFF00; // Match box color
                break;
            case 7: // Neon
                lineThickness = 3;
                boxAlpha = 0.8f;
                maskAlpha = 0.7f;
                boxColor = 0xFF00FF99; // Neon Green
                textStyle = TEXT_STYLE_NEON;
                maskEdgeColor = 0xFF00FF99; // Match box color
                maskEdgeThickness = 2;
                maskEdgeType = MASK_EDGE_TYPE_DASHED;
                break;
            case 8: // Digital
                lineType = LINE_TYPE_DASHED;
                fontSize = 0.55f;
                boxColor = 0xFF99FFFF; // Light Cyan
                textStyle = TEXT_STYLE_TECH;
                maskEdgeColor = 0xFF99FFFF; // Match box color
                maskEdgeType = MASK_EDGE_TYPE_DOTTED;
                break;
            case 9: // Quantum
                lineThickness = 2;
                lineType = LINE_TYPE_CORNER;
                boxColor = 0xFFAA00FF; // Purple
                textStyle = TEXT_STYLE_FUTURE;
                maskEdgeColor = 0xFFAA00FF; // Match box color
                maskEdgeThickness = 2;
                break;
            default:
                boxColor = 0xFF00FF00; // Default Green
                maskEdgeColor = 0xFF00FF00; // Match box color
                break;
        }
        
        // 根据文字风格设置相应文字颜色
        updateTextColorByStyle();
    }
    
    // 根据文字风格更新文字颜色
    private void updateTextColorByStyle() {
        switch (textStyle) {
            case TEXT_STYLE_TECH:
                textColor = Color.WHITE;
                break;
            case TEXT_STYLE_FUTURE:
                textColor = Color.CYAN;
                break;
            case TEXT_STYLE_NEON:
                textColor = 0xFF00FF99; // 亮绿色
                break;
            case TEXT_STYLE_MILITARY:
                textColor = 0xFFFFFF00; // 黄色
                break;
            case TEXT_STYLE_MINIMAL:
                textColor = Color.WHITE;
                break;
            default:
                textColor = Color.WHITE;
                break;
        }
    }
    
    // Save parameters to SharedPreferences
    public void saveToPreferences(Context context) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        SharedPreferences.Editor editor = prefs.edit();
        
        String prefix = "style_" + styleId + "_";
        
        editor.putInt(prefix + "line_thickness", lineThickness);
        editor.putInt(prefix + "line_type", lineType);
        editor.putFloat(prefix + "box_alpha", boxAlpha);
        editor.putInt(prefix + "box_color", boxColor);
        editor.putFloat(prefix + "font_size", fontSize);
        editor.putInt(prefix + "font_type", fontType);
        editor.putInt(prefix + "text_color", textColor);
        editor.putInt(prefix + "text_style", textStyle);
        editor.putBoolean(prefix + "full_text_background", fullTextBackground);
        editor.putFloat(prefix + "mask_alpha", maskAlpha);
        editor.putFloat(prefix + "mask_contrast", maskContrast);
        // 保存掩码边缘参数
        editor.putInt(prefix + "mask_edge_thickness", maskEdgeThickness);
        editor.putInt(prefix + "mask_edge_type", maskEdgeType);
        editor.putInt(prefix + "mask_edge_color", maskEdgeColor);
        
        editor.apply();
    }
    
    // Load parameters from SharedPreferences
    public void loadFromPreferences(Context context) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        
        String prefix = "style_" + styleId + "_";
        
        // If no saved preferences, keep defaults
        if (!prefs.contains(prefix + "line_thickness")) {
            return;
        }
        
        lineThickness = prefs.getInt(prefix + "line_thickness", lineThickness);
        lineType = prefs.getInt(prefix + "line_type", lineType);
        boxAlpha = prefs.getFloat(prefix + "box_alpha", boxAlpha);
        boxColor = prefs.getInt(prefix + "box_color", boxColor);
        fontSize = prefs.getFloat(prefix + "font_size", fontSize);
        fontType = prefs.getInt(prefix + "font_type", fontType);
        textColor = prefs.getInt(prefix + "text_color", textColor);
        textStyle = prefs.getInt(prefix + "text_style", textStyle);
        fullTextBackground = prefs.getBoolean(prefix + "full_text_background", fullTextBackground);
        maskAlpha = prefs.getFloat(prefix + "mask_alpha", maskAlpha);
        maskContrast = prefs.getFloat(prefix + "mask_contrast", maskContrast);
        // 加载掩码边缘参数
        maskEdgeThickness = prefs.getInt(prefix + "mask_edge_thickness", maskEdgeThickness);
        maskEdgeType = prefs.getInt(prefix + "mask_edge_type", maskEdgeType);
        maskEdgeColor = prefs.getInt(prefix + "mask_edge_color", maskEdgeColor);
    }
    
    // Convert parameters to native C++ format
    public String toNativeFormat() {
        return String.format("%d,%d,%.2f,%d,%.2f,%d,%d,%d,%b,%.2f,%.2f,%d,%d,%d",
                lineThickness, lineType, boxAlpha, boxColor,
                fontSize, fontType, textColor, textStyle, fullTextBackground,
                maskAlpha, maskContrast,
                maskEdgeThickness, maskEdgeType, maskEdgeColor);
    }

    // Getters and setters
    public int getLineThickness() {
        return lineThickness;
    }

    public void setLineThickness(int lineThickness) {
        this.lineThickness = lineThickness;
    }

    public int getLineType() {
        return lineType;
    }

    public void setLineType(int lineType) {
        this.lineType = lineType;
    }

    public float getBoxAlpha() {
        return boxAlpha;
    }

    public void setBoxAlpha(float boxAlpha) {
        this.boxAlpha = boxAlpha;
    }

    public int getBoxColor() {
        return boxColor;
    }

    public void setBoxColor(int boxColor) {
        this.boxColor = boxColor;
    }

    public float getFontSize() {
        return fontSize;
    }

    public void setFontSize(float fontSize) {
        this.fontSize = fontSize;
    }

    public int getFontType() {
        return fontType;
    }

    public void setFontType(int fontType) {
        this.fontType = fontType;
    }

    public int getTextColor() {
        return textColor;
    }

    public void setTextColor(int textColor) {
        this.textColor = textColor;
    }
    
    public int getTextStyle() {
        return textStyle;
    }
    
    public void setTextStyle(int textStyle) {
        this.textStyle = textStyle;
        // 更新文字颜色
        updateTextColorByStyle();
    }
    
    public boolean isFullTextBackground() {
        return fullTextBackground;
    }
    
    public void setFullTextBackground(boolean fullTextBackground) {
        this.fullTextBackground = fullTextBackground;
    }

    public float getMaskAlpha() {
        return maskAlpha;
    }

    public void setMaskAlpha(float maskAlpha) {
        this.maskAlpha = maskAlpha;
    }

    public float getMaskContrast() {
        return maskContrast;
    }

    public void setMaskContrast(float maskContrast) {
        this.maskContrast = maskContrast;
    }
    
    // 掩码边缘参数的getter和setter
    public int getMaskEdgeThickness() {
        return maskEdgeThickness;
    }
    
    public void setMaskEdgeThickness(int maskEdgeThickness) {
        this.maskEdgeThickness = maskEdgeThickness;
    }
    
    public int getMaskEdgeType() {
        return maskEdgeType;
    }
    
    public void setMaskEdgeType(int maskEdgeType) {
        this.maskEdgeType = maskEdgeType;
    }
    
    public int getMaskEdgeColor() {
        return maskEdgeColor;
    }
    
    public void setMaskEdgeColor(int maskEdgeColor) {
        this.maskEdgeColor = maskEdgeColor;
    }

    public int getStyleId() {
        return styleId;
    }
} 