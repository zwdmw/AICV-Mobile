package ylov.colorpicker;

import android.app.Dialog;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;

/**
 * A simple color picker dialog
 */
public class ColorPickerDialog extends Dialog {
    
    public interface OnColorSelectedListener {
        void onColorSelected(int color);
    }
    
    private OnColorSelectedListener listener;
    private int currentColor;
    private ColorWheelView colorWheel;
    private SeekBar brightnessSeekBar;
    private View colorPreview;
    
    public ColorPickerDialog(Context context, int initialColor) {
        super(context);
        this.currentColor = initialColor;
    }
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Create the layout programmatically
        LinearLayout layout = new LinearLayout(getContext());
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(20, 20, 20, 20);
        
        // Add color wheel
        colorWheel = new ColorWheelView(getContext());
        colorWheel.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 
                500));
        layout.addView(colorWheel);
        
        // Add brightness control
        TextView brightnessLabel = new TextView(getContext());
        brightnessLabel.setText("亮度");
        brightnessLabel.setPadding(0, 20, 0, 10);
        layout.addView(brightnessLabel);
        
        brightnessSeekBar = new SeekBar(getContext());
        brightnessSeekBar.setMax(255);
        brightnessSeekBar.setProgress(getBrightness(currentColor));
        brightnessSeekBar.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 
                LinearLayout.LayoutParams.WRAP_CONTENT));
        layout.addView(brightnessSeekBar);
        
        // Add color preview
        colorPreview = new View(getContext());
        colorPreview.setBackgroundColor(currentColor);
        colorPreview.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 
                100));
        layout.addView(colorPreview);
        
        // Add buttons
        LinearLayout buttonLayout = new LinearLayout(getContext());
        buttonLayout.setOrientation(LinearLayout.HORIZONTAL);
        buttonLayout.setPadding(0, 20, 0, 0);
        
        Button cancelButton = new Button(getContext());
        cancelButton.setText("取消");
        cancelButton.setLayoutParams(new LinearLayout.LayoutParams(
                0, 
                LinearLayout.LayoutParams.WRAP_CONTENT, 
                1));
        
        Button okButton = new Button(getContext());
        okButton.setText("确定");
        okButton.setLayoutParams(new LinearLayout.LayoutParams(
                0, 
                LinearLayout.LayoutParams.WRAP_CONTENT, 
                1));
        
        buttonLayout.addView(cancelButton);
        buttonLayout.addView(okButton);
        layout.addView(buttonLayout);
        
        // Set dialog content view
        setContentView(layout);
        
        // Set up listeners
        colorWheel.setOnColorChangedListener(new ColorWheelView.OnColorChangedListener() {
            @Override
            public void onColorChanged(int color) {
                // Preserve brightness
                int brightness = brightnessSeekBar.getProgress();
                currentColor = adjustBrightness(color, brightness);
                updateColorPreview();
            }
        });
        
        brightnessSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) {
                    currentColor = adjustBrightness(currentColor, progress);
                    updateColorPreview();
                }
            }
            
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        
        cancelButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                dismiss();
            }
        });
        
        okButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (listener != null) {
                    listener.onColorSelected(currentColor);
                }
                dismiss();
            }
        });
        
        // Set initial values
        colorWheel.setColor(currentColor);
        updateColorPreview();
    }
    
    private void updateColorPreview() {
        if (colorPreview != null) {
            colorPreview.setBackgroundColor(currentColor);
        }
    }
    
    private int getBrightness(int color) {
        float[] hsv = new float[3];
        Color.colorToHSV(color, hsv);
        return Math.round(hsv[2] * 255);
    }
    
    private int adjustBrightness(int color, int brightness) {
        float[] hsv = new float[3];
        Color.colorToHSV(color, hsv);
        hsv[2] = brightness / 255f;
        return Color.HSVToColor(hsv);
    }
    
    public void setOnColorSelectedListener(OnColorSelectedListener listener) {
        this.listener = listener;
    }
    
    // Color wheel view implementation
    private static class ColorWheelView extends View {
        
        public interface OnColorChangedListener {
            void onColorChanged(int color);
        }
        
        private Paint paint;
        private Paint strokePaint;
        private RectF wheelRect;
        private int color = Color.RED;
        private OnColorChangedListener listener;
        
        public ColorWheelView(Context context) {
            super(context);
            init();
        }
        
        private void init() {
            paint = new Paint(Paint.ANTI_ALIAS_FLAG);
            
            strokePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            strokePaint.setStyle(Paint.Style.STROKE);
            strokePaint.setStrokeWidth(5);
            strokePaint.setColor(Color.BLACK);
            
            wheelRect = new RectF();
        }
        
        @Override
        protected void onSizeChanged(int w, int h, int oldw, int oldh) {
            super.onSizeChanged(w, h, oldw, oldh);
            int minDim = Math.min(w, h);
            int padding = minDim / 10;
            wheelRect.set(padding, padding, minDim - padding, minDim - padding);
        }
        
        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            
            // Draw color wheel
            final int wheelRadius = (int) wheelRect.width() / 2;
            final int centerX = (int) wheelRect.centerX();
            final int centerY = (int) wheelRect.centerY();
            
            for (int angle = 0; angle < 360; angle++) {
                float[] hsv = {angle, 1.0f, 1.0f};
                paint.setColor(Color.HSVToColor(hsv));
                canvas.drawArc(wheelRect, angle, 1.5f, true, paint);
            }
            
            // Draw center circle
            paint.setColor(Color.WHITE);
            canvas.drawCircle(centerX, centerY, wheelRadius / 3, paint);
            canvas.drawCircle(centerX, centerY, wheelRadius / 3, strokePaint);
            
            // Draw selected color
            paint.setColor(color);
            canvas.drawCircle(centerX, centerY, wheelRadius / 4, paint);
        }
        
        @Override
        public boolean onTouchEvent(MotionEvent event) {
            float x = event.getX();
            float y = event.getY();
            
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                case MotionEvent.ACTION_MOVE:
                    // Calculate angle
                    float centerX = wheelRect.centerX();
                    float centerY = wheelRect.centerY();
                    double angle = Math.toDegrees(Math.atan2(y - centerY, x - centerX));
                    if (angle < 0) angle += 360;
                    
                    // Calculate distance from center
                    float radius = wheelRect.width() / 2;
                    float dx = x - centerX;
                    float dy = y - centerY;
                    float distance = (float) Math.sqrt(dx * dx + dy * dy);
                    
                    // Only process if within the wheel
                    if (distance <= radius) {
                        // Calculate color
                        float saturation = Math.min(distance / radius, 1.0f);
                        float[] hsv = {(float) angle, saturation, 1.0f};
                        int newColor = Color.HSVToColor(hsv);
                        
                        if (newColor != color) {
                            color = newColor;
                            invalidate();
                            
                            if (listener != null) {
                                listener.onColorChanged(color);
                            }
                        }
                    }
                    return true;
            }
            
            return super.onTouchEvent(event);
        }
        
        public void setColor(int color) {
            this.color = color;
            invalidate();
        }
        
        public void setOnColorChangedListener(OnColorChangedListener listener) {
            this.listener = listener;
        }
    }
} 