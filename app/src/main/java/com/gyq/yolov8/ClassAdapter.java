package com.gyq.yolov8;

import android.util.SparseBooleanArray;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.Filter;
import android.widget.Filterable;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.List;

public class ClassAdapter extends RecyclerView.Adapter<ClassAdapter.ClassViewHolder> implements Filterable {
    
    private List<ClassItem> allClasses;
    private List<ClassItem> filteredClasses;
    private SparseBooleanArray selectedClassIds;
    
    public ClassAdapter(List<ClassItem> classes, SparseBooleanArray selectedClassIds) {
        this.allClasses = classes;
        this.filteredClasses = new ArrayList<>(classes);
        this.selectedClassIds = selectedClassIds;
    }
    
    @NonNull
    @Override
    public ClassViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_class_checkbox, parent, false);
        return new ClassViewHolder(view);
    }
    
    @Override
    public void onBindViewHolder(@NonNull ClassViewHolder holder, int position) {
        ClassItem item = filteredClasses.get(position);
        holder.checkBox.setText(item.getDisplayName());
        
        // Set checked state without triggering listener
        holder.checkBox.setOnCheckedChangeListener(null);
        holder.checkBox.setChecked(selectedClassIds.get(item.getClassId(), false));
        
        // Set new listener
        holder.checkBox.setOnCheckedChangeListener((buttonView, isChecked) -> {
            selectedClassIds.put(item.getClassId(), isChecked);
        });
    }
    
    @Override
    public int getItemCount() {
        return filteredClasses.size();
    }
    
    public void selectAll() {
        for (ClassItem item : filteredClasses) {
            selectedClassIds.put(item.getClassId(), true);
        }
        notifyDataSetChanged();
    }
    
    public void clearAll() {
        for (ClassItem item : filteredClasses) {
            selectedClassIds.put(item.getClassId(), false);
        }
        notifyDataSetChanged();
    }
    
    @Override
    public Filter getFilter() {
        return new Filter() {
            @Override
            protected FilterResults performFiltering(CharSequence constraint) {
                String query = constraint.toString().toLowerCase().trim();
                
                List<ClassItem> filtered = new ArrayList<>();
                if (query.isEmpty()) {
                    filtered.addAll(allClasses);
                } else {
                    for (ClassItem item : allClasses) {
                        if (item.getEnglishName().toLowerCase().contains(query) ||
                                item.getDisplayName().toLowerCase().contains(query)) {
                            filtered.add(item);
                        }
                    }
                }
                
                FilterResults results = new FilterResults();
                results.values = filtered;
                results.count = filtered.size();
                return results;
            }
            
            @Override
            protected void publishResults(CharSequence constraint, FilterResults results) {
                filteredClasses.clear();
                //noinspection unchecked
                filteredClasses.addAll((List<ClassItem>) results.values);
                notifyDataSetChanged();
            }
        };
    }
    
    static class ClassViewHolder extends RecyclerView.ViewHolder {
        CheckBox checkBox;
        
        public ClassViewHolder(@NonNull View itemView) {
            super(itemView);
            checkBox = itemView.findViewById(R.id.checkBoxClass);
        }
    }
    
    // Data class to hold class information
    public static class ClassItem {
        private final int classId;
        private final String englishName;
        private final String displayName;
        
        public ClassItem(int classId, String englishName, String displayName) {
            this.classId = classId;
            this.englishName = englishName;
            this.displayName = displayName;
        }
        
        public int getClassId() {
            return classId;
        }
        
        public String getEnglishName() {
            return englishName;
        }
        
        public String getDisplayName() {
            return displayName;
        }
    }
} 