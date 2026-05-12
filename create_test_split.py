"""
Create train/val/test split for evaluation
"""
import os
import shutil
import random
from pathlib import Path

def create_split(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """Split data into train/val/test folders"""
    
    # Get all classes (subdirectories)
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }
        
        for split_name, split_images in splits.items():
            split_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            
            for img in split_images:
                src = os.path.join(cls_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy2(src, dst)
            
            print(f"{cls}/{split_name}: {len(split_images)} images")

if __name__ == "__main__":
    random.seed(42)
    data_dir = "data/data_processed/processed"
    output_dir = "data/data_processed/processed_split"
    
    create_split(data_dir, output_dir)
    print(f"\n✅ Split created at: {output_dir}")
