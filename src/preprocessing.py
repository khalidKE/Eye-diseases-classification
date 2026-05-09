import os
import cv2
import glob
import numpy as np
import yaml

def load_config(config_path="shared/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def crop_image_from_gray(img, tol=7):
    """
    Crop the black borders from fundus images.
    Finds the contours of the actual eye circle and crops around it.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads, crops, and resizes an image. Returns None if corrupt.
    Normalization [0, 1] is typically handled by ImageDataGenerator or Dataset API,
    but we can optionally apply it here or just save as standard RGB.
    We will save it as standard RGB (0-255) to processed dir, 
    and let the dataloader normalize it to float32 [0,1].
    """
    try:
        # Load in BGR
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Corrupt or unreadable image -> {image_path}")
            return None
            
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Medical crop (remove black borders)
        img = crop_image_from_gray(img)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        return img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_dataset(config_path="configs/config.yaml"):
    """
    Reads from data/raw, processes images, and saves to data/processed.
    """
    config = load_config(config_path)
    raw_dir = config['data']['raw_dir']
    dataset_dir = config['data']['dataset_dir']
    target_size = tuple(config['data']['target_size'])
    classes = config['data']['classes']
    
    print(f"Starting preprocessing pipeline...")
    print(f"Reading from: {raw_dir}")
    print(f"Writing to: {dataset_dir}")
    print(f"Target size: {target_size}")
    
    if not os.path.exists(raw_dir):
        print(f"Error: {raw_dir} does not exist.")
        return
        
    for cls in classes:
        in_cls_dir = os.path.join(raw_dir, cls)
        out_cls_dir = os.path.join(dataset_dir, cls)
        os.makedirs(out_cls_dir, exist_ok=True)
        
        if not os.path.exists(in_cls_dir):
            print(f"Skipping class {cls}, directory not found.")
            continue
            
        image_paths = glob.glob(os.path.join(in_cls_dir, '*.*'))
        print(f"Processing {len(image_paths)} images for class '{cls}'...")
        
        valid_count = 0
        corrupt_count = 0
        
        for img_path in image_paths:
            img = preprocess_image(img_path, target_size)
            if img is not None:
                # Save processed image. Convert back to BGR for cv2.imwrite
                save_path = os.path.join(out_cls_dir, os.path.basename(img_path))
                if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    save_path += '.jpg'
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                valid_count += 1
            else:
                corrupt_count += 1
                
        print(f"Class '{cls}': {valid_count} valid, {corrupt_count} corrupt.")

if __name__ == "__main__":
    process_dataset()
