#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================

# This dictionary is copied directly from the original script to ensure labels are IDENTICAL.
# It's the source of truth for all class labels.
dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    }
}

# --- Define which clothing labels to remove for the agnostic map ---
# Based on the LIP dataset labels above:
# 5: Upper-clothes, 6: Dress, 7: Coat, 10: Jumpsuits, 12: Skirt
CLOTHING_IDS_TO_REMOVE = [5, 6, 7]

# --- Define the target resolution for the final agnostic map ---
# This is often a specific size required by the VTON model.
AGNOSTIC_MAP_HEIGHT = 256
AGNOSTIC_MAP_WIDTH = 192

# ==============================================================================

def get_arguments():
    """Parse all the arguments provided from the CLI."""
    parser = argparse.ArgumentParser(description="Unified Human Parsing and Agnostic Map Generation for Video")
    parser.add_argument("--video-path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the folder where output subdirectories will be created.")
    parser.add_argument("--model-restore", type=str, required=True, help="Path to the pretrained model parameters.")
    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr'], help="Dataset standard to use for labels.")
    parser.add_argument("--gpu", type=str, default='0', help="Choose GPU device.")
    return parser.parse_args()

def main():
    args = get_arguments()

    # --- Setup Device, Model, and Directories ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get settings from the dictionary
    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    labels = dataset_settings[args.dataset]['label']
    print(f"Using '{args.dataset}' settings. Classes: {num_classes}. Input size: {input_size}")
    print(f"Labels to be removed for agnostic map: {[labels[i] for i in CLOTHING_IDS_TO_REMOVE]}")

    # Load the model
    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
    state_dict = torch.load(args.model_restore, map_location=device)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Define the transformation for each frame
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
    ])

    # Setup video capture and output directories
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {args.video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directories for both parse types
    output_parse_dir = os.path.join(args.output_dir, 'image-parse-v3')
    output_agnostic_dir = os.path.join(args.output_dir, 'image-parse-agnostic-v3.2')
    os.makedirs(output_parse_dir, exist_ok=True)
    os.makedirs(output_agnostic_dir, exist_ok=True)

    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Processing video frames")

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            original_h, original_w, _ = frame.shape
            
            # --- 1. Preprocess Frame for Model ---
            # Convert frame from BGR (OpenCV) to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Resize and transform
            img_resized = img_pil.resize(input_size, Image.BILINEAR)
            img_tensor = transform(img_resized).unsqueeze(0).to(device)

            # --- 2. Run Inference ---
            outputs = model(img_tensor)
            # We are interested in the final, most refined output
            parsing_logits = outputs[0][-1] 
            
            # Upsample logits to the input size for processing
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsampled_logits = upsample(parsing_logits)
            
            # Get the parsing result by finding the class with the max logit
            parsing_result = upsampled_logits.squeeze().argmax(0).cpu().numpy()
            
            # --- 3. Generate and Save Full Parse Map (`image-parse-v3`) ---
            # Resize the parsing map back to the original video frame size
            # IMPORTANT: Use NEAREST interpolation to avoid creating new, invalid labels
            final_parse_map = cv2.resize(parsing_result.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            
            # Save the result as a single-channel indexed PNG
            parse_map_filename = f"frame_{frame_count:06d}.png"
            Image.fromarray(final_parse_map).save(os.path.join(output_parse_dir, parse_map_filename))
            
            # --- 4. Generate and Save Agnostic Parse Map (`image-parse-agnostic-v3.2`) ---
            # Create a copy to modify
            agnostic_map = final_parse_map.copy()
            # Set all clothing pixels to 0 (background)
            for label_id in CLOTHING_IDS_TO_REMOVE:
                agnostic_map[final_parse_map == label_id] = 0
                
            # Resize this agnostic map to the final target size required by the VTON model
            final_agnostic_map = cv2.resize(
                agnostic_map,
                (AGNOSTIC_MAP_WIDTH, AGNOSTIC_MAP_HEIGHT),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Save the final agnostic map
            agnostic_map_filename = f"frame_{frame_count:06d}.png"
            Image.fromarray(final_agnostic_map.astype(np.uint8)).save(os.path.join(output_agnostic_dir, agnostic_map_filename))
            
            frame_count += 1
            pbar.update(1)

    pbar.close()
    cap.release()
    print("\n--- Processing Complete ---")
    print(f"Saved {frame_count} full parse maps to: {output_parse_dir}")
    print(f"Saved {frame_count} agnostic maps to: {output_agnostic_dir}")


if __name__ == '__main__':
    main()