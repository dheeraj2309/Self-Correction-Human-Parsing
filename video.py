#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import ffmpeg # <<< NEW: Import ffmpeg for robust video writing

import torchvision.transforms as transforms

import networks

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================
dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512], 'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    }
}

CLOTHING_IDS_TO_REMOVE = [5, 6, 7] # Upper-clothes, Dress, Coat
AGNOSTIC_MAP_HEIGHT = 256
AGNOSTIC_MAP_WIDTH = 192

# ==============================================================================

# <<< NEW: Function to generate a color palette for visualization >>>
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map as a list of RGB tuples.
    """
    palette = np.zeros((num_cls, 3), dtype=np.uint8)
    for j in range(0, num_cls):
        lab = j
        i = 0
        while lab:
            palette[j, 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j, 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j, 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

# <<< NEW: Robust FFMPEG Video Writer Class >>>
class FFMPEG_VideoWriter:
    def __init__(self, filename, fps, width, height):
        self.process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', r=fps)
            .output(filename, pix_fmt='yuv420p', vcodec='libx264')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    def write_frame(self, frame):
        self.process.stdin.write(frame.astype(np.uint8).tobytes())
    def close(self):
        self.process.stdin.close()
        self.process.wait()


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

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    labels = dataset_settings[args.dataset]['label']
    print(f"Using '{args.dataset}' settings. Classes: {num_classes}. Input size: {input_size}")
    print(f"Labels to be removed for agnostic map: {[labels[i] for i in CLOTHING_IDS_TO_REMOVE]}")

    # Load the model
    # ... (model loading code is unchanged)
    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
    state_dict = torch.load(args.model_restore, map_location=device)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {args.video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create output directories
    output_parse_dir = os.path.join(args.output_dir, 'image-parse-v3')
    output_agnostic_dir = os.path.join(args.output_dir, 'image-parse-agnostic-v3.2')
    # <<< NEW: Directory for colored frames (for visualization) >>>
    output_colored_dir = os.path.join(args.output_dir, 'image-parse-v3-colored')
    os.makedirs(output_parse_dir, exist_ok=True)
    os.makedirs(output_agnostic_dir, exist_ok=True)
    os.makedirs(output_colored_dir, exist_ok=True)

    # <<< NEW: Get the color palette >>>
    palette = get_palette(num_classes)
    
    video_writer = None # <<< NEW: Initialize video writer to None
    
    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Processing video frames")

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            original_h, original_w, _ = frame.shape

            # <<< NEW: Initialize video writer on the first frame >>>
            if video_writer is None:
                video_output_path = os.path.join(args.output_dir, "segmentation_video.mp4")
                video_writer = FFMPEG_VideoWriter(video_output_path, fps, original_w, original_h)

            # --- 1. Preprocess Frame for Model ---
            # ... (preprocessing is unchanged)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_resized = img_pil.resize(input_size, Image.BILINEAR)
            img_tensor = transform(img_resized).unsqueeze(0).to(device)

            # --- 2. Run Inference ---
            # ... (inference is unchanged)
            outputs = model(img_tensor)
            parsing_logits = outputs[0][-1]
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsampled_logits = upsample(parsing_logits)
            parsing_result = upsampled_logits.squeeze().argmax(0).cpu().numpy()
            
            # --- 3. Generate and Save Full Parse Map (`image-parse-v3`) ---
            final_parse_map = cv2.resize(parsing_result.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            
            # Save the CORRECT single-channel indexed PNG for the model
            parse_map_filename = f"frame_{frame_count:06d}.png"
            Image.fromarray(final_parse_map).save(os.path.join(output_parse_dir, parse_map_filename))
            
            # <<< NEW: Create and save the colored version for visualization >>>
            colored_parse_map = palette[final_parse_map].astype(np.uint8)
            colored_parse_map_bgr = cv2.cvtColor(colored_parse_map, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
            Image.fromarray(colored_parse_map).save(os.path.join(output_colored_dir, parse_map_filename))
            
            # <<< NEW: Write the colored frame to the video >>>
            video_writer.write_frame(colored_parse_map_bgr)

            # --- 4. Generate and Save Agnostic Parse Map (`image-parse-agnostic-v3.2`) ---
            # ... (agnostic map generation is unchanged)
            agnostic_map = final_parse_map.copy()
            for label_id in CLOTHING_IDS_TO_REMOVE:
                agnostic_map[final_parse_map == label_id] = 0
            final_agnostic_map = cv2.resize(
                agnostic_map, (AGNOSTIC_MAP_WIDTH, AGNOSTIC_MAP_HEIGHT), interpolation=cv2.INTER_NEAREST
            )
            agnostic_map_filename = f"frame_{frame_count:06d}.png"
            Image.fromarray(final_agnostic_map.astype(np.uint8)).save(os.path.join(output_agnostic_dir, agnostic_map_filename))
            
            frame_count += 1
            pbar.update(1)

    pbar.close()
    cap.release()
    # <<< NEW: Close the video writer >>>
    if video_writer:
        video_writer.close()
        
    print("\n--- Processing Complete ---")
    print(f"Saved {frame_count} raw parse maps to: {output_parse_dir} (Use these for VTON)")
    print(f"Saved {frame_count} colored parse maps to: {output_colored_dir} (For visualization)")
    print(f"Saved {frame_count} agnostic maps to: {output_agnostic_dir}")
    if video_writer:
        print(f"Saved segmentation video to: {video_output_path}")


if __name__ == '__main__':
    main()