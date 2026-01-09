"""
Visualize bounding boxes from label files on images.

Usage:
    python visualize_labels.py --images /path/to/images --labels /path/to/labels --output /path/to/output
    python visualize_labels.py --images /path/to/images --labels /path/to/labels --data data/raddet.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.general import LOGGER, check_yaml, colorstr, increment_path, xywh2xyxy
from utils.plots import Annotator, colors


def load_labels(label_path):
    """Load labels from a txt file.
    
    Format: class x y w h [conf] (normalized xywh)
    Returns: list of [class, x, y, w, h, conf] or [class, x, y, w, h]
    """
    labels = []
    if not label_path.exists():
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) > 5 else None
                if conf is not None:
                    labels.append([cls, x, y, w, h, conf])
                else:
                    labels.append([cls, x, y, w, h])
    return labels


def visualize_labels_on_image(image_path, label_path, names, save_path=None, show_conf=True):
    """Draw bounding boxes from label file on image.
    
    Args:
        image_path: Path to image file
        label_path: Path to label txt file
        names: List of class names
        save_path: Path to save annotated image (optional)
        show_conf: Whether to show confidence scores in labels
    
    Returns:
        Annotated image (numpy array)
    """
    # Load image
    im0 = cv2.imread(str(image_path))
    if im0 is None:
        LOGGER.warning(f'Could not load image: {image_path}')
        return None
    
    h, w = im0.shape[:2]
    
    # Load labels
    labels = load_labels(label_path)
    if not labels:
        LOGGER.info(f'No labels found in {label_path}')
        return im0
    
    # Create annotator
    annotator = Annotator(im0, line_width=max(round(sum(im0.shape) / 2 * 0.003), 2), example=str(names))
    
    # Convert labels to numpy array for processing
    labels_array = np.array(labels)
    
    # Extract components
    classes = labels_array[:, 0].astype(int)
    xywh_norm = labels_array[:, 1:5]  # normalized xywh
    
    # Convert normalized xywh to pixel xyxy
    xywh = xywh_norm * np.array([w, h, w, h])  # denormalize
    xyxy = xywh2xyxy(torch.tensor(xywh)).numpy()  # convert to xyxy
    
    # Draw boxes
    for i, (cls, box) in enumerate(zip(classes, xyxy)):
        # Get class name
        cls_name = names[cls] if cls < len(names) else f'class{cls}'
        
        # Create label
        if len(labels[i]) > 5 and show_conf:
            conf = labels[i][5]
            label = f'{cls_name} {conf:.2f}'
        else:
            label = cls_name
        
        # Get color for this class
        color = colors(cls, bgr=True)
        
        # Draw box
        annotator.box_label(box, label, color=color)
    
    # Get annotated image
    im0_annotated = annotator.result()
    
    # Save if path provided
    if save_path:
        cv2.imwrite(str(save_path), im0_annotated)
        LOGGER.info(f'Saved annotated image to {save_path}')
    
    return im0_annotated


def main(opt):
    # Load class names from data yaml or use defaults
    if opt.data:
        with open(opt.data, 'r') as f:
            data = yaml.safe_load(f)
        names = data.get('names', [f'class{i}' for i in range(data.get('nc', 80))])
    else:
        # Default class names (you can modify this)
        names = ['Rect', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4', 'Px', 'ZadoffChu', 'LFM', 'FMCW']
    
    # Setup paths
    images_dir = Path(opt.images)
    labels_dir = Path(opt.labels) if opt.labels else images_dir.parent / 'labels'
    output_dir = Path(opt.output) if opt.output else images_dir.parent / 'visualized'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']
    image_files = []
    for ext in IMG_FORMATS:
        image_files.extend(list(images_dir.glob(f'*.{ext}')))
        image_files.extend(list(images_dir.glob(f'*.{ext.upper()}')))
    
    if not image_files:
        LOGGER.error(f'No images found in {images_dir}')
        return
    
    LOGGER.info(f'Found {len(image_files)} images')
    
    # Process each image
    for img_path in sorted(image_files):
        # Find corresponding label file
        label_path = labels_dir / f'{img_path.stem}.txt'
        
        if not label_path.exists():
            LOGGER.warning(f'Label file not found: {label_path}, skipping {img_path.name}')
            continue
        
        # Create output path
        output_path = output_dir / img_path.name
        
        # Visualize
        visualize_labels_on_image(img_path, label_path, names, save_path=output_path, show_conf=opt.show_conf)
    
    LOGGER.info(f'Visualization complete! Annotated images saved to {output_dir}')


def parse_opt():
    parser = argparse.ArgumentParser(description='Visualize bounding boxes from label files on images')
    parser.add_argument('--images', type=str, required=True, help='path to images directory')
    parser.add_argument('--labels', type=str, default='', help='path to labels directory (default: images_dir/../labels)')
    parser.add_argument('--output', type=str, default='', help='path to output directory (default: images_dir/../visualized)')
    parser.add_argument('--data', type=str, default='', help='path to data yaml file (for class names)')
    parser.add_argument('--show-conf', action='store_true', help='show confidence scores in labels')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

