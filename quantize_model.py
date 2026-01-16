#!/usr/bin/env python3
"""
Quantize a trained EMS-YOLO model to different precision levels.

Usage:
    python quantize_model.py --weights path/to/model.pt --quant-type fp16 --output quantized_models/
    python quantize_model.py --weights path/to/model.pt --quant-type dynamic_int8 --output quantized_models/
    python quantize_model.py --weights path/to/model.pt --quant-type static_int8 --data data/coco.yaml --output quantized_models/
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.quantization import (
    quantize_dynamic, prepare, convert, 
    get_default_qconfig, get_default_qat_qconfig,
    QConfig, MinMaxObserver, PerChannelMinMaxObserver
)

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from models.yolo import Model
from utils.general import LOGGER, colorstr, check_file, check_yaml, check_dataset
from utils.datasets import create_dataloader


def get_model_size_mb(model_path):
    """Get model file size in MB"""
    return Path(model_path).stat().st_size / (1024 * 1024)


def quantize_fp16(model, output_path, device='cpu'):
    """
    Convert model to FP16 (half precision).
    
    Args:
        model: PyTorch model
        output_path: Path to save quantized model
        device: Device to use ('cpu' or 'cuda')
    """
    LOGGER.info(f"{colorstr('bold', 'Quantizing to FP16...')}")
    
    # Move to device and convert to FP16
    model = model.to(device)
    model = model.half()
    model.eval()
    
    # Save model
    torch.save({
        'model': model,
        'quantization_type': 'fp16',
        'dtype': 'float16'
    }, output_path)
    
    size_mb = get_model_size_mb(output_path)
    LOGGER.info(f"{colorstr('green', '✓')} FP16 model saved to {output_path} ({size_mb:.2f} MB)")
    return model


def quantize_dynamic_int8(model, output_path, device='cpu'):
    """
    Apply dynamic INT8 quantization.
    Weights are quantized to INT8, activations remain FP32.
    
    Args:
        model: PyTorch model
        output_path: Path to save quantized model
        device: Device to use ('cpu' or 'cuda')
    """
    LOGGER.info(f"{colorstr('bold', 'Quantizing to Dynamic INT8...')}")
    
    model = model.to(device)
    model.eval()
    
    # Dynamic quantization - quantizes weights to INT8
    # Note: This works best with Linear and LSTM layers
    # For Conv layers, static quantization is usually better
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},  # Layers to quantize
        dtype=torch.qint8
    )
    
    # Save quantized model
    torch.save({
        'model': quantized_model,
        'quantization_type': 'dynamic_int8',
        'dtype': 'qint8'
    }, output_path)
    
    size_mb = get_model_size_mb(output_path)
    LOGGER.info(f"{colorstr('green', '✓')} Dynamic INT8 model saved to {output_path} ({size_mb:.2f} MB)")
    return quantized_model


def quantize_static_int8(model, calibration_loader, output_path, device='cpu', num_calib_batches=100):
    """
    Apply static INT8 quantization with calibration.
    Both weights and activations are quantized to INT8.
    
    Args:
        model: PyTorch model
        calibration_loader: DataLoader for calibration data
        output_path: Path to save quantized model
        device: Device to use ('cpu' or 'cuda')
        num_calib_batches: Number of batches to use for calibration
    """
    LOGGER.info(f"{colorstr('bold', 'Quantizing to Static INT8...')}")
    LOGGER.info(f"Calibrating with {num_calib_batches} batches...")
    
    model = model.to(device)
    model.eval()
    
    # Determine backend based on device
    if device == 'cpu':
        backend = 'fbgemm'  # For x86 CPUs
    else:
        backend = 'qnnpack'  # For ARM/mobile
    
    # Set quantization config
    try:
        qconfig = get_default_qconfig(backend)
        model.qconfig = qconfig
    except Exception as e:
        LOGGER.warning(f"Could not set {backend} qconfig, using default: {e}")
        # Fallback to default
        from torch.quantization import default_qconfig
        model.qconfig = default_qconfig
    
    # Prepare model for calibration
    model_prepared = prepare(model)
    
    # Calibrate model
    with torch.no_grad():
        for i, (imgs, targets, paths, _) in enumerate(calibration_loader):
            if i >= num_calib_batches:
                break
            imgs = imgs.to(device).float() / 255.0
            try:
                model_prepared(imgs)
            except Exception as e:
                LOGGER.warning(f"Error during calibration batch {i}: {e}")
                continue
            
            if (i + 1) % 10 == 0:
                LOGGER.info(f"  Calibrated {i + 1}/{num_calib_batches} batches...")
    
    # Convert to quantized model
    LOGGER.info("Converting to quantized model...")
    quantized_model = convert(model_prepared)
    
    # Save quantized model
    torch.save({
        'model': quantized_model,
        'quantization_type': 'static_int8',
        'dtype': 'qint8',
        'backend': backend,
        'num_calib_batches': num_calib_batches
    }, output_path)
    
    size_mb = get_model_size_mb(output_path)
    LOGGER.info(f"{colorstr('green', '✓')} Static INT8 model saved to {output_path} ({size_mb:.2f} MB)")
    return quantized_model


def quantize_per_channel_int8(model, calibration_loader, output_path, device='cpu', num_calib_batches=100):
    """
    Apply per-channel INT8 quantization (better accuracy than per-tensor).
    
    Args:
        model: PyTorch model
        calibration_loader: DataLoader for calibration data
        output_path: Path to save quantized model
        device: Device to use ('cpu' or 'cuda')
        num_calib_batches: Number of batches to use for calibration
    """
    LOGGER.info(f"{colorstr('bold', 'Quantizing to Per-Channel INT8...')}")
    LOGGER.info(f"Calibrating with {num_calib_batches} batches...")
    
    model = model.to(device)
    model.eval()
    
    # Per-channel quantization config (better accuracy)
    per_channel_qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric
        )
    )
    
    # Apply to all Conv2d layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            module.qconfig = per_channel_qconfig
    
    # Prepare model for calibration
    model_prepared = prepare(model)
    
    # Calibrate model
    with torch.no_grad():
        for i, (imgs, targets, paths, _) in enumerate(calibration_loader):
            if i >= num_calib_batches:
                break
            imgs = imgs.to(device).float() / 255.0
            try:
                model_prepared(imgs)
            except Exception as e:
                LOGGER.warning(f"Error during calibration batch {i}: {e}")
                continue
            
            if (i + 1) % 10 == 0:
                LOGGER.info(f"  Calibrated {i + 1}/{num_calib_batches} batches...")
    
    # Convert to quantized model
    LOGGER.info("Converting to quantized model...")
    quantized_model = convert(model_prepared)
    
    # Save quantized model
    torch.save({
        'model': quantized_model,
        'quantization_type': 'per_channel_int8',
        'dtype': 'qint8',
        'num_calib_batches': num_calib_batches
    }, output_path)
    
    size_mb = get_model_size_mb(output_path)
    LOGGER.info(f"{colorstr('green', '✓')} Per-Channel INT8 model saved to {output_path} ({size_mb:.2f} MB)")
    return quantized_model


def main():
    parser = argparse.ArgumentParser(
        description='Quantize EMS-YOLO model to different precision levels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--weights', type=str, required=True, 
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--quant-type', type=str, required=True,
                       choices=['fp16', 'dynamic_int8', 'static_int8', 'per_channel_int8'],
                       help='Quantization type: fp16, dynamic_int8, static_int8, or per_channel_int8')
    parser.add_argument('--output', type=str, default='quantized_models',
                       help='Output directory to save quantized models')
    parser.add_argument('--data', type=str, default=None,
                       help='Dataset YAML file path (required for static_int8 and per_channel_int8)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for quantization')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for calibration (if needed)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for calibration data')
    parser.add_argument('--num-calib-batches', type=int, default=100,
                       help='Number of batches to use for calibration (static quantization)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    # Validate inputs
    weights_path = Path(args.weights)
    if not weights_path.exists():
        LOGGER.error(f"Model file not found: {weights_path}")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if calibration data is needed
    needs_calibration = args.quant_type in ['static_int8', 'per_channel_int8']
    if needs_calibration and args.data is None:
        LOGGER.error(f"--data argument is required for {args.quant_type} quantization")
        return
    
    # Load model
    LOGGER.info(f"{colorstr('bold', 'Loading model from:')} {weights_path}")
    device = torch.device(args.device)
    
    try:
        model = attempt_load(str(weights_path), map_location=device, inplace=True, fuse=False)
        
        # Handle ensemble models
        if hasattr(model, '__len__') and len(model) > 1:
            model = model[0]  # Take first model from ensemble
        
        model.eval()
        LOGGER.info(f"Model loaded successfully")
        
        # Get original model size
        original_size = get_model_size_mb(weights_path)
        LOGGER.info(f"Original model size: {original_size:.2f} MB")
        
    except Exception as e:
        LOGGER.error(f"Error loading model: {e}")
        return
    
    # Prepare calibration data if needed
    calibration_loader = None
    if needs_calibration:
        LOGGER.info(f"{colorstr('bold', 'Loading calibration data...')}")
        try:
            data_dict = check_dataset(args.data) if args.data else None
            train_path = data_dict['train'] if data_dict else None
            
            if train_path:
                calibration_loader, _ = create_dataloader(
                    train_path,
                    args.imgsz,
                    args.batch_size,
                    32,  # grid size
                    single_cls=False,
                    hyp=None,
                    augment=False,
                    cache=None,
                    rect=False,
                    rank=-1,
                    workers=args.workers,
                    pad=0.5,
                    prefix=colorstr('calibration: ')
                )
                LOGGER.info(f"Calibration dataloader created with {len(calibration_loader)} batches")
            else:
                LOGGER.error("Could not create calibration dataloader")
                return
        except Exception as e:
            LOGGER.error(f"Error creating calibration dataloader: {e}")
            return
    
    # Generate output filename
    model_name = weights_path.stem
    quant_suffix = args.quant_type
    output_filename = f"{model_name}_{quant_suffix}.pt"
    output_path = output_dir / output_filename
    
    # Apply quantization
    try:
        if args.quant_type == 'fp16':
            quantized_model = quantize_fp16(model, output_path, device)
            
        elif args.quant_type == 'dynamic_int8':
            quantized_model = quantize_dynamic_int8(model, output_path, device)
            
        elif args.quant_type == 'static_int8':
            quantized_model = quantize_static_int8(
                model, calibration_loader, output_path, 
                device, args.num_calib_batches
            )
            
        elif args.quant_type == 'per_channel_int8':
            quantized_model = quantize_per_channel_int8(
                model, calibration_loader, output_path,
                device, args.num_calib_batches
            )
        
        # Print summary
        quantized_size = get_model_size_mb(output_path)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        
        LOGGER.info("")
        LOGGER.info("=" * 60)
        LOGGER.info(f"{colorstr('bold', 'Quantization Summary:')}")
        LOGGER.info(f"  Original model: {weights_path}")
        LOGGER.info(f"  Quantized model: {output_path}")
        LOGGER.info(f"  Quantization type: {args.quant_type}")
        LOGGER.info(f"  Original size: {original_size:.2f} MB")
        LOGGER.info(f"  Quantized size: {quantized_size:.2f} MB")
        LOGGER.info(f"  Compression ratio: {compression_ratio:.2f}x")
        LOGGER.info("=" * 60)
        
    except Exception as e:
        LOGGER.error(f"Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()

