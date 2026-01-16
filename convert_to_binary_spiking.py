#!/usr/bin/env python3
"""
Script to convert trained model with SiLU activation to binary spiking activation.

Usage:
    python convert_to_binary_spiking.py --weights path/to/model.pt --output path/to/converted_model.pt
"""

import argparse
import sys
import traceback
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import models.yolo as yolo_module
from models.common import mem_update, Conv, Conv_A, Conv_1, Conv_2, BasicBlock, BasicBlock_1, BasicBlock_2
from models.experimental import attempt_load
from models.yolo import Model, Detect
from utils.general import LOGGER, colorstr, check_dataset, check_yaml
from utils.datasets import LoadImagesAndLabels


def convert_silu_to_binary(model):
    """
    Convert all SiLU activations to binary spiking activations.
    
    Args:
        model: PyTorch model with SiLU activations
        
    Returns:
        model: Modified model with binary activations
        conversion_stats: Dictionary with conversion statistics
    """
    conversion_stats = {
        'mem_update_silu_to_binary': 0,
        'conv_silu_to_binary': 0,
        'conv_a_silu_removed': 0,
        'total_modules_checked': 0
    }
    
    def convert_module(m, name=''):
        """Recursively convert modules"""
        conversion_stats['total_modules_checked'] += 1
        
        # Convert mem_update with act=True to act=False
        if isinstance(m, mem_update):
            if m.act:  # SiLU mode
                m.act = False  # Switch to binary spiking
                conversion_stats['mem_update_silu_to_binary'] += 1
                LOGGER.info(f"  Converted mem_update at {name}: SiLU → Binary Spiking")
        
        # Convert Conv layers (which use mem_update with act=True)
        elif isinstance(m, Conv):
            if hasattr(m, 'act') and isinstance(m.act, mem_update):
                if m.act.act:  # SiLU mode
                    m.act.act = False  # Switch to binary
                    conversion_stats['conv_silu_to_binary'] += 1
                    LOGGER.info(f"  Converted Conv layer at {name}: SiLU → Binary Spiking")
        
        # Convert Conv_A layers (which use nn.SiLU directly)
        elif isinstance(m, Conv_A):
            if hasattr(m, 'act'):
                if isinstance(m.act, nn.SiLU):
                    # Replace SiLU with mem_update(act=False)
                    # mem_update is already imported at top of file
                    m.act = mem_update(act=False)
                    conversion_stats['conv_a_silu_removed'] += 1
                    LOGGER.info(f"  Converted Conv_A layer at {name}: nn.SiLU → Binary Spiking")
        
        # Recursively process child modules
        for child_name, child_module in m.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            convert_module(child_module, full_name)
    
    LOGGER.info(f"\n{colorstr('bold', 'Converting SiLU activations to binary spiking...')}")
    convert_module(model)
    
    return model, conversion_stats


def get_layer_info(module, name='', depth=0, time_window=None, executed_module_ids=None):
    """
    Extract detailed information about a layer/module.
    
    Args:
        module: PyTorch module
        name: Module name
        depth: Depth in model hierarchy
        time_window: Time window parameter from checkpoint (optional)
        executed_module_ids: Set of module IDs that were executed during forward pass (optional)
    
    Returns:
        dict: Layer information including type, parameters, and basic blocks
    """
    info = {
        'name': name,
        'type': type(module).__name__,
        'depth': depth,
        'children': [],
        'parameters': {},
        'basic_blocks': []
    }
    
    # Get module parameters - read from actual module attributes
    if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
        info['parameters']['in_channels'] = module.in_channels
        info['parameters']['out_channels'] = module.out_channels
    
    # For Conv layers, get parameters from the conv submodule
    if isinstance(module, (Conv, Conv_A, Conv_1, Conv_2)) and hasattr(module, 'conv'):
        if hasattr(module.conv, 'in_channels'):
            info['parameters']['in_channels'] = module.conv.in_channels
        if hasattr(module.conv, 'out_channels'):
            info['parameters']['out_channels'] = module.conv.out_channels
        if hasattr(module.conv, 'kernel_size'):
            info['parameters']['kernel_size'] = module.conv.kernel_size
        if hasattr(module.conv, 'stride'):
            info['parameters']['stride'] = module.conv.stride
        if hasattr(module.conv, 'padding'):
            info['parameters']['padding'] = module.conv.padding
        if hasattr(module.conv, 'groups'):
            info['parameters']['groups'] = module.conv.groups
    
    # For BasicBlock layers, try to get from residual_function
    if isinstance(module, (BasicBlock, BasicBlock_1, BasicBlock_2)) and hasattr(module, 'residual_function'):
        # Try to find conv layers in residual_function
        for child in module.residual_function:
            if hasattr(child, 'in_channels') and hasattr(child, 'out_channels'):
                if 'in_channels' not in info['parameters']:
                    info['parameters']['in_channels'] = child.in_channels
                if 'out_channels' not in info['parameters']:
                    info['parameters']['out_channels'] = child.out_channels
                break
    
    # Direct module attributes
    if hasattr(module, 'kernel_size') and 'kernel_size' not in info['parameters']:
        info['parameters']['kernel_size'] = module.kernel_size
    if hasattr(module, 'stride') and 'stride' not in info['parameters']:
        info['parameters']['stride'] = module.stride
    if hasattr(module, 'padding') and 'padding' not in info['parameters']:
        info['parameters']['padding'] = module.padding
    
    # Decompose to basic blocks based on module type
    # Check actual activation mode from the module
    activation_type = 'binary spiking'
    if isinstance(module, Conv):
        if hasattr(module, 'act') and isinstance(module.act, mem_update):
            activation_type = 'SiLU' if module.act.act else 'binary spiking'
        info['basic_blocks'] = [
            'Snn_Conv2d',
            'batch_norm_2d',
            f'mem_update ({activation_type})'
        ]
        info['data_flow'] = 'sequential'
    elif isinstance(module, Conv_A):
        if hasattr(module, 'act'):
            if isinstance(module.act, nn.SiLU):
                activation_type = 'SiLU'
            elif isinstance(module.act, mem_update):
                activation_type = 'SiLU' if module.act.act else 'binary spiking'
        info['basic_blocks'] = [
            'Snn_Conv2d',
            'batch_norm_2d',
            f'mem_update ({activation_type})'
        ]
        info['data_flow'] = 'sequential'
    elif isinstance(module, Conv_1) or isinstance(module, Conv_2):
        info['basic_blocks'] = [
            'Snn_Conv2d',
            'batch_norm_2d'
        ]
        info['data_flow'] = 'sequential'
    elif isinstance(module, BasicBlock_2):
        # Extract actual kernel sizes from the module structure
        kernel_sizes = []
        for child in module.residual_function:
            if isinstance(child, (Conv, Conv_1, Conv_2, Conv_A)) and hasattr(child, 'conv'):
                if hasattr(child.conv, 'kernel_size'):
                    ks = child.conv.kernel_size
                    # Handle tuple kernel sizes
                    if isinstance(ks, tuple):
                        kernel_sizes.append(ks[0])  # Use first dimension
                    else:
                        kernel_sizes.append(ks)
            elif hasattr(child, 'kernel_size'):
                ks = child.kernel_size
                if isinstance(ks, tuple):
                    kernel_sizes.append(ks[0])
                else:
                    kernel_sizes.append(ks)
        
        # Use extracted kernel sizes or mark as unknown
        if len(kernel_sizes) >= 2:
            k1, k2 = kernel_sizes[0], kernel_sizes[1]
            k1_str = f"{k1}×{k1}" if k1 is not None else "?"
            k2_str = f"{k2}×{k2}" if k2 is not None else "?"
        elif len(kernel_sizes) == 1:
            k1 = kernel_sizes[0]
            k1_str = f"{k1}×{k1}" if k1 is not None else "?"
            k2_str = "?"
        else:
            k1_str = "?"
            k2_str = "?"
        
        # Extract stride and channel info to determine shortcut condition
        stride_value = None
        in_ch = info['parameters'].get('in_channels')
        out_ch = info['parameters'].get('out_channels')
        
        # Try to extract stride from first conv in residual_function
        if hasattr(module, 'residual_function'):
            for child in module.residual_function:
                if hasattr(child, 'stride'):
                    stride_value = child.stride
                    if isinstance(stride_value, tuple):
                        stride_value = stride_value[0]
                    break
                elif hasattr(child, 'conv') and hasattr(child.conv, 'stride'):
                    stride_value = child.conv.stride
                    if isinstance(stride_value, tuple):
                        stride_value = stride_value[0]
                    break
        
        # Determine which condition triggered shortcut
        shortcut_condition = []
        if stride_value is not None and stride_value != 1:
            shortcut_condition.append(f"stride={stride_value} (downsampling)")
        if in_ch is not None and out_ch is not None and in_ch != out_ch:
            shortcut_condition.append(f"channels {in_ch}→{out_ch} (dimension mismatch)")
        
        # Check if shortcut has actual layers (not empty Sequential)
        has_shortcut = False
        shortcut_was_executed = False
        if hasattr(module, 'shortcut'):
            shortcut_children = list(module.shortcut.children())
            has_shortcut = len(shortcut_children) > 0
            
            # If tracing execution, check if shortcut layers were executed
            if executed_module_ids is not None and has_shortcut:
                # Check if any shortcut layer was executed
                for child in shortcut_children:
                    if id(child) in executed_module_ids:
                        shortcut_was_executed = True
                        break
            elif has_shortcut:
                # If not tracing, assume shortcut exists if it has children
                shortcut_was_executed = True
        
        # Build residual path blocks
        residual_blocks = [
            'mem_update (binary)',
            f'Snn_Conv2d ({k1_str})',
            'batch_norm_2d',
            'mem_update (binary)',
            f'Snn_Conv2d ({k2_str})',
            'batch_norm_2d1',
        ]
        
        # Build shortcut path blocks (if exists)
        shortcut_blocks = []
        if shortcut_was_executed:
            shortcut_blocks = [
                'MaxPool3d (shortcut)',
                'mem_update (binary)',
                'Snn_Conv2d (1×1, shortcut)',
                'batch_norm_2d (shortcut)'
            ]
        
        info['basic_blocks'] = residual_blocks
        info['shortcut_blocks'] = shortcut_blocks
        info['data_flow'] = 'parallel_merge'
        info['merge_operation'] = 'Element-wise Adder (residual)'
        if shortcut_condition:
            info['shortcut_condition'] = " OR ".join(shortcut_condition)
    elif isinstance(module, BasicBlock_1):
        # Extract actual kernel sizes from the module structure
        kernel_sizes = []
        for child in module.residual_function:
            if isinstance(child, (Conv, Conv_1, Conv_2, Conv_A)) and hasattr(child, 'conv'):
                if hasattr(child.conv, 'kernel_size'):
                    ks = child.conv.kernel_size
                    if isinstance(ks, tuple):
                        kernel_sizes.append(ks[0])
                    else:
                        kernel_sizes.append(ks)
            elif hasattr(child, 'kernel_size'):
                ks = child.kernel_size
                if isinstance(ks, tuple):
                    kernel_sizes.append(ks[0])
                else:
                    kernel_sizes.append(ks)
        
        # Use extracted kernel sizes or mark as unknown
        if len(kernel_sizes) >= 2:
            k1, k2 = kernel_sizes[0], kernel_sizes[1]
            k1_str = f"{k1}×{k1}" if k1 is not None else "?"
            k2_str = f"{k2}×{k2}" if k2 is not None else "?"
        elif len(kernel_sizes) == 1:
            k1 = kernel_sizes[0]
            k1_str = f"{k1}×{k1}" if k1 is not None else "?"
            k2_str = "?"
        else:
            k1_str = "?"
            k2_str = "?"
        
        # Extract stride and channel info to determine shortcut condition
        stride_value = None
        in_ch = info['parameters'].get('in_channels')
        out_ch = info['parameters'].get('out_channels')
        
        # Try to extract stride from first conv in residual_function
        if hasattr(module, 'residual_function'):
            for child in module.residual_function:
                if hasattr(child, 'stride'):
                    stride_value = child.stride
                    if isinstance(stride_value, tuple):
                        stride_value = stride_value[0]
                    break
                elif hasattr(child, 'conv') and hasattr(child.conv, 'stride'):
                    stride_value = child.conv.stride
                    if isinstance(stride_value, tuple):
                        stride_value = stride_value[0]
                    break
        
        # Determine which condition triggered shortcut
        shortcut_condition = []
        if stride_value is not None and stride_value != 1:
            shortcut_condition.append(f"stride={stride_value} (downsampling)")
        if in_ch is not None and out_ch is not None and in_ch != out_ch:
            shortcut_condition.append(f"channels {in_ch}→{out_ch} (dimension mismatch)")
        
        # Check if shortcut has actual layers (not empty Sequential)
        has_shortcut = False
        shortcut_was_executed = False
        if hasattr(module, 'shortcut'):
            shortcut_children = list(module.shortcut.children())
            has_shortcut = len(shortcut_children) > 0
            
            # If tracing execution, check if shortcut layers were executed
            if executed_module_ids is not None and has_shortcut:
                # Check if any shortcut layer was executed
                for child in shortcut_children:
                    if id(child) in executed_module_ids:
                        shortcut_was_executed = True
                        break
            elif has_shortcut:
                # If not tracing, assume shortcut exists if it has children
                shortcut_was_executed = True
        
        # Build residual path blocks
        residual_blocks = [
            'mem_update (binary)',
            f'Snn_Conv2d ({k1_str})',
            'batch_norm_2d',
            'mem_update (binary)',
            f'Snn_Conv2d ({k2_str})',
            'batch_norm_2d1',
        ]
        
        # Build shortcut path blocks (if exists)
        shortcut_blocks = []
        if shortcut_was_executed:
            shortcut_blocks = [
                'MaxPool3d (shortcut)',
                'mem_update (binary)',
                'Snn_Conv2d (1×1, shortcut)',
                'batch_norm_2d (shortcut)'
            ]
        
        info['basic_blocks'] = residual_blocks
        info['shortcut_blocks'] = shortcut_blocks
        info['data_flow'] = 'parallel_merge'
        info['merge_operation'] = 'Element-wise Adder (residual)'
        if shortcut_condition:
            info['shortcut_condition'] = " OR ".join(shortcut_condition)
    elif isinstance(module, BasicBlock):
        # Extract stride and channel info to determine shortcut condition
        stride_value = None
        in_ch = info['parameters'].get('in_channels')
        out_ch = info['parameters'].get('out_channels')
        
        # Try to extract stride from cv1 (first conv)
        if hasattr(module, 'cv1') and hasattr(module.cv1, 'conv'):
            if hasattr(module.cv1.conv, 'stride'):
                stride_value = module.cv1.conv.stride
                if isinstance(stride_value, tuple):
                    stride_value = stride_value[0]
        
        # Determine which condition triggered shortcut
        shortcut_condition = []
        if stride_value is not None and stride_value != 1:
            shortcut_condition.append(f"stride={stride_value} (downsampling)")
        if in_ch is not None and out_ch is not None and in_ch != out_ch:
            shortcut_condition.append(f"channels {in_ch}→{out_ch} (dimension mismatch)")
        
        # Check if shortcut has actual layers (not empty Sequential)
        has_shortcut = False
        shortcut_was_executed = False
        if hasattr(module, 'shortcut'):
            shortcut_children = list(module.shortcut.children())
            has_shortcut = len(shortcut_children) > 0
            
            # If tracing execution, check if shortcut layers were executed
            if executed_module_ids is not None and has_shortcut:
                # Check if any shortcut layer was executed
                for child in shortcut_children:
                    if id(child) in executed_module_ids:
                        shortcut_was_executed = True
                        break
            elif has_shortcut:
                # If not tracing, assume shortcut exists if it has children
                shortcut_was_executed = True
        
        # Build residual path blocks
        residual_blocks = [
            'Conv (with mem_update)',
            'Conv (with mem_update)',
        ]
        
        # Build shortcut path blocks (if exists)
        shortcut_blocks = []
        if shortcut_was_executed:
            shortcut_blocks = ['Snn_Conv2d (1×1, shortcut)']
        
        info['basic_blocks'] = residual_blocks
        info['shortcut_blocks'] = shortcut_blocks
        info['data_flow'] = 'parallel_merge'
        info['merge_operation'] = 'Element-wise Adder (residual)'
        if shortcut_condition:
            info['shortcut_condition'] = " OR ".join(shortcut_condition)
    elif isinstance(module, Detect):
        # Use time_window from parameter if provided, otherwise mark as unknown
        tw_str = str(time_window) if time_window is not None else "5"  # Default to 5 if None (shouldn't happen after extract_model_parameters)
        info['basic_blocks'] = [
            'Snn_Conv2d (1×1, per scale)',
            'Reshape/Permute',
            f'Temporal Average (sum over time_window={tw_str})',
            'Sigmoid',
            'Coordinate Transform',
            'Multi-scale Concatenation'
        ]
        info['data_flow'] = 'sequential'
        if hasattr(module, 'nc'):
            info['parameters']['num_classes'] = module.nc
        if hasattr(module, 'na'):
            info['parameters']['num_anchors'] = module.na
        if hasattr(module, 'nl'):
            info['parameters']['num_detection_layers'] = module.nl
    elif hasattr(module, '__class__'):
        # Generic module - try to identify basic operations
        module_str = str(module)
        if 'Conv2d' in module_str or 'Snn_Conv2d' in module_str:
            info['basic_blocks'] = ['Convolution']
        elif 'BatchNorm' in module_str:
            info['basic_blocks'] = ['Batch Normalization']
        elif 'Pool' in module_str:
            info['basic_blocks'] = ['Pooling']
        elif 'Upsample' in module_str or 'Sample' in module_str:
            info['basic_blocks'] = ['Upsampling (Nearest Neighbor)']
        elif 'Concat' in module_str:
            info['basic_blocks'] = ['Concatenation']
        elif 'mem_update' in module_str:
            info['basic_blocks'] = ['mem_update (binary spiking)']
    
    return info


def extract_model_parameters(model, checkpoint_data=None):
    """
    Extract global parameters from the model and checkpoint.
    Parameters are read from the checkpoint file when available.
    time_window defaults to 5 if not found (matches hardcoded value in model).
    
    Args:
        model: PyTorch model
        checkpoint_data: Optional checkpoint dictionary
    
    Returns:
        dict: Dictionary with model parameters (time_window defaults to 5 if not found)
    """
    params = {
        'time_window': None,
        'threshold': None,
        'decay': None,
        'lens': None,
        'num_classes': None,
        'input_channels': None,
        'activation_mode': None
    }
    
    # First priority: Extract from checkpoint metadata if available
    if checkpoint_data is not None:
        # Check for explicitly stored parameters in checkpoint
        if 'time_window' in checkpoint_data:
            params['time_window'] = checkpoint_data['time_window']
        if 'threshold' in checkpoint_data:
            params['threshold'] = checkpoint_data['threshold']
        if 'decay' in checkpoint_data:
            params['decay'] = checkpoint_data['decay']
        if 'lens' in checkpoint_data:
            params['lens'] = checkpoint_data['lens']
        
        # Check if checkpoint has model config
        if 'model' in checkpoint_data:
            model_obj = checkpoint_data['model']
            
            # Extract from model.yaml if available
            if hasattr(model_obj, 'yaml'):
                yaml_config = model_obj.yaml
                if 'nc' in yaml_config:
                    params['num_classes'] = yaml_config['nc']
                if 'ch' in yaml_config:
                    params['input_channels'] = yaml_config['ch']
                # Check if parameters are stored in yaml
                if 'time_window' in yaml_config:
                    params['time_window'] = yaml_config['time_window']
                if 'threshold' in yaml_config:
                    params['threshold'] = yaml_config['threshold']
                if 'decay' in yaml_config:
                    params['decay'] = yaml_config['decay']
                if 'lens' in yaml_config:
                    params['lens'] = yaml_config['lens']
    
    # Second priority: Extract from loaded model structure
    # Try to get from model.yaml if available (from loaded model)
    if hasattr(model, 'yaml'):
        yaml_config = model.yaml
        if params['num_classes'] is None and 'nc' in yaml_config:
            params['num_classes'] = yaml_config['nc']
        if params['input_channels'] is None and 'ch' in yaml_config:
            params['input_channels'] = yaml_config['ch']
        # Check if parameters are stored in yaml
        if params['time_window'] is None and 'time_window' in yaml_config:
            params['time_window'] = yaml_config['time_window']
        if params['threshold'] is None and 'threshold' in yaml_config:
            params['threshold'] = yaml_config['threshold']
        if params['decay'] is None and 'decay' in yaml_config:
            params['decay'] = yaml_config['decay']
        if params['lens'] is None and 'lens' in yaml_config:
            params['lens'] = yaml_config['lens']
    
    # Extract num_classes from Detect layer (if not found in checkpoint)
    if params['num_classes'] is None:
        for m in model.modules():
            if isinstance(m, Detect):
                params['num_classes'] = m.nc
                break
    
    # Extract input channels from first Conv layer (if not found in checkpoint)
    if params['input_channels'] is None:
        for m in model.modules():
            if isinstance(m, (Conv, Conv_1, Conv_2, Conv_A)) and hasattr(m, 'conv'):
                if hasattr(m.conv, 'in_channels'):
                    params['input_channels'] = m.conv.in_channels
                    break
    
    # Determine activation mode from model structure
    has_silu = False
    has_binary = False
    for m in model.modules():
        if isinstance(m, mem_update):
            if m.act:
                has_silu = True
            else:
                has_binary = True
    
    if has_silu and not has_binary:
        params['activation_mode'] = 'SiLU'
    elif has_binary and not has_silu:
        params['activation_mode'] = 'Binary Spiking'
    elif has_silu and has_binary:
        params['activation_mode'] = 'Mixed (SiLU + Binary)'
    else:
        params['activation_mode'] = 'Unknown'
    
    # Set default time_window=5 if not found in checkpoint (matches hardcoded value in model)
    if params['time_window'] is None:
        params['time_window'] = 5
        LOGGER.debug(f"time_window not found in checkpoint, using default value: 5")
    
    return params


def trace_executed_modules(model, device='cpu', data_yaml=None, img_size=640):
    """
    Trace which modules are actually executed during a forward pass.
    Uses a real test data point from the dataset if YAML is provided, otherwise uses dummy data.
    
    Note: The model uses a hardcoded time_window=5 (from models/yolo.py and models/common.py).
    The forward method automatically duplicates the 4D input [B, C, H, W] to create a 5D tensor
    [time_window, B, C, H, W] = [5, B, C, H, W] before processing.
    
    Args:
        model: PyTorch model
        device: Device to run forward pass on (string like 'cpu' or 'cuda')
        data_yaml: Path to dataset YAML file (optional, if None uses dummy data)
        img_size: Input image size (default: 640, used as fallback if YAML not provided)
    
    Returns:
        set: Set of module IDs that were executed during forward pass
    """
    if data_yaml:
        LOGGER.info(f"  Tracing executed modules with forward pass using test data from {data_yaml}...")
    else:
        LOGGER.info(f"  Tracing executed modules with forward pass using dummy data (img_size={img_size})...")
    
    # Get time_window from model (hardcoded in models/yolo.py and models/common.py)
    # The model's forward method uses this to duplicate the input
    try:
        time_window = getattr(yolo_module, 'time_window', 5)
        LOGGER.info(f"  Using time_window={time_window} (from model definition)")
    except:
        time_window = 5  # Default fallback
        LOGGER.info(f"  Using time_window={time_window} (default, model definition not accessible)")
    
    # Handle device with fallback to CPU if CUDA not available
    if device == 'cuda' and not torch.cuda.is_available():
        LOGGER.warning(f"  CUDA not available, using CPU for tracing")
        device = 'cpu'
    
    device_obj = torch.device(device)
    
    # Get actual model (unwrap if needed)
    actual_model = model
    if hasattr(model, 'model'):
        actual_model = model.model
    elif hasattr(model, 'module'):
        actual_model = model.module
    
    # Set model to eval mode
    actual_model.eval()
    
    # Create a set to track executed modules
    executed_modules = set()
    hooks = []
    
    def register_hook(module):
        """Register forward hook to track module execution"""
        def hook(module, input, output):
            # Store module id (using id() to get unique identifier)
            executed_modules.add(id(module))
        return hook
    
    # Register hooks on all modules
    for name, module in actual_model.named_modules():
        hook = module.register_forward_hook(register_hook(module))
        hooks.append(hook)
    
    # Load test data from dataset if YAML provided, otherwise use dummy data
    try:
        if data_yaml:
            # Load dataset YAML
            data_dict = check_dataset(data_yaml, autodownload=False)
            
            # Get test path (prefer test, fallback to val, then train)
            test_path = data_dict.get('test') or data_dict.get('val') or data_dict.get('train')
            if not test_path:
                raise ValueError("No test/val/train path found in dataset YAML")
            
            # Create dataset (no augmentation, single class if needed)
            dataset = LoadImagesAndLabels(
                path=test_path,
                img_size=img_size,
                batch_size=1,
                augment=False,  # No augmentation for tracing
                single_cls=False,
                stride=32,
                prefix=''
            )
            
            if len(dataset) == 0:
                raise ValueError(f"No images found in test dataset: {test_path}")
            
            # Get first test image
            img, labels, img_path, shapes = dataset[0]
            
            # Convert to proper format: [B, C, H, W] and normalize to 0-1
            # The dataset returns img as [C, H, W] in 0-255 range
            # Model expects [B, C, H, W] and divides by 255 internally, so we need to pass 0-255 range
            if img.dim() == 3:
                img = img.unsqueeze(0)  # Add batch dimension: [C, H, W] -> [1, C, H, W]
            
            # Move to device
            test_input = img.to(device_obj)
            
            LOGGER.info(f"  Loaded test image from: {img_path}")
            LOGGER.info(f"  Test image shape: {test_input.shape} (will be duplicated to [5, B, C, H, W] by model)")
            
        else:
            # Use dummy data as fallback (in 0-255 range to match real data format)
            test_input = (torch.rand(1, 3, img_size, img_size) * 255).to(device_obj)
            LOGGER.info(f"  Using dummy input (shape: {test_input.shape}, range: 0-255)")
            LOGGER.info(f"  Input will be duplicated to [5, B, C, H, W] by model's forward method")
        
        # Run forward pass
        # Note: The model's forward() method automatically handles the time dimension:
        # - If input is 4D [B, C, H, W], it duplicates it time_window times to [5, B, C, H, W]
        # - If input is 5D [B, T, C, H, W], it extracts first frame and then duplicates
        # The time_window=5 is hardcoded in models/yolo.py and models/common.py
        with torch.no_grad():
            _ = actual_model(test_input)
        
        LOGGER.info(f"  Forward pass completed. {len(executed_modules)} modules executed.")
    except Exception as e:
        LOGGER.warning(f"  Forward pass failed: {e}. Will include all modules in diagram.")
        LOGGER.debug(traceback.format_exc())
        # If forward pass fails, return None to include all modules
        executed_modules = None
    finally:
        # Remove all hooks
        for hook in hooks:
            hook.remove()
    
    return executed_modules


def generate_architecture_diagram(model, output_file='model_architecture.txt', checkpoint_data=None, converted=True, trace_execution=True, device='cpu', data_yaml=None, img_size=640):
    """
    Generate a detailed layer-by-layer architecture diagram.
    Only includes modules that are actually executed during forward pass.
    
    Args:
        model: PyTorch model (should be the converted model with binary activations)
        output_file: Output file path for the diagram
        checkpoint_data: Optional checkpoint dictionary for metadata
        converted: Whether the model has been converted from SiLU to binary (default: True)
        trace_execution: If True, only show modules executed during forward pass (default: True)
        device: Device to run forward pass on (default: 'cpu')
        data_yaml: Path to dataset YAML file for loading test data (optional)
        img_size: Input image size for forward pass tracing (default: 640, used as fallback)
    """
    LOGGER.info(f"\n{colorstr('bold', 'Generating architecture diagram...')}")
    if converted:
        LOGGER.info(f"  Generating diagram from converted model (binary spiking activations)")
    
    # Trace executed modules if requested
    executed_module_ids = None
    if trace_execution:
        executed_module_ids = trace_executed_modules(model, device=device, data_yaml=data_yaml, img_size=img_size)
        if executed_module_ids is not None:
            LOGGER.info(f"  Will only include {len(executed_module_ids)} executed modules in diagram")
        else:
            LOGGER.warning(f"  Forward pass tracing failed. Will include all modules in diagram.")
    
    # Extract model parameters
    model_params = extract_model_parameters(model, checkpoint_data)
    
    # Update activation mode to reflect conversion status
    if converted and model_params['activation_mode'] == 'Binary Spiking':
        model_params['activation_mode'] = 'Binary Spiking (converted from SiLU)'
    
    diagram_lines = []
    diagram_lines.append("=" * 100)
    diagram_lines.append("EMS-YOLO MODEL ARCHITECTURE")
    diagram_lines.append("=" * 100)
    if trace_execution and executed_module_ids is not None:
        diagram_lines.append("NOTE: Only modules executed during forward pass are shown.")
        diagram_lines.append("=" * 100)
    diagram_lines.append("")
    diagram_lines.append("GLOBAL PARAMETERS:")
    
    # Use extracted values from checkpoint (time_window defaults to 5 if not found)
    time_window = model_params['time_window']  # Will be 5 if not found in checkpoint
    threshold = model_params['threshold']
    decay = model_params['decay']
    lens = model_params['lens']
    
    # Display parameters (time_window will always have a value, others may be None)
    diagram_lines.append(f"  - Time Window (T): {time_window}")
    diagram_lines.append(f"  - Threshold: {threshold if threshold is not None else 'Unknown (not found in checkpoint)'}")
    diagram_lines.append(f"  - Decay: {decay if decay is not None else 'Unknown (not found in checkpoint)'}")
    if lens is not None:
        diagram_lines.append(f"  - Lens (approximation parameter): {lens}")
    else:
        diagram_lines.append(f"  - Lens (approximation parameter): Unknown (not found in checkpoint)")
    if model_params['num_classes'] is not None:
        diagram_lines.append(f"  - Number of Classes: {model_params['num_classes']}")
    else:
        diagram_lines.append(f"  - Number of Classes: Unknown (not found in checkpoint)")
    if model_params['input_channels'] is not None:
        diagram_lines.append(f"  - Input Channels: {model_params['input_channels']}")
    else:
        diagram_lines.append(f"  - Input Channels: Unknown (not found in checkpoint)")
    diagram_lines.append(f"  - Activation: {model_params['activation_mode'] if model_params['activation_mode'] is not None else 'Unknown'}")
    diagram_lines.append("")
    diagram_lines.append("=" * 100)
    diagram_lines.append("LAYER-BY-LAYER DECOMPOSITION")
    diagram_lines.append("=" * 100)
    diagram_lines.append("")
    
    layer_idx = 0
    layer_info_list = []
    
    def traverse_model(module, name='model', depth=0, parent_info=None):
        nonlocal layer_idx
        
        # If tracing execution, skip modules that weren't executed
        if trace_execution and executed_module_ids is not None:
            module_id = id(module)
            # Always include the root model module
            if depth > 0 and module_id not in executed_module_ids:
                # Check if any child module was executed (for containers like Sequential)
                has_executed_child = False
                for child in module.children():
                    if id(child) in executed_module_ids:
                        has_executed_child = True
                        break
                
                # If no children were executed either, skip this module
                if not has_executed_child:
                    return
        
        indent = "  " * depth
        info = get_layer_info(module, name, depth, time_window=time_window, executed_module_ids=executed_module_ids)
        
        # Skip trivial modules but keep important ones
        is_important = isinstance(module, (
            Conv, Conv_A, Conv_1, Conv_2, BasicBlock, BasicBlock_1, BasicBlock_2, 
            Detect, nn.Sequential, mem_update, nn.BatchNorm2d, nn.BatchNorm3d
        ))
        
        if not is_important:
            if len(list(module.children())) == 0 and len(list(module.parameters())) == 0:
                if not isinstance(module, (nn.Identity,)):
                    return
        
        # Store layer info
        info['layer_idx'] = layer_idx
        layer_info_list.append(info)
        
        # Print layer information
        diagram_lines.append(f"{indent}[Layer {layer_idx}] {info['type']}")
        if name and name != 'model':
            diagram_lines.append(f"{indent}  Name: {name}")
        
        # Print parameters
        if info['parameters']:
            param_str = ", ".join([f"{k}={v}" for k, v in info['parameters'].items()])
            diagram_lines.append(f"{indent}  Parameters: {param_str}")
        
        # Print basic blocks with data flow
        if info['basic_blocks']:
            diagram_lines.append(f"{indent}  Components:")
            
            # Check if this is a parallel merge structure (like BasicBlock)
            if info.get('data_flow') == 'parallel_merge':
                # Show residual path
                diagram_lines.append(f"{indent}    Residual Path:")
                for i, block in enumerate(info['basic_blocks']):
                    if i == len(info['basic_blocks']) - 1:
                        diagram_lines.append(f"{indent}      └─ {block} →")
                    else:
                        diagram_lines.append(f"{indent}      ├─ {block} →")
                
                # Show shortcut path (if exists)
                if info.get('shortcut_blocks'):
                    diagram_lines.append(f"{indent}    Shortcut Path:")
                    # Show condition that triggered shortcut
                    if info.get('shortcut_condition'):
                        diagram_lines.append(f"{indent}      (Triggered by: {info['shortcut_condition']})")
                    for i, block in enumerate(info['shortcut_blocks']):
                        if i == len(info['shortcut_blocks']) - 1:
                            diagram_lines.append(f"{indent}      └─ {block} →")
                        else:
                            diagram_lines.append(f"{indent}      ├─ {block} →")
                else:
                    diagram_lines.append(f"{indent}    Shortcut Path: (Identity)")
                    # Show why shortcut is identity
                    if info.get('shortcut_condition'):
                        # This shouldn't happen, but just in case
                        pass
                    else:
                        diagram_lines.append(f"{indent}      (No shortcut needed: stride=1 AND channels match)")
                
                # Show merge operation
                merge_op = info.get('merge_operation', 'Element-wise Adder (residual)')
                diagram_lines.append(f"{indent}    └─ Merge: {merge_op}")
                
            elif info.get('data_flow') == 'sequential' or len(info['basic_blocks']) > 1:
                # Show sequential flow with arrows
                for i, block in enumerate(info['basic_blocks']):
                    if i == len(info['basic_blocks']) - 1:
                        diagram_lines.append(f"{indent}    └─ {block}")
                    else:
                        diagram_lines.append(f"{indent}    ├─ {block} →")
            else:
                # Single component, no arrow needed
                diagram_lines.append(f"{indent}    └─ {info['basic_blocks'][0]}")
        
        # Add separator for major layers
        if isinstance(module, (Conv, Conv_A, Conv_1, Conv_2, BasicBlock, BasicBlock_1, BasicBlock_2, Detect)):
            diagram_lines.append("")
        
        layer_idx += 1
        
        # Recursively process children
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            traverse_model(child_module, full_name, depth + 1, info)
    
    # Traverse the model - get actual model structure
    actual_model = model
    if hasattr(model, 'model'):
        # Model wrapped in DetectMultiBackend or similar
        actual_model = model.model
    elif hasattr(model, 'module'):
        # DDP wrapped model
        actual_model = model.module
    
    traverse_model(actual_model, 'model', 0)
    
    # Add layer summary statistics
    diagram_lines.append("")
    diagram_lines.append("=" * 100)
    diagram_lines.append("LAYER SUMMARY STATISTICS")
    diagram_lines.append("=" * 100)
    
    # Count layer types
    layer_counts = {}
    for info in layer_info_list:
        layer_type = info['type']
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
    
    diagram_lines.append("")
    diagram_lines.append("Layer Type Counts:")
    for layer_type, count in sorted(layer_counts.items()):
        diagram_lines.append(f"  {layer_type}: {count}")
    
    # Add data flow section
    diagram_lines.append("")
    diagram_lines.append("=" * 100)
    diagram_lines.append("DATA FLOW")
    diagram_lines.append("=" * 100)
    diagram_lines.append("")
    time_window_str = str(time_window)  # time_window always has a value (defaults to 5 if not in checkpoint)
    input_ch = model_params['input_channels']
    input_ch_str = str(input_ch) if input_ch is not None else "?"
    diagram_lines.append(f"  Input Image [B, {input_ch_str}, H, W]")
    diagram_lines.append("    ↓")
    diagram_lines.append(f"  [Temporal Replication] → [{time_window_str}, B, {input_ch_str}, H, W]")
    diagram_lines.append("    ↓")
    diagram_lines.append("  [Backbone Layers] → Feature Extraction")
    diagram_lines.append("    ├─ Conv_1: Initial convolution")
    diagram_lines.append("    ├─ BasicBlock_2: Residual blocks")
    diagram_lines.append("    └─ Feature maps at multiple scales")
    diagram_lines.append("    ↓")
    diagram_lines.append("  [Head Layers] → Feature Fusion")
    diagram_lines.append("    ├─ Upsampling")
    diagram_lines.append("    ├─ Concatenation")
    diagram_lines.append("    └─ Feature refinement")
    diagram_lines.append("    ↓")
    diagram_lines.append("  [Detect Layer] → Object Detection")
    diagram_lines.append("    ├─ 1×1 Convolution")
    diagram_lines.append("    ├─ Temporal Average")
    diagram_lines.append("    ├─ Sigmoid")
    diagram_lines.append("    └─ Coordinate Transform")
    diagram_lines.append("    ↓")
    nc = model_params['num_classes']
    nc_str = str(nc) if nc is not None else "?"
    diagram_lines.append(f"  Output [B, N_detections, {nc_str}+5]")
    diagram_lines.append("")
    diagram_lines.append("=" * 100)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(diagram_lines))
    
    LOGGER.info(f"Architecture diagram saved to: {colorstr('bold', output_file)}")
    
    # Also print to console
    print('\n'.join(diagram_lines))


def main():
    parser = argparse.ArgumentParser(description='Convert SiLU activations to binary spiking activation')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained model weights (.pt file)')
    parser.add_argument('--output', type=str, default=None, help='Output path for converted model (default: adds _binary to input name)')
    parser.add_argument('--diagram', type=str, default='model_architecture.txt', help='Output file for architecture diagram')
    parser.add_argument('--device', type=str, default='cuda', help='Device to load model on (cpu or cuda)')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset YAML file (e.g., data/coco.yaml). If provided, uses test data for forward pass tracing instead of dummy data')
    parser.add_argument('--no-trace', action='store_true', help='Disable forward pass tracing (show all modules, including unused ones)')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size for forward pass tracing (default: 640, used as fallback if --data not provided)')
    
    args = parser.parse_args()
    
    # Validate input file
    weights_path = Path(args.weights)
    if not weights_path.exists():
        LOGGER.error(f"Model file not found: {weights_path}")
        return
    
    # Set output path
    if args.output is None:
        output_path = weights_path.parent / f"{weights_path.stem}_binary{weights_path.suffix}"
    else:
        output_path = Path(args.output)
    
    LOGGER.info(f"{colorstr('bold', 'Loading model from:')} {weights_path}")
    
    # Load model and checkpoint
    checkpoint_data = None
    try:
        # Handle device selection with fallback to CPU
        if args.device == 'cuda' and not torch.cuda.is_available():
            LOGGER.warning(f"CUDA not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(args.device)
        
        # Try to load checkpoint to extract metadata
        try:
            checkpoint_data = torch.load(str(weights_path), map_location=device, weights_only=False)
            LOGGER.info(f"Checkpoint loaded, extracting metadata...")
        except:
            pass
        
        model = attempt_load(str(weights_path), map_location=device, inplace=True, fuse=False)
        
        # Handle ensemble models
        if hasattr(model, '__len__') and len(model) > 1:
            model = model[0]  # Take first model from ensemble
        
        LOGGER.info(f"Model loaded successfully")
        LOGGER.info(f"Model type: {type(model).__name__}")
        
    except Exception as e:
        LOGGER.error(f"Error loading model: {e}")
        return
    
    # STEP 1: Convert SiLU to binary spiking FIRST
    # This must happen before generating the architecture diagram
    # so that the diagram reflects the converted model (binary activations)
    LOGGER.info(f"\n{colorstr('bold', 'STEP 1: Converting SiLU activations to binary spiking...')}")
    model, stats = convert_silu_to_binary(model)
    
    # Print conversion statistics
    LOGGER.info(f"\n{colorstr('bold', 'Conversion Statistics:')}")
    LOGGER.info(f"  Total modules checked: {stats['total_modules_checked']}")
    LOGGER.info(f"  mem_update SiLU → Binary: {stats['mem_update_silu_to_binary']}")
    LOGGER.info(f"  Conv SiLU → Binary: {stats['conv_silu_to_binary']}")
    LOGGER.info(f"  Conv_A SiLU → Binary: {stats['conv_a_silu_removed']}")
    total_conversions = stats['mem_update_silu_to_binary'] + stats['conv_silu_to_binary'] + stats['conv_a_silu_removed']
    LOGGER.info(f"  Total conversions: {total_conversions}")
    
    # Verify conversion was successful
    if total_conversions == 0:
        LOGGER.warning(f"{colorstr('yellow', 'Warning:')} No SiLU activations were converted. Model may already use binary activations.")
    else:
        LOGGER.info(f"{colorstr('green', '✓')} Conversion completed: {total_conversions} activation(s) converted to binary spiking")
    
    # STEP 2: Generate architecture diagram from the CONVERTED model
    # The diagram will reflect binary spiking activations, not SiLU
    # This happens AFTER conversion, so the diagram shows the converted architecture
    LOGGER.info(f"\n{colorstr('bold', 'STEP 2: Generating architecture diagram from converted model...')}")
    LOGGER.info(f"  Note: Diagram reflects the model AFTER conversion (binary spiking activations)")
    trace_execution = not args.no_trace
    device_str = str(device)  # Convert torch.device to string
    
    # Validate data YAML if provided
    data_yaml = None
    if args.data:
        data_yaml = check_yaml(args.data) if args.data else None
        if data_yaml:
            LOGGER.info(f"  Using test data from: {data_yaml}")
        else:
            LOGGER.warning(f"  Data YAML file not found: {args.data}. Will use dummy data for tracing.")
    
    generate_architecture_diagram(model, args.diagram, checkpoint_data, converted=True, 
                                   trace_execution=trace_execution, device=device_str, 
                                   data_yaml=data_yaml, img_size=args.img_size)
    
    # STEP 3: Save the converted model
    LOGGER.info(f"\n{colorstr('bold', 'STEP 3: Saving converted model...')}")
    try:
        # Save as checkpoint
        checkpoint = {
            'model': model,
            'epoch': -1,
            'best_fitness': None,
            'converted_from_silu': True,
            'conversion_stats': stats
        }
        torch.save(checkpoint, output_path)
        LOGGER.info(f"Converted model saved to: {colorstr('bold', output_path)}")
        
        # Also save just the model state dict for easier loading
        state_dict_path = output_path.parent / f"{output_path.stem}_state_dict{output_path.suffix}"
        torch.save(model.state_dict(), state_dict_path)
        LOGGER.info(f"Model state dict saved to: {colorstr('bold', state_dict_path)}")
        
    except Exception as e:
        LOGGER.error(f"Error saving model: {e}")
        return
    
    LOGGER.info(f"\n{colorstr('bold', 'green', 'Conversion completed successfully!')}")
    LOGGER.info(f"  - Original model: {weights_path}")
    LOGGER.info(f"  - Converted model: {output_path}")
    LOGGER.info(f"  - Architecture diagram: {args.diagram}")


if __name__ == '__main__':
    main()

