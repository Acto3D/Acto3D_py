# Acto3D/main.py
import subprocess
import os
import time
import numpy as np
import socket
import struct
from typing import Union, List, Tuple

from tqdm import tqdm  # Assuming tqdm is installed and imported for progress indication
import napari
from napari.components import LayerList

from .config import config  # 設定インスタンスをインポート


def is_Acto3D_running():
    """Check if Acto3D is running on macOS."""
    try:
        output = subprocess.check_output(['pgrep', 'Acto3D'], text=True)
        return bool(output.strip())
    
    except subprocess.CalledProcessError:
        return False

def openActo3D() -> bool:
    """Launch Acto3D."""
    if os.path.exists(config.path_to_Acto3D):
        if is_Acto3D_running():
            return True
        else:
            try:
                subprocess.Popen(['open', config.path_to_Acto3D])
                while not is_Acto3D_running():
                    print("Waiting for Acto3D launching...")
                    time.sleep(0.5)
                    
                time.sleep(2.0)
                return True
            
            except Exception as e:
                print(f"Failed to launch {config.path_to_Acto3D}: {e}")
                return False
    else:
        print("Please install Acto3D first.")
        print("https://github.com/Acto3D/Acto3D")
        return False
    
def is_version_compatible(version_string, min_version) -> bool:
    """
    Check if the given version string is compatible with a specified minimum version.

    Parameters:
    - version_string (str): The version to check, in the format 'major.minor.patch'.
    - min_version (str): The minimum required version, in the format 'major.minor.patch'.

    Returns:
    - bool: True if version_string is greater than or equal to min_version, False otherwise.
    """
    major, minor, patch = map(int, version_string.split('.'))
    min_major, min_minor, min_patch = map(int, min_version.split('.'))
    
    if (major, minor, patch) < (min_major, min_minor, min_patch):
        return False
    
    return True

def transfer(layers: Union[napari.layers.Layer, LayerList, list], remote: bool = False):
    """
    Transfer napari Layer(s) to Acto3D.
    
    Parameters:
    - layers: A single napari.layers.Layer object, a napari.components.layerlist.LayerList, 
              or a list containing napari.layers.Layer objects.
    - remote: Transfer to remote Acto3D. (Default is false.)
    """
    check_layers_structure(layers, onlyCheck=False, remote=remote)

def check_layers_structure(layers: Union[napari.layers.Layer, LayerList, list], onlyCheck: bool = True, remote: bool = False) -> str:
    """
    Checks the structure of napari Layer(s) and returns a message describing the structure.
    
    Parameters:
    - layers: A single napari.layers.Layer object, a napari.components.layerlist.LayerList, 
              or a list containing napari.layers.Layer objects.
    - onlyCheck: Just check whether layers are compatible with Acto3D.
    - remote: Transfer to remote Acto3D.
    
    Returns:
    - A string message describing the structure of the input layers.
    """
    
    # Convert input to a list of layers if necessary
    if isinstance(layers, napari.layers.Layer):
        layers = [layers]
    elif isinstance(layers, LayerList):
        layers = list(layers)
    
    # Check the number of layers and their shapes
    if len(layers) == 1:
        layer = layers[0]
        print(f"Layer: {layer.name}")
        if isinstance(layer, napari.layers.Image):
            print(f"  Shape: {layer.data.shape}")
            print(f"  Contrast Limits: {layer.contrast_limits}")
            print(f"  Gamma: {layer.gamma}")
            shape = layers[0].data.shape
            
            if len(shape) == 3:
                # ZYX image
                print("Single layer with 3D structure (depth, height, width).")
                print("Compatible with Acto3D.")
                
                if (onlyCheck == False):
                    if (remote == False):
                       openActo3D()
                    transferZYX(layer)
                
            elif len(shape) == 4 and shape[1] <= 4:
                print("Single layer with 4D structure (depth, channel, height, width).")
                print("Compatible with Acto3D.")
                
                if (onlyCheck == False):
                    if (remote == False):
                       openActo3D()
                    transferZCYX(layer)
                
            else:
                print("Not compatible with Acto3D.")
            
    else:
        for layer in layers:
            print(f"Layer: {layer.name}")
            if isinstance(layer, napari.layers.Image):
                print(f"  Shape: {layer.data.shape}")
                print(f"  Contrast Limits: {layer.contrast_limits}")
                print(f"  Gamma: {layer.gamma}")
            else:
                print("  This layer is not an Image layer and may not have shape, contrast_limits, or gamma attributes.")
            print("-" * 40)
            
        if len(layers) <= 4:
            shapes = [layer.data.shape for layer in layers if isinstance(layer, napari.layers.Image)]
            if all(shape == shapes[0] for shape in shapes) and len(shapes[0]) == 3:
                print( "Multiple layers with the same 3D structure (depth, height, width).")
                print("Compatible with Acto3D.")
                
                if (onlyCheck == False):
                    if (remote == False):
                       openActo3D()
                    transferLZYX(layers)
                
            else:
                print( "Not compatible with Acto3D.")
    
        else:
            # Acto3D only supports up to 4 channels
            print( "Not compatible with Acto3D.")
    
    
def adjust_image_to_uint8(data, min, max, gamma):
    """
    Adjust the image data to uint8 format applying contrast limits, normalization, and gamma correction.

    Parameters:
    - data (numpy.ndarray): The input image data.
    - min (float): The minimum intensity value for contrast adjustment.
    - max (float): The maximum intensity value for contrast adjustment.
    - gamma (float): The gamma value for gamma correction.

    Returns:
    - numpy.ndarray: The adjusted image data as an 8-bit unsigned integer array.
    """
    # Clip the data based on the contrast limits
    clipped_data = np.clip(data, min, max)
    # Normalize the data to the range [0, 1]
    normalized_data = (clipped_data - min) / (max - min)
    # Apply gamma correction
    gamma_adjusted_data = normalized_data ** gamma
    # Scale the data to the range [0, 255] and cast to uint8
    scaled_data = (gamma_adjusted_data * 255).astype(np.uint8)
    return scaled_data

def transferZYX(layer):
    """
    Transfer image data in ZYX order to a server via TCP/IP after ensuring version compatibility.

    Parameters:
    - layer: A napari image layer object containing the image data and metadata.

    This function establishes a TCP connection to a predefined host and port, checks the server's version
    compatibility, and if compatible, transfers the image data slice by slice in ZYX order after applying
    contrast adjustment and gamma correction. It communicates with the server using a simple protocol
    involving start, data transfer, and end signals.
    """
    
    depth, height, width = layer.data.shape
    min, max = layer.contrast_limits
    gamma = layer.gamma

    # TCPソケットオブジェクトの作成
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((config.host, config.port))
        
        # 1. データ通信の開始信号
        start_signal = b'START'
        s.sendall(start_signal)
        
        # サーバーからバージョン情報を受信
        version_info = s.recv(1024).decode('utf-8')
        print(f"Received version info from server: {version_info}")
        
        # バージョン情報が互換性があるか確認
        if is_version_compatible(version_info, min_version='1.7.0'):
            s.sendall(b'ZYX__')  # バージョン互換性があることをサーバーに通知
            
            # 2. 送信する幅、高さ、枚数の情報
            s.sendall(struct.pack('III', depth, height, width))  # 幅、高さ、枚数をバイト列で送信
            
            
            # 3. 実際に1スライスずつのデータ送信
            for z in tqdm(range(depth)):
                slice_data = layer.data[z, :, :]
                adjusted_data_uint8 = adjust_image_to_uint8(slice_data, min, max, gamma)
                data_bytes = adjusted_data_uint8.tobytes()
                # スライスデータを送信
                s.sendall(data_bytes)
            
            # 4. 終了信号
            end_signal = b'END'
            s.sendall(end_signal)
        
        else:
            s.sendall(b"ERROR")  # エラーコードをサーバーに送信
            # ここで通信終了などのエラーハンドリング

def transferLZYX(layers):
    depth, height, width = layers[0].data.shape
    channel = len(layers)
    
    
    min = [layer.contrast_limits[0] for layer in layers]
    max = [layer.contrast_limits[1] for layer in layers]
    gamma = [layer.gamma for layer in layers]

    # TCPソケットオブジェクトの作成
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((config.host, config.port))
        
        # 1. データ通信の開始信号
        start_signal = b'START'
        s.sendall(start_signal)
        
        # サーバーからバージョン情報を受信
        version_info = s.recv(1024).decode('utf-8')
        print(f"Received version info from server: {version_info}")
        
        # バージョン情報が互換性があるか確認
        if is_version_compatible(version_info, min_version='1.7.0'):
            s.sendall(b'LZYX_')  # バージョン互換性があることをサーバーに通知
            
            # 2. 送信する幅、高さ、枚数の情報
            s.sendall(struct.pack('IIII', channel, depth, height, width))  
            
            # 3. 実際に1スライスずつのデータ送信
            for z in tqdm(range(depth)):
                concatenated = np.stack([adjust_image_to_uint8(layer.data[z, :, :], min[i], max[i], gamma[i]) for (i, layer) in enumerate(layers)], axis=0)
                data_bytes = concatenated.tobytes()
                # スライスデータを送信
                s.sendall(data_bytes)
            
            # 4. 終了信号
            end_signal = b'END'
            s.sendall(end_signal)
        
        else:
            s.sendall(b"ERROR")  # エラーコードをサーバーに送信
            # ここで通信終了などのエラーハンドリング

def transferZCYX(layer):
    depth, channel, height, width = layer.data.shape
    min, max = layer.contrast_limits
    gamma = layer.gamma

    # TCPソケットオブジェクトの作成
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((config.host, config.port))
        
        # 1. データ通信の開始信号
        start_signal = b'START'
        s.sendall(start_signal)
        
        # サーバーからバージョン情報を受信
        version_info = s.recv(1024).decode('utf-8')
        print(f"Received version info from server: {version_info}")
        
        # バージョン情報が互換性があるか確認
        if is_version_compatible(version_info, min_version='1.7.0'):
            s.sendall(b'ZCYX_')  # バージョン互換性があることをサーバーに通知
            
            # 2. 送信する幅、高さ、枚数の情報
            s.sendall(struct.pack('IIII', depth, channel, height, width))  # 幅、高さ、枚数をバイト列で送信
            
            
            # 3. 実際に1スライスずつのデータ送信
            for z in tqdm(range(depth)):
                slice_data = layer.data[z,:, :, :]
                adjusted_data_uint8 = adjust_image_to_uint8(slice_data, min, max, gamma)
                data_bytes = adjusted_data_uint8.tobytes()
                # スライスデータを送信
                s.sendall(data_bytes)
            
            # 4. 終了信号
            end_signal = b'END'
            s.sendall(end_signal)
        
        else:
            s.sendall(b"ERROR")  # エラーコードをサーバーに送信
            # ここで通信終了などのエラーハンドリング

def transferImage(image: np.ndarray, order:str = 'ZCYX', display_ranges:List[Tuple[int, int]]=[]):
    """
    Transfers image data to Acto3D.

    This function transfers a specified 4- or 3-dimensional NumPy array image to Acto3D. 
    The image is rearranged according to the specified order and normalized based on the display range for each channel.

    Parameters:
    - image: np.ndarray
        The image data to be transferred. A 4-dimensional or 3-dimensional NumPy array is expected.
    - order: str, optional
        The order of axes in the image data. The default is 'ZCYX'.
    - display_ranges: List[Tuple[int, int]], optional
        A list of display ranges for each channel. Each tuple is in the form of (min, max).
        If not specified, [0, 255] is used for uint8 images, and [0, 65535] for others.

    Returns:
    - None
    """
    print('=== Transfer image data to Acto3D ===')
    print('Image shape:', image.shape)
    print('Data type:', image.dtype)
    print('Order:', order)
    
    
    if(len(order) != len(image.shape)):
        print('Please specify order correctly.')
        return
    
    order_indices = None
    x_index = None
    y_index = None
    z_index = None
    c_index = None
    
    if len(order) == 3:
        order_indices = {axis: order.index(axis) for axis in "XYZ"}
        x_index = order_indices["X"]
        y_index = order_indices["Y"]
        z_index = order_indices["Z"]
        
    elif len(order) == 4:
        order_indices = {axis: order.index(axis) for axis in "XYZC"}
        x_index = order_indices["X"]
        y_index = order_indices["Y"]
        z_index = order_indices["Z"]
        c_index = order_indices["C"]
        
    
    print('X channel:', x_index)
    print('Y channel:', y_index)
    print('Z channel:', z_index)
    print('C channel:', c_index)
    
    
    depth, height, width = image.shape[z_index], image.shape[y_index], image.shape[x_index]
    
    channel = 1 if c_index is None else image.shape[c_index]
    
    print(f'Width: {width}, Height: {height}, Depth: {depth}, Channel: {channel}')
    
    
    displayRanges = []
    
    if(len(display_ranges) == 0):
        for c in range(channel):
                displayRanges.append([0, 255 if image.dtype == np.uint8 else 65535])
                
        print("Use default display ranges:", displayRanges)
    else:
        if(len(display_ranges) == channel):
            displayRanges = display_ranges
            print("Use user defined display ranges:", displayRanges)
            
        else:
            print("Invalid display ranges:", display_ranges)
            return
    
    
    print('\nStart connection.')
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((config.host, config.port))
        
        # Send start signal
        start_signal = b'START'
        s.sendall(start_signal)
        
        # Recieve version info
        version_info = s.recv(1024).decode('utf-8')
        print(f"Received Acto3D version: {version_info}")
        
        if is_version_compatible(version_info, min_version='1.7.0'):
            rearranged_indices = []
            rearranged_indices2 = []
            
            if(c_index == None):    
                s.sendall(b'ZYX__')  
                s.sendall(struct.pack('III', depth, height, width))  
                
                rearranged_indices = [y_index, x_index]
                    
                if(x_index > z_index):
                    x_index -= 1
                if(y_index > z_index):
                    y_index -= 1
                rearranged_indices2 = [y_index, x_index]
                
            else:
                s.sendall(b'ZCYX_')  
                s.sendall(struct.pack('IIII', depth, channel, height, width))  
                
                rearranged_indices = [c_index, y_index, x_index]
                    
                if(c_index > z_index):
                    c_index -= 1
                if(x_index > z_index):
                    x_index -= 1
                if(y_index > z_index):
                    y_index -= 1
                rearranged_indices2 = [c_index, y_index, x_index]
                
            
            normalized_image = np.zeros((channel, height, width), dtype=np.float32) if c_index is not None else np.zeros((height, width), dtype=np.float32)
            scaled_data = np.zeros((channel, height, width), dtype=np.uint8) if c_index is not None else np.zeros((height, width), dtype=np.uint8)
            print("Re-arranged shape:" ,normalized_image.shape)
            
            for z in tqdm(range(depth)):
                slice_data = np.take(image, z, axis=z_index)
                
                rearranged_data = np.transpose(slice_data, rearranged_indices2)
                
                for c in range(channel):  # チャンネル数
                    min_val, max_val = displayRanges[c]
                    
                    # Normalize image data with display ranges.
                    # Clip the data within 0-255, 8 bits image.
                    if(c_index == None):
                        normalized_image[:, :] = (rearranged_data[:, :].astype(np.float32) - min_val) / (max_val - min_val)
                        scaled_data[:, :] = (np.clip(normalized_image[:, :] * 255, 0, 255)).astype(np.uint8)
                    else:
                        normalized_image[c, :, :] = (rearranged_data[c, :, :].astype(np.float32) - min_val) / (max_val - min_val)
                        scaled_data[c, :, :] = (np.clip(normalized_image[c, :, :] * 255, 0, 255)).astype(np.uint8)
                    
                    # gamma_adjusted_data = normalized_data ** gamma
    
                    
                
                s.sendall(scaled_data.tobytes())
                
                
            end_signal = b'END'
            s.sendall(end_signal)
            
            print('=== Transfer Completed ===')
            
        else:
            print("Error")
            print("Please update Acto3D.")
            s.sendall(b"ERROR")  
        
    return