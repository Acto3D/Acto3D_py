import socket
import struct
import numpy as np
from .main import connect_to_server, finalize_connection, send_command_to_server


def getCurrentSliceNo():
    """
    Retrieves the current slice number.
    
    Returns:
    - An integer representing the current slice number.
    """
    s = connect_to_server(min_version='1.7.9')
    if(s):
        s.sendall(b'GPARA')
        
        cmd = 'getCurrentSliceNo'
        send_command_to_server(s, cmd)  
        
        # recieve data
        data = s.recv(4)
        value = None
        if len(data) == 4:
            value, = struct.unpack('I', data)
        
        s.close()
        return value
        
    else:
        print("Failed")
        
def getCurrentScale():
    """
    Retrieves the current scale.
    
    Returns:
    - A float representing the current scale.
    """
    s = connect_to_server(min_version='1.7.9')
    if(s):
        s.sendall(b'GPARA')
        
        cmd = 'getCurrentScale'
        send_command_to_server(s, cmd)  
        
        # recieve data
        data = s.recv(4)
        value = None
        if len(data) == 4:
            value, = struct.unpack('f', data)
        
        s.close()
        return value
        
    else:
        print("Failed")
        
def getCurrentZScale():
    """
    Retrieves the current z scale.
    
    Returns:
    - A float representing the current z scale.
    """
    s = connect_to_server(min_version='1.7.9')
    if(s):
        s.sendall(b'GPARA')
        
        cmd = 'getCurrentZScale'
        send_command_to_server(s, cmd)  
        
        # recieve data
        data = s.recv(4)
        value = None
        if len(data) == 4:
            value, = struct.unpack('f', data)
        
        s.close()
        return value
        
    else:
        print("Failed")
        

def getSliceImage(slice_no: int, target_size: int = 512, refresh_view: bool = False):
    """
    Retrieves the image for a specified slice.

    GPARAeters:
    - slice_no: The slice number for which to retrieve the image.
    - target_size: The target view size for the image. This will determine the size of the image.
    - refresh_view: Specifies whether the view in Acto3D should be refreshed.

    Returns:
    - np.ndarray: The image data for the specified slice.
    """
    
    s = connect_to_server(min_version='1.7.9')
    if(s):
        s.sendall(b'GPARA')
        
        cmd = 'getSliceImage'
        send_command_to_server(s, cmd)  
        
        args = struct.pack('II?', slice_no, target_size, refresh_view)
        s.sendall(args)
        
        # Calculate the total size of the image data
        total_size = target_size * target_size * 3
        received_size = 0
        image_data = bytearray()
        
        # Receive the image data
        while received_size < total_size:
            packet = s.recv(total_size - received_size)
            if not packet:
                break
            image_data.extend(packet)
            received_size += len(packet)
        
        image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((target_size, target_size, 3))
        
        s.close()
        
        return image_array
            
        
    else:
        print("Failed")
    
    pass

def getCurrentImage() -> np.ndarray:
    """
    Get current image from Acto3D.

    Returns:
    - np.ndarray: The image data.
    """
    
    s = connect_to_server(min_version='1.7.0')
    if(s):
        s.sendall(b'CURIM')  
        
        # Recieve size info
        size_data = s.recv(8)
        width, height = struct.unpack('II', size_data)
        
        # Calculate the total size of the image data
        total_size = width * height * 3
        received_size = 0
        image_data = bytearray()
        
        # Receive the image data
        while received_size < total_size:
            packet = s.recv(total_size - received_size)
            if not packet:
                break
            image_data.extend(packet)
            received_size += len(packet)
        
        image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 3))
        
        print("Image shape:", image_array.shape)
        finalize_connection(s)
        
        return image_array
        
        
        