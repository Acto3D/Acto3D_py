import socket
import struct
import numpy as np
from .main import connect_to_server, finalize_connection, send_command_to_server



def setScale(scale: float):
    """
    Set scale value.
    """
    s = connect_to_server('1.7.9')
    if(s):
        s.sendall(b'SPARA')  
        cmd = 'setScale'
        send_command_to_server(s, cmd)  
        
        s.sendall(struct.pack('f', scale))
        s.close()
    
    else:
        print("Failed")
        
def setZScale(zscale: float):
    """
    Set z scale value.
    """
    s = connect_to_server('1.7.9')
    if(s):
        s.sendall(b'SPARA')  
        cmd = 'setZScale'
        send_command_to_server(s, cmd)  
        
        s.sendall(struct.pack('f', zscale))
        s.close()
    
    else:
        print("Failed")
        
def setSlice(slice: int):
    """
    Set slice no.
    """
    s = connect_to_server('1.7.9')
    if(s):
        s.sendall(b'SPARA')  
        cmd = 'setSlice'
        send_command_to_server(s, cmd)  
        
        s.sendall(struct.pack('I', slice))
        s.close()
    
    else:
        print("Failed")
        
        
def setVoxelSize(x: float, y: float, z: float, unit: str = ''):
    """
    Set voxel size.

    Parameters:
    - x, y, z: The XYZ resolution.
    - unit(str): e.g., 'mm', 'µm', 'px' (< 20 letters)
    """
    if(len(unit) > 20):
        print('unit string must be shorter than 20 letters.')
        return
    
    s = connect_to_server('1.7.9')
    if(s):
        s.sendall(b'SPARA')

        cmd = 'setVoxelSize'
        send_command_to_server(s, cmd)  
    
        # Padding unit to 20 bytes
        unit_padded = unit.encode().ljust(20, b'\0')
        
        voxel_data = struct.pack('fff', x, y, z) + unit_padded
        s.sendall(voxel_data)
        s.close()
    
    else:
        print("Failed")
        
        
        