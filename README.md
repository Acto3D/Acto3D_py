# Acto3D_py
Acto3D_py is a python package for transferring image data(**numpy.ndarray data**) to Acto3D.

This package enables integration between Python-based image processing workflows and Acto3D's advanced 3D visualization capabilities.  
However, it's important to note that the current set of functionalities available through this method is limited.  
Additionally, while input from a remote PC is supported, users should be aware that the communication speed may be much slower than local working.

## Requirements
Before using Acto3D_py, ensure that you have Acto3D version 1.7.0 or higher installed on your MacOS system.  
This is essential for compatibility with the package's features.  

While Acto3D is a MacOS application, it's worth noting that when used remotely, the package itself is not bound by the operating system of the client.  
This means that regardless of the client's operating system, as long as there is network accessibility, users can interact with Acto3D on a MacOS remotely, enabling cross-platform compatibility for remote operations.


# Installation Instructions for Acto3D_py
You can install Acto3D_py directly from the GitHub repository using pip with the following command:

```bash
pip install git+https://github.com/Acto3D/Acto3D_py.git
```

Once installed, you can import and use the package in your project as follows:

```Python
import Acto3D as a3d
```

# Sample code
```Python
import tifffile
import numpy as np
import Acto3D as a3d

# Load multichannel, zstack tif.
image = tifffile.imread('./image.tif')

# `image` is object of numpy.ndarray 
# type(image)
# numpy.ndarray

image.shape
# (664, 3, 1344, 1344)
# In this case, channel order is ZCYX.

# Transfer image data to Acto3D.
# Specify the dimension order correctly. 
a3d.transferImage(image, order='ZCYX')

# Also you can set display ranges as [[min, max], ...].
a3d.transferImage(image, order='ZCYX', display_ranges=[[500,2000],[500,2000],[500,2000]])

# Set voxel size for isotropic view
a3d.setVoxelSize(1.4, 1.4, 3.2, 'micron')

# Set zoom value
a3d.setScale(1.5)

# Set slice no
a3d.setSlice(450)

# Get the current image
current_image = a3d.getCurrentSlice()


```


# Sample code for napari
```Python
import tifffile
import napari

import Acto3D as a3d

# Check the default configuration. Modify values if necessary.
a3d.config.print_params()
# Example: 
# a3d.config.path_to_Acto3D = {Path to Acto3D.app}
# a3d.config.port = 63242

# Launch Acto3D
a3d.openActo3D()

# Import a multichannel z-stack tif image file.
image = tifffile.imread('./image.tif')

# Display it in the napari viewer.
viewer = napari.view_image(image)
# Alternatively, to split channels:
viewer = napari.view_image(
        image,
        channel_axis=1,
        name=["c1", "c2", "c3", "c4"]
        )


# Verify compatibility with Acto3D.
a3d.check_layers_structure(viewer.layers)
# If you've split into each channel, you can specify the layers as a list.
a3d.check_layers_structure([viewer.layers[0]], viewer.layers[1])

# If the ndarray shape in the napari layer is compatible, the output will resemble this:

#Layer: c1
#  Shape: (780, 768, 768)
#  Contrast Limits: [130.17482517482517, 255.0]
#  Gamma: 1
#----------------------------------------
#Layer: c2
#  Shape: (780, 768, 768)
#  Contrast Limits: [0, 255]
#  Gamma: 1
#----------------------------------------
#Multiple layers with the same 3D structure (depth, height, width).
#Compatible with Acto3D.


# You can now transfer them to Acto3D.
a3d.transfer(viewer.layers)
# Or for specific layers:
a3d.transfer([viewer.layers[0],viewer.layers[1]])

```

# Compatible layer structure
For napari's image layers, the ndarray's shape must be ZCYX (where C is between 2-4) or ZYX for individual layers, or a list of ZYX layers. When grouping layers into a list, all layers must have identical shapes.