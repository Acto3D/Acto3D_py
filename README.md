# Acto3D_py
Acto3D_py is a Python package designed for transferring image data (***numpy.ndarray data***) to Acto3D, facilitating the integration of Python-based image processing workflows with Acto3D's 3D visualization tools.  
Note that functionality is currently limited, and remote input may experience slower communication speeds than local operations.


## Requirements
To use Acto3D_py, **Acto3D version 1.7.0** or newer must be installed on your MacOS system. Though Acto3D is a MacOS app, the package supports cross-platform remote interaction via network, allowing users on any operating system to work with Acto3D remotely.

## Installation
You can install Acto3D_py directly from the GitHub repository using pip with the following command:

```bash
pip install git+https://github.com/Acto3D/Acto3D_py.git
```

Once installed, you can import and use the package in your project as follows:

```Python
import Acto3D as a3d
```

### Acto3D External Connection Feature Notice

When using this package, be aware that **TCP-based data transmission is turned off by default**.  
To enable this feature, navigate through the menu: [Connection] > [Enable external connection]

Please note the following important points regarding this functionality:

- **Data transmission is not encrypted.** This functionality is primarily intended for operations within your local PC. If you use this feature, keep in mind that data sent or received is not protected.

- **Performance considerations for remote transmissions:** Sending and receiving data remotely may result in slower performance from remote PCs. 

- **Avoid exposing ports publicly.** Given that the data transfer is not encrypted, exposing your ports to the external network could pose significant security risks. Do not make your ports publicly accessible.

<img src="./img/enable_feature.png" width = 600>

## Sample code
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
# For an image with multiple channels,
# (664, 3, 1344, 1344)
# In this case, channel order is ZCYX.

# For a single-channel image, the shape 
# (664, 1344, 1344)
# In this case, the channel order is ZYX.

# Transfer image data to Acto3D.
# Specify the dimension order correctly. 
a3d.transferImage(image, order='ZCYX')

# Also you can set display ranges as [[min, max], ...].
# If you want to specify display ranges or gammas, please create a list with an entry for each channel.
a3d.transferImage(image, order='ZYX', display_ranges=[[500,2000]], gammas=[0.8])
a3d.transferImage(image, order='ZCYX', display_ranges=[[500,2000],[500,2000],[500,2000]])
a3d.transferImage(image, order='ZCYX', display_ranges=[[500,2000],[500,2000],[500,2000]], gammas=[1.0, 0.9, 1.0])

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# 
# If you're working with napari, layer.data is a numpy.ndarray,
# which means it can be transferred as follows:
a3d.transferImage(viewer.layers[0].data, order='ZYX')
a3d.transferImage(viewer.layers[0].data, order='ZCYX')

# It appears that frequently accessing layer.data in napari can lead to performance degradation. 
# If there is enough memory available, specifying as follows can sometimes improve speed.
a3d.transferImage(viewer.layers[0].data.copy(), order='ZYX')

# Furthermore, since display ranges can be obtained with layers.contrast_limits and gamma with layers.gamma, 
# you can use these values as well. 
# However, by using a3d.check_layers_structure(layers), 
# it's also possible to transfer and apply these values 
# (please refer to the following section for more details).
# 
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# Set voxel size for isotropic view
a3d.setVoxelSize(1.4, 1.4, 3.2, 'micron')

# Set zoom value
a3d.setScale(1.5)

# Set slice no
a3d.setSlice(450)

# Get the current image
current_image = a3d.getCurrentSlice()
```

If you want to transfer binary data to remote Acto3D, you should specify the remote address.
```Python
import Acto3D as a3d

a3d.config.host = '<YOUR REMOTE IP ADDRESS>'
# You can check the IP address at [Acto3D] > [Connection]
```


## Sample code for napari
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
a3d.transferLayers(viewer.layers)

# Or for specific layers:
a3d.transferLayers([viewer.layers[0],viewer.layers[1]])

```

### Compatible layer structure
For napari's image layers, the ndarray's shape must be ZCYX (where C is between 2-4) or ZYX for individual layers, or a list of ZYX layers. When grouping layers into a list, all layers must have identical shapes.