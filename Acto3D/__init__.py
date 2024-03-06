# /Acto3D/__init__.py
from .config import config
import os
from .main import openActo3D, check_layers_structure, transferLayers, transferImage, setVoxelSize, setSlice, setScale, getCurrentSlice, is_version_compatible

__version__ = '0.1.5'


# オプショナル: パッケージをインポートした際に表示するメッセージ（デバッグや情報提供用）
print(f"Acto3D ({__version__}) has been imported.")
config.print_params()

if not os.path.exists(config.path_to_Acto3D):
    print(f"Acto3D not found at {config.path_to_Acto3D}")
    print("Please install Acto3D first.")
    print("https://github.com/Acto3D/Acto3D")
    print("Or please specify the path to Acto3D. a3d.config.path = PATH_TO_ACTO3D")