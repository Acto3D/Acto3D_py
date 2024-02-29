# /Acto3D/__init__.py
from .config import config
from .main import openActo3D, check_layers_structure, transfer, transferImage

__version__ = '0.1.3'


# オプショナル: パッケージをインポートした際に表示するメッセージ（デバッグや情報提供用）
print(f"Acto3D ({__version__}) has been imported.")
config.print_params()