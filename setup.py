
from setuptools import setup, find_packages
import re


def get_version():
    with open('Acto3D/__init__.py', 'r') as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")
    
setup(
    name='Acto3D', 
    version=get_version(),
    # version='0.1.3',
    install_requires=["numpy", "tqdm"],
    description="Preview ndarray in 3D with Acto3D (MacOS App)", 
    author='Naoki Takeshita', 
    packages=find_packages(),  
    license='MIT'  
)