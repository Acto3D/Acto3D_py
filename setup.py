
from setuptools import setup, find_packages

setup(
    name='Acto3D', 
    version="0.1", 
    install_requires=["numpy", "socket", "struct", "tqdm"],
    description="Preview ndarray in 3D with Acto3D (MacOS App)", 
    author='Naoki Takeshita', 
    packages=find_packages(),  
    license='MIT'  
)