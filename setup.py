# setup.py
from setuptools import setup, find_packages

setup(
    name='dtrocr',
    version='0.1.0',  # Replace with your version
    packages=find_packages(),
    install_requires=[
        # List any dependencies here (e.g., torch, torchvision, Pillow)
        'torch',
        'torchvision',
    ]
)

setup(
    name='util',
    version='0.1.0',  # Replace with your version
    packages=find_packages(),
    install_requires=[
        # List any dependencies here (e.g., torch, torchvision, Pillow)
        'numpy',
        'torchvision',
    ]
)