from setuptools import setup

setup(
    name='dtrocr',
    version='0.3.2',
    packages=['dtrocr'],  # Explicitly list your package
    package_dir={'dtrocr': '.'},  # Map package to current directory
    install_requires=[
        'torch',
        'torchvision',
    ],
)