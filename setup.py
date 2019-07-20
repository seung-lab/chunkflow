from setuptools import setup, find_packages

version = '0.3.1'

with open("README.md", "r") as fh:
    long_description = fh.read()

# This is a fix for numpy to work with setuptools
# https://github.com/numpy/numpy/blob/master/setup.py
# This is a bit hackish: we are setting a global variable so that the main
# numpy __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being used.
import builtins
builtins.__NUMPY_SETUP__ = True


setup(
    name='chunkflow',
    description='Large Scale 3d Convolution Net Inference',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    version=version,
    author='Jingpeng Wu',
    author_email='jingpeng.wu@gmail.com',
    packages=find_packages(exclude=['chunkflow.test']),
    url='https://github.com/seung-lab/chunkflow',
    install_requires=[
        'six>=1.12.0',
        'numpy>=1.16',
        'click',
        'cloud-volume>0.14.2',
        'scikit-image',
        'boto3',
        'h5py',
        'tifffile',
        'neuroglancer',
        'tinybrain',
        'zmesh'
    ],
    entry_points='''
        [console_scripts]
        chunkflow=chunkflow.flow:cli
    ''',
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
