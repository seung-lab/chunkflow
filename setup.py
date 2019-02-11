version = '0.0.1'
from setuptools import setup, find_packages, Command 

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='chunkflow',
    description='Large Scale 3d Convolution Net Inference',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    version="0.0.2",
    author='Jingpeng Wu',
    author_email='jingpeng.wu@gmail.com',
    packages=find_packages(exclude=['chunkflow/test*']),
    url='https://github.com/seung-lab/chunkflow',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
