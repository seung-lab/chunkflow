from setuptools import setup, find_packages

version = '0.3.1'

with open("README.md", "r") as fh:
    long_description = fh.read()

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
        'click',
        'numpy',
        'cloud-volume',
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
