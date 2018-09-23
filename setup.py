import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chunkflow",
    version="0.0.1",
    author="Jingpeng Wu",
    author_email="jingpeng.wu@gmail.com",
    description="convnet inference of 3D image stacks",
    long_description=long_description,
    long_description_type="text/markdown",
    url="https://github.com/seung-lab/chunkflow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
