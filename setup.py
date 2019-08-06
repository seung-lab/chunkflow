from setuptools import setup, find_packages
import chunkflow


with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]


with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='chunkflow',
    description='Large Scale 3d Convolution Net Inference',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    version=chunkflow.__version__,
    author='Jingpeng Wu',
    author_email='jingpeng.wu@gmail.com',
    packages=find_packages(exclude=[
        'tests', 
        'bin', 
        'docker', 
        'kubernetes'
    ]),
    url='https://github.com/seung-lab/chunkflow',
    install_requires=requirements,
    tests_require = [
    'pytest',
    ],
    entry_points='''
        [console_scripts]
        chunkflow=chunkflow.flow:cli
    ''',
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires='>=3.5',
)
