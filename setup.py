from setuptools import setup, find_packages
import os
import sys
import re
import setuptools
from shutil import move

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PACKAGE_DIR, 'requirements.txt')) as f:
    install_requires = f.read().splitlines()
    install_requires = [l for l in install_requires if not l.startswith('#')]

with open(os.path.join(PACKAGE_DIR, 'tests/requirements.txt')) as f:
    tests_require = f.read().splitlines()
    tests_require = [l for l in tests_require if not l.startswith('#')]


with open("README.md", "r") as fh:
    long_description = fh.read()

VERSIONFILE = os.path.join(PACKAGE_DIR, "chunkflow/__version__.py")
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." %
                       (VERSIONFILE, ))


setup(
    name='chunkflow',
    description='Composable image chunk operators to create pipeline for distributed computation.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    version=version,
    author='Jingpeng Wu',
    author_email='jingpeng.wu@gmail.com',
    packages=find_packages(exclude=['bin', 'docker', 'kubernetes']),
    url='https://github.com/seung-lab/chunkflow',
    install_requires=install_requires,
    tests_require=tests_require,
    entry_points='''
        [console_scripts]
        chunkflow=chunkflow.flow.flow:main
    ''',
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3',
    zip_safe=False,
)
