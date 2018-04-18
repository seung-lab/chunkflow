version = '0.0.1'
from setuptools import setup, find_packages, Command
setup(
    name='chunkflow',
    description='Large Scale 3d Convolution Net Inference',
    license='Apache License 2.0',
    version=version,
    packages=find_packages(exclude=['tests*']),
    package_data={'': ['data/*' ]},
    include_package_data=True,
    zip_safe=False,
    entry_points="""
        [console_scripts]
        chunkflow=chunkflow.bin.cli:main
    """,
    install_requires=[
        'cloud-volume >= 0.17.0',
        'numpy',
        'Click'
        ],
    extras_require={
    },
    author='Seung Lab',
    url='https://github.com/seung-lab/chunkflow',
    download_url=(
        'https://github.com/seung-lab/chunkflow'
    )
)
