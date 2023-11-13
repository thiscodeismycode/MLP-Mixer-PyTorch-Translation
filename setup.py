from setuptools import find_packages, setup

setup(
    name='mlp_mixer',
    version='0.0.0',
    author='Hyeonjin Kim',
    author_email='peaceful1@snu.ac.kr',
    url='https://github.com/thiscodeismycode/MLP-Mixer-PyTorch-Translation',
    install_requires=[
        'einops>=0.7.0',
        'numpy>=1.24.4',
        'PyYAML>=6.0.1',
        'six>=1.16.0',
        'tensorboard==2.14.0',
        'tensorboard-data-server==0.7.2',
        'torch==2.1.0',
        'torchvision==0.16.0',
        'packaging>=23.2'
    ],
    packages=find_packages(where='.'),
    license='MIT',
    description='PyTorch implementation of MLP Mixer',
    keywords=['PyTorch', 'Deep learning', 'Computer vision', 'Image classification'],
)
