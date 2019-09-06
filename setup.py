"""
Just a regular `setup.py` file.

Author: Nikolay Lysenko
"""


import os
from setuptools import setup, find_packages


current_dir = os.path.abspath(os.path.dirname(__file__))

description = 'Composition of music with reinforcement learning.'
with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rl-musician',
    version='0.0.1',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Nikolay-Lysenko/rl-musician',
    author='Nikolay Lysenko',
    author_email='nikolay-lysenco@yandex.ru',
    license='MIT',
    keywords='ai_music algorithmic_composition reinforcement_learning',
    packages=find_packages(),
    package_data={
        'rlmusician': ['default_config.yml', 'sinethesizer_presets.yml']
    },
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'gym', 'keras', 'numpy', 'PyYAML', 'sinethesizer', 'tensorflow'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Artistic Software',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ]
)
