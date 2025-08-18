from setuptools import setup, find_packages  
from pathlib import Path

setup(
    name='etabstopython',
    version='0.1.0',
    packages=find_packages(),  
    install_requires=[
        'comtypes',
        'numpy',
        'pandas',
        'matplotlib',
        'ipython',
    ],
    author='Patricio Palacios',
    description='Tools for converting ETABS models into Python objects for analysis and visualization',
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
