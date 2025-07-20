from setuptools import setup, find_packages

setup(
    name='EtabsToPython',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'comtypes',
        'numpy',
        'pandas',
        'matplotlib',
        'ipython'
    ],
    author='Patricio Palacios',
    description='Tools for converting ETABS models into Python objects for analysis and visualization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
