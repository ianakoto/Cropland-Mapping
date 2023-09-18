from setuptools import setup, find_packages

setup(
    name='croplandclassification',
    author='Ian Akoto',
    author_email='iancecilakoto@gmail.com',
    description='A package for pulling data from earth engine to gcp and supervised classification for cropland',
    packages=find_packages(),
    version='0.0.1',
    install_requires=[
    "earthengine-api",
    "numpy",
    "IPython"
    ],
)