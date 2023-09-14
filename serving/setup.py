from setuptools import setup, find_packages

setup(
    name='cropland_classification',
    author='Ian Akoto',
    author_email='iancecilakoto@gmail.com',
    description='A package for pulling data from earth engine to gcp for cropland classification',
    packages=find_packages(),
    version='0.0.1',
    install_requires=[
    "earthengine-api==0.1.336",
    "tensorflow==2.9.0",
    "sklearn==0.24.2",
    "google-api-core==2.0.1",
    "google-auth==2.0.2",
    "apache-beam[gcp]==2.36.0",
    "IPython"
    ],
)