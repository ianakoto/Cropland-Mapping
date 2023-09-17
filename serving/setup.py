from setuptools import setup, find_packages

setup(
    name='croplandclassification',
    author='Ian Akoto',
    author_email='iancecilakoto@gmail.com',
    description='A package for pulling data from earth engine to gcp for cropland classification',
    packages=find_packages(),
    version='0.0.1',
    install_requires=[
    "earthengine-api",
    "tensorflow",
    "numpy",
    "scikit-learn",
    "google-api-core",
    "google-auth",
    "apache-beam[gcp]",
    "IPython"
    ],
)