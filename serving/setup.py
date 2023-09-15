from setuptools import setup, find_packages

setup(
    name='croplandclassification',
    author='Ian Akoto',
    author_email='iancecilakoto@gmail.com',
    description='A package for pulling data from earth engine to gcp for cropland classification',
    packages=find_packages(),
    version='0.0.1',
    install_requires=[
    "earthengine-api==0.1.369",
    "tensorflow==2.13.0",
    "scikit-learn==1.3.0",
    "google-api-core==2.11.1",
    "google-auth==2.11.1",
    "apache-beam[gcp]==2.50.0",
    "IPython"
    ],
)