from setuptools import setup, find_packages, find_namespace_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines() 

setup(
    name="MLOPS-PROJECT",
    version="0.1",
    author="Rishav",
    packages=find_packages(),
    install_requires = requirements,
)