import os
from setuptools import setup, find_packages

VERSION_FILE = "VERSION"

def get_version():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r") as f:
            return f.read().strip()
    return "0.1.0"

def increment_version(version):
    major, minor, patch = map(int, version.split("."))
    patch += 1
    return f"{major}.{minor}.{patch}"

current_version = get_version()
new_version = increment_version(current_version)

with open(VERSION_FILE, "w") as f:
    f.write(new_version)

setup(
    name="FeatureFlex",  
    version=new_version,
    author="SaintAngeLs",
    author_email="info@itsharppro.com",
    description="An AutoML project with various machine learning capabilities.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SaintAngeLs/CS-MINI-2024Z-AutoML_project_2",
    packages=find_packages(where="src"),  
    package_dir={"": "src"},  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "featureflex=FeatureFlex.main:main",
        ],
    },
)
