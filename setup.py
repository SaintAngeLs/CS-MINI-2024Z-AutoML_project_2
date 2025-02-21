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
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "featureflex=FeatureFlex.main:main",
        ],
    },  
    install_requires=[
        "alembic==1.14.0",
        "autofeat==2.1.3",
        "Boruta==0.4.3",
        "certifi==2024.12.14",
        "charset-normalizer==3.4.1",
        "cloudpickle==3.1.0",
        "colorlog==6.9.0",
        "contourpy==1.3.1",
        "cycler==0.12.1",
        "filelock==3.16.1",
        "flexcache==0.3",
        "flexparser==0.4",
        "fonttools==4.55.3",
        "fsspec==2024.12.0",
        "greenlet==3.1.1",
        "idna==3.10",
        "imbalanced-learn==0.13.0",
        "imblearn==0.0",
        "Jinja2==3.1.5",
        "joblib==1.4.2",
        "kagglehub==0.3.6",
        "kiwisolver==1.4.8",
        "lightgbm==4.5.0",
        "llvmlite==0.43.0",
        "Mako==1.3.8",
        "MarkupSafe==3.0.2",
        "matplotlib==3.10.0",
        "mpmath==1.3.0",
        "networkx==3.4.2",
        "numba==0.60.0",
        "numpy==1.26.4",
        "nvidia-cublas-cu12==12.4.5.8",
        "nvidia-cuda-cupti-cu12==12.4.127",
        "nvidia-cuda-nvrtc-cu12==12.4.127",
        "nvidia-cuda-runtime-cu12==12.4.127",
        "nvidia-cudnn-cu12==9.1.0.70",
        "nvidia-cufft-cu12==11.2.1.3",
        "nvidia-curand-cu12==10.3.5.147",
        "nvidia-cusolver-cu12==11.6.1.9",
        "nvidia-cusparse-cu12==12.3.1.170",
        "nvidia-nccl-cu12==2.21.5",
        "nvidia-nvjitlink-cu12==12.4.127",
        "nvidia-nvtx-cu12==12.4.127",
        "optuna==4.1.0",
        "packaging==24.2",
        "pandas==2.2.3",
        "pdfkit==1.0.0",
        "pillow==11.0.0",
        "Pint==0.24.4",
        "platformdirs==4.3.6",
        "pyparsing==3.2.1",
        "python-dateutil==2.9.0.post0",
        "pytz==2024.2",
        "PyYAML==6.0.2",
        "requests==2.32.3",
        "scikit-learn==1.5.0",
        "scipy==1.14.1",
        "setuptools==75.6.0",
        "shap==0.46.0",
        "six==1.17.0",
        "sklearn-compat==0.1.3",
        "slicer==0.0.8",
        "SQLAlchemy==2.0.36",
        "sympy==1.13.1",
        "threadpoolctl==3.5.0",
        "torch==2.5.1",
        "tqdm==4.67.1",
        "triton==3.1.0",
        "typing_extensions==4.12.2",
        "tzdata==2024.2",
        "urllib3==2.3.0",
        "xgboost==2.1.3",
    ],
)


