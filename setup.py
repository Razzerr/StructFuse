#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="structfuse",
    version="0.1.0",
    description="StructFuse: Retrieval-Based Structural Fusion for Protein Contact Prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Michał Budnik",
    author_email="",
    url="https://github.com/Razzerr/StructFuse",
    license="MIT",
    packages=find_packages(exclude=["tests", "scripts", "notebooks", "configs", "data", "figures"]),
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies
        "torch>=2.8.0",
        "torchvision>=0.15.0",
        "lightning>=2.0.0",
        "torchmetrics>=0.11.4",
        # Hydra config management
        "hydra-core==1.3.2",
        "hydra-colorlog==1.2.0",
        "hydra-optuna-sweeper==1.2.0",
        # Loggers
        "neptune==1.14.0",
        # Utilities
        "rootutils",
        "pre-commit",
        "rich",
        "pytest",
        # Data science & bioinformatics
        "biopython==1.85",
        "matplotlib==3.10.7",
        "faiss-cpu==1.12.0",
        "numba==0.62.1",
        "parasail==1.3.4",
        "blosum==2.0.3",
        "tqdm==4.67.1",
        "pillow==11.3.0",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pytest",
            "pytest-cov",
        ],
    },
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "structfuse-train = src.train:main",
            "structfuse-eval = src.eval:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="protein contact-prediction deep-learning ESM2 bioinformatics structural-biology",
)
