"""
Setup script for StringSight package.
"""

from setuptools import setup, find_packages

with open("README_ABSTRACTION.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stringsight",
    version="0.1.0",
    author="Lisa Dunlap",
    description="Explain Large Language Model Behavior Patterns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "pydantic>=1.8.0",
        "litellm>=1.0.0",
        "sentence-transformers>=2.0.0",
        "hdbscan>=0.8.0",
        "umap-learn>=0.5.0",
        "wandb>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
        ],
        "viz": [
            "gradio>=4.0.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stringsight=stringsight.cli:main",
        ],
    },
) 