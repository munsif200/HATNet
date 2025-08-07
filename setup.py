from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hatnet-rest",
    version="0.1.0",
    author="HATNet Team",
    author_email="",
    description="HATNet: Hierarchical Attention Transformer Network with ReST implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/munsif200/HATNet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gpu": [
            "dgl>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hatnet-rest=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yml", "configs/*.yaml"],
    },
    keywords="deep-learning, graph-neural-networks, spatial-temporal, tracking, materials-science",
    project_urls={
        "Bug Reports": "https://github.com/munsif200/HATNet/issues",
        "Source": "https://github.com/munsif200/HATNet",
    },
)