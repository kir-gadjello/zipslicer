from setuptools import setup

__version__ = "0.8.0"

python_min_version = (3, 8, 0)
version_range_max = 12

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="zipslicer",
    version=__version__,
    description="A library for efficient incremental access to tensors stored in PyTorch checkpoints",
    py_modules=["zipslicer"],
    install_requires=["torch >= 1.10.0"],
    extras_require={
        "dev": [
            "pytest >= 3.10",
        ]
    },
    # PyPI package information from pytorch
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
    ] + [
        "Programming Language :: Python :: 3.{}".format(i)
        for i in range(python_min_version[1], version_range_max)
    ],
    license="BSD-3",
    keywords="pytorch, machine learning",
    python_requires=">=3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kirill Gadjello",
    author_email="kirill.gadjello@protonmail.com",
    url="https://github.com/kir-gadjello/zipslicer",
)
