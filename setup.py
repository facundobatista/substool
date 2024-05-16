#!/usr/bin/env python3

# Copyright 2024 Facundo Batista
# Licensed under the Apache v2 License
# For further info, check https://github.com/facundobatista/substool

"""Setup script for substool."""

from setuptools import setup


VERSION = "2.0"


setup(
    name="substool",
    version=VERSION,
    author="Facundo Batista",
    author_email="facundo@taniquetil.com.ar",
    description="Tool to handle and fix subtitles general issues.",
    long_description=open("README.md", "rt", encoding="utf8").read(),
    url="https://github.com/facundobatista/substool",
    license="Apace-v2",
    packages=["substool"],
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": ["substool = substool.main:main"],
    },
    install_requires=["craft_cli"],
    python_requires=">=3",
)
