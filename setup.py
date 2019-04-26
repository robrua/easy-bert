#!/usr/bin/env python
from pathlib import Path
import os

from setuptools import setup, find_packages


install_requires = [
    "numpy",
    "tensorflow-hub==0.4.0",
    "bert-tensorflow==1.0.1",
    "click"
]

# Hacky check for whether CUDA is installed
has_cuda = any("CUDA" in name.split("_") for name in os.environ.keys())
install_requires.append("tensorflow-gpu==1.13.1" if has_cuda else "tensorflow==1.13.1")

version_file = Path(__file__).parent.joinpath("easybert", "VERSION.txt")
version = version_file.read_text(encoding="UTF-8").strip()

setup(
    name="easybert",
    version=version,
    url="https://github.com/robrua/easy-bert",
    author="Rob Rua",
    author_email="robertrua@gmail.com",
    description="A Dead Simple BERT API (https://github.com/google-research/bert)",
    keywords=["BERT", "Natural Language Processing", "NLP", "Language Model", "Language Models", "Machine Learning", "ML", "TensorFlow", "Embeddings", "Word Embeddings", "Sentence Embeddings"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3"
    ],
    license="MIT",
    packages=find_packages(),
    entry_points={"console_scripts": ["bert=easybert.__main__:_main"]},
    zip_safe=True,
    install_requires=install_requires,
    include_package_data=True
)
