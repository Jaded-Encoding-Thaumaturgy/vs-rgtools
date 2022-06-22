#!/usr/bin/env python3

import setuptools

with open("requirements.txt") as fh:
    install_requires = fh.read()

name = 'vsrgtools'
version = "0.1.3"
release = "0.1.3"

setuptools.setup(
    name=name,
    version=version,
    author="Irrational Encoding Wizardry",
    author_email="wizards@encode.moe",
    maintainer="Setsugen no ao",
    maintainer_email="setsugen@setsugen.dev",
    packages=["vsrgtools"],
    project_urls={
        "Source Code": 'https://github.com/Irrational-Encoding-Wizardry/vs-rgtools',
        "Documentation": 'https://vsrgtools.encode.moe/en/latest/',
        "Contact": 'https://discord.gg/qxTxVJGtst',
    },
    package_data={
        'vsrgtools': ['py.typed'],
    },
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10'
)
