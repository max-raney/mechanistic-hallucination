# setup.py — at the project root (same level as src/, data/, tests/, README.md)

from setuptools import setup, find_packages

setup(
    # ────────────────────────────────────────────────────────────────────────────
    # Package metadata
    # ────────────────────────────────────────────────────────────────────────────
    name="mechanistic_interventions",        # the PyPI / import name
    version="0.1.0",                         # start with 0.1.0, bump on each release
    description="Mechanistic feature‐level control modules for LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Max Raney, Ruitian Yang, Sheng Qian, Keshav Ratra",
    author_email="maxraney@berkeley.edu",
    url="https://github.com/yourusername/mechanistic_interventions",  # adjust if you have a repo
    license="MIT",                           # change if you use a different license

    # ────────────────────────────────────────────────────────────────────────────
    # Tell setuptools where to find your code
    # ────────────────────────────────────────────────────────────────────────────
    package_dir={"": "src"},                 # root of your Python packages
    packages=find_packages(where="src"),     # find all packages under src/

    # ────────────────────────────────────────────────────────────────────────────
    # Runtime dependencies
    # ────────────────────────────────────────────────────────────────────────────
    install_requires=[
        "torch>=2.0",                        # or your minimum tested version
        "transformers>=4.0",
        "python-dotenv>=0.19.0",
        # add any others you use: scikit-learn, hydra-core, etc.
    ],

    # ────────────────────────────────────────────────────────────────────────────
    # Optional command-line entry points
    # ────────────────────────────────────────────────────────────────────────────
    entry_points={
        "console_scripts": [
            # after install, user can run `bench` in the shell:
            "bench=mechanistic_interventions.evaluation.benchmark:main",
        ],
    },

    # ────────────────────────────────────────────────────────────────────────────
    # Python version and classifiers
    # ────────────────────────────────────────────────────────────────────────────
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
