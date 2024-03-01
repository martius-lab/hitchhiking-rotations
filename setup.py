from setuptools import find_packages
from distutils.core import setup

INSTALL_REQUIRES = [
    "numpy",
    "pip",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pytictac",
    "roma",
    "black",
    "pyyaml",
    "hydra",
    "omegaconf",
    "tqdm",
    "hydra-core",
]

setup(
    name="hitchhiking_rotations",
    version="0.0.1",
    author="Rene Geist, Jonas Frey, Mikel Zhobro",
    author_email="jonfrey@ethz.ch",
    packages=find_packages(),
    python_requires=">=3.7",
    description="Code for: Position Paper: Learning with 3D rotations, a hitchhiker’s guide to SO(3)",
    install_requires=[INSTALL_REQUIRES],
    dependencies=[],
    dependency_links=[],
)
