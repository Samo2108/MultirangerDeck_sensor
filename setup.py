from setuptools import setup, find_packages

setup(
    name="MultirangerDeck",
    version="1.0.0",
    description="Custom Crazyflie environments and Multiranger sensors for Isaac Lab.",
    author="Alexandru Zaporojanu, Luca Samorì and Tommaso Tieri",
    packages=find_packages(),
    install_requires=[
        # We assume Isaac Lab is already installed, so we don't list it here to avoid conflicts.
        "matplotlib",
        "imageio",
    ],
)