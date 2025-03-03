"""Setup configuration for the ice breaking game."""
from setuptools import setup, find_packages

setup(
    name="thinice",
    version="0.1.0",
    description="A hex-based ice breaking game",
    author="Eric",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pygame>=2.5.0",
        "watchdog>=3.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "thinice=thinice.__main__:main",
        ],
    },
) 