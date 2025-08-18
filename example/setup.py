"""
Setup script for example plugin
"""

from setuptools import setup, find_packages

setup(
    name="monk-example",
    version="0.1.0",
    description="A",
    author="Monk",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/monk-example",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800"
        ]
    },
    entry_points={
        "monk.plugins": [
            "example = example.plugin:example_plugin"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="monk, cli, plugin, command",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/monk-example/issues",
        "Source": "https://github.com/yourusername/monk-example",
        "Documentation": "https://github.com/yourusername/monk-example/blob/main/README.md",
    }
)
