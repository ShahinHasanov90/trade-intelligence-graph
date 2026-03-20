"""Setup configuration for trade-intelligence-graph."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh.readlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="trade-intelligence-graph",
    version="1.0.0",
    author="Shahin Hasanov",
    author_email="shahin.hasanov@example.com",
    description="Graph-based network analysis for trade fraud ring detection and customs intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShahinHasanov90/trade-intelligence-graph",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Security",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-asyncio>=0.21",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "neo4j": [
            "neo4j>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "graph-intel=graph_intel.api.app:main",
        ],
    },
)
