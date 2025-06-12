from setuptools import setup, find_packages
import os

# Read version from _version.py
version_file = os.path.join(os.path.dirname(__file__), 'disruptsc', '_version.py')
with open(version_file) as f:
    exec(f.read())

setup(
    name="disruptsc",
    version=__version__,
    description="A spatial agent-based model to simulate the dynamics supply chains subject to disruptions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Celian Colon",
    author_email="celian.colon.2007@polytechnique.org",
    url="https://github.com/ccolon/disruptsc",
    packages=find_packages(),
    license="CC BY-NC-ND 4.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10,<3.12",
    install_requires=[
        # Core data processing
        "pandas>=1.3.0,<3.0",
        "numpy>=1.20.0,<2.0",
        
        # Geospatial processing
        "geopandas>=0.11.1,<1.0",
        "shapely>=1.8.0,<3.0",
        
        # Network analysis
        "networkx>=2.8.0,<4.0",
        
        # Scientific computing
        "scipy>=1.7.0,<2.0",
        
        # Configuration and utilities
        "PyYAML>=5.3.0,<7.0",
        "tqdm>=4.60.0,<5.0"
    ],
)
