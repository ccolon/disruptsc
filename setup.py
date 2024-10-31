from setuptools import setup, find_packages

setup(
    name="dsc",
    version="1.0.0",
    description="A spatial agent-based model to simulate the dynamics supply chains subject to disruptions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Celian Colon",
    author_email="celian.colon.2007@polytechnique.org",
    url="https://github.com/ccolon/dsc",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: CC BY-NC-ND 4.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "networkx=0.11.1",
        "geopandas=2.8.6",
        "PyYAML=0.2.5"
    ],
)