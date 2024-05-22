from setuptools import setup, find_packages
import os

VERSION = '0.0.1'
DESCRIPTION = 'Quality data analysis with control charts'
LONG_DESCRIPTION = 'A package that allows to create charts for statistical process control tasks.'

# Setting up
setup(
    name="qda",
    version=VERSION,
    author="IC Labs (Matteo Bugatti)",
    author_email="<matteo.bugatti@polimi.it>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'statsmodels', 'scipy', 'matplotlib'],
    keywords=['python', 'statistics', 'statistical process control', 'quality control'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
