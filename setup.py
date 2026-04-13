from pathlib import Path

from setuptools import find_packages, setup

VERSION = '0.2.0'
DESCRIPTION = 'Quality data analysis toolkit'
LONG_DESCRIPTION = 'A package to create charts and models for statistical process control.'
BASE_DIR = Path(__file__).resolve().parent

setup(
    name='qda-toolkit',
    version=VERSION,
    author="IC Labs (Matteo Bugatti)",
    author_email="<matteo.bugatti@polimi.it>",
    description=DESCRIPTION,
    long_description=(BASE_DIR / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    license='CC BY-NC-SA 4.0',
    url='https://github.com/polimi-iclabs/qda-toolkit',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'Jinja2>=3.1,<4',
        'matplotlib>=3.5,<4',
        'numpy>=1.21.5,<3',
        'pandas>=1.3.5,<3',
        'scipy>=1.7.3,<2',
        'statsmodels>=0.13.5,<1',
    ],
)
