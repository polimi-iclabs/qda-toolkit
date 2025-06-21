from setuptools import setup, find_packages

VERSION = '0.1.6'
DESCRIPTION = 'Quality data analysis toolkit'
LONG_DESCRIPTION = 'A package to create charts and models for statistical process control.'

setup(
    name='qda-toolkit',
    version=VERSION,
    author="IC Labs (Matteo Bugatti)",
    author_email="<matteo.bugatti@polimi.it>",
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='CC BY-NC-SA 4.0',
    url='https://github.com/polimi-iclabs/qda',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'Jinja2==3.1.2',
        'matplotlib==3.5.2',
        'numpy==1.21.5',
        'pandas==1.3.5',
        'scipy==1.7.3',
        'statsmodels==0.13.5',
    ],
)
