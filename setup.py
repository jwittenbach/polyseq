from setuptools import setup, find_packages

version = '0.0.0'

setup(
    name='polyseq',
    version=version,
    description='scRNAseq data analysis in Python',
    long_description='See https://github.com/jwittenbach/polyseq',
    author='jwittenbach',
    author_email='jason.wittenbach@gmail.com',
    url='https://github.com/jwittenbach/polyseq',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy',
        'sklearn',
        'pandas',
        'matplotlib',
        'phenograph'
    ],
    dependency_links = [
        'https://github.com/jacoblevine/PhenoGraph#egg=phenograph'
    ]
)
