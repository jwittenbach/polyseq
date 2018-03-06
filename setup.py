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
    install_requires=open('requirements.txt').read().split('\n'),
)
