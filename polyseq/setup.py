from setuptools import setup

version = '0.0.0'

setup(
    name='polyseq',
    version=version,
    description='RNA-seq data analysis in Python',
    long_description='See https://github.com/jwittenbach/polyseq',
    author='jwittenbach',
    author_email='jason.wittenbach@gmail.com',
    url='https://github.com/jwittenbach/polyseq',
    install_requires=open('requirements.txt').read().split('\n'),
)
