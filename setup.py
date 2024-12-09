from setuptools import setup, find_packages

setup(name='CN2Simul',
    version='0.1',
    description='Computational Neuroscientists Comprehensive Neuronal Simulator including',
    url='https://github.com/NICALab/CN2-Simulator',
    author='Minho Eom',
    install_requires=['numpy', 'tqdm', 'pyyaml'],
    author_email='djaalsgh214@gmail.com',
    packages=find_packages(include=["SimulMotif", "SimulMotif.*"]),
    zip_safe=False)