from setuptools import setup, find_packages
import os, re

verfile = ".{}intand{}_version.py".format(os.sep, os.sep)
verstrline = open(verfile, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
version_str = mo.group(1)

data_files = [ 'datasets' + os.sep + 'data' + os.sep + '*' ]

setup(name='intand',
    version=version_str,
    description='INTAND: INTeractive ANomaly Detection',
    url='https://github.com/caisr-hh',
    author='Mohamed-Rafik Bouguelia - Center for Applied Intelligent Systems Research (CAISR)',
    author_email='mohbou@hh.se',
    license='MIT',
    packages=find_packages(),
    package_data={'intand':data_files},
    install_requires=['matplotlib>=2.1.0', 'numpy>=1.17.2', 'scipy>=1.0.0', 'scikit-learn>=0.20.0'],
    zip_safe=False,
    python_requires='>=3.4')
