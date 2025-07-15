from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent

setup(
    name='ivtdmetrics',
    version='0.1',
    author='Yiliang Chen',
    install_requires=['scikit-learn>=1.0.2',
                      'numpy>=1.21',
                      ],
    long_description='An extended metrics package for surgical triplet detection, \
    Significantly based on ivtmetrics by Chinedu Nwoye.',
) 
