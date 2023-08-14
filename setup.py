from setuptools import setup
from setuptools import find_packages

setup(name='jaeger',
      version='1.1.2',
      description='A homology-free machine learning tool to identify phage genome sequences that are hidden within metagenomes',
      url='https://github.com/Yasas1994/Jaeger',
      author='Yasas Wijesekra',
      author_email='yasas.wijesekara@uni-greifswald.de',
      license='MIT',
      packages=find_packages(exclude=('*test*',)),
      package_data={'jaegeraa': ['data/WRes_1024.h5']},
      scripts=['bin/jaeger'],
      install_requires=[
    
        'tqdm >=4.64.0',
        'biopython >=1.78',
      ]
      )
