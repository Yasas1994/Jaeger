from setuptools import setup
from setuptools import find_namespace_packages

setup(name='jaeger',
      version='1.1.2',
      description='A alignmnet-free machine learning tool to identify phage genome sequences that are hidden within metagenomes',
      url='https://github.com/Yasas1994/Jaeger',
      author='Yasas Wijesekra',
      author_email='yasas.wijesekara@uni-greifswald.de',
      license='MIT',
      packages=find_namespace_packages(where="src"),
      package_dir={"": "src"},
      package_data={'jaegeraa.data': ["*.h5", "*.json"],
                },
      scripts=['bin/Jaeger', 'bin/Jaeger_parallel'],
      install_requires=[
        'pip <= 22.3.1',
        'tqdm >=4.64.0',
        'biopython >=1.78',
        'psutil >=5',
        'pandas >= 1.5',
        'kneed',
        'ruptures',
        'keras',
        'tensorflow',
        'seaborn',
        'matplotlib',
        'seaborn',
        'parasail'
      ]
      )
