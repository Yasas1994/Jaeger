from setuptools import setup
from setuptools import find_namespace_packages

setup(name='jaeger-bio',
      version='1.1.3',
      description='A alignmnet-free machine learning tool to identify phage genome sequences that are hidden within metagenomes',
      url='https://github.com/Yasas1994/Jaeger',
      author='Yasas Wijesekra',
      author_email='yasas.wijesekara@uni-greifswald.de',
      license='MIT',
      data_files = [(".", ["LICENSE", "README.md"])],
      packages=find_namespace_packages(where="src"),
      package_dir={"": "src"},
      package_data={'jaegeraa.data': ["*.h5", "*.json", ".npy", ".pkl"]},
      scripts=['bin/Jaeger', 'bin/Jaeger_parallel'],

      install_requires=[
        'pip <= 22.3.1',
        'tqdm >=4.64.0',
        'biopython >=1.78',
        'psutil >=5',
        'pandas >= 1.5',
        'kneed >= 0.8.5',
        'ruptures >= 1.1.9',
        'keras >= 2.10',
        'tensorflow >= 2.10',
        'seaborn >= 0.12.2',
        'matplotlib >= 3.7',
        'parasail >= 1.3.4'
      ],

      classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: OS Independent",
        ]
)
