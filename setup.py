from setuptools import setup
from setuptools import find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="jaeger-bio",
    version="1.1.30",
    description="A quick and precise pipeline for detecting phages in sequence assemblies.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yasas1994/Jaeger",
    author="Yasas Wijesekra",
    author_email="yasas.wijesekara@uni-greifswald.de",
    license="MIT",
    data_files=[
        (".", ["LICENSE", "README.md"]),
    ],
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    package_data={"jaegeraa.data": ["*.h5", "*.json", "*.npy", "*.pkl", "*.fasta", ".env"],
                  },
    scripts=["bin/jaeger", "bin/jaeger_parallel"],
    python_requires='>=3.9, <3.12',
    install_requires=[
        "h5py >=3.8",
        "progressbar2 >=4.4.2",
        "psutil >=5",
        "pandas >= 1.5",
        "kneed >= 0.8.5",
        "numpy >= 1.24",
        "ruptures >= 1.1.9",
        "keras >= 2.15, <2.16",
        "matplotlib >= 3.7",
        "scikit-learn == 1.3.2",
        "parasail == 1.3.4 ",
        "pycirclize",
        "pyfastx >= 2"
    ],
    extras_require={
        "darwin-arm": ["tensorflow >=2.15, <2.16", "tensorflow-metal"],    # macOS only
        "gpu": ["tensorflow[and-cuda] >=2.15, <2.16"],     # Linux only
        "cpu": ["tensorflow >=2.15, <2.16"],   
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: Unix",
    ],

)

# python3 -m twine upload --repository testpypi dist/* --verbose