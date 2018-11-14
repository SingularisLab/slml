import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="slml",
    version="0.0.1",
    author="Singularis Lab",
    author_email="info@singularis-lab.com",
    description="Singularis Lab Machine Learning templates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SingularisLab/slml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)