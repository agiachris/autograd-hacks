import setuptools


setuptools.setup(
    name="autograd-hacks",
    version="0.0.1",
    author="Christopher Agia",
    author_email="cagia@stanford.edu",
    description="Wrapper for batched autograd computation.",
    url="https://github.com/agiachris/autograd-hacks.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.6',
    install_requires=["torch"]
)