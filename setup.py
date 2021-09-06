from setuptools import setup, find_packages

requirements = [
    "jax==0.2.13",
    "jaxlib==0.1.67",
    "scipy==1.6.1",
    "numpy==1.20.1",
    "pytest==6.2.2",
    "matplotlib==3.3.4",
    "pandas==1.2.3",
]

setup(
    name="nvkm",
    author="Magnus Ross",
    packages=["nvkm"],
    description="Implementation of nonparametric Volterra kernels model",
    long_description=open("README.md").read(),
    install_requires=requirements,
    # python_requires="==3.8",
)
