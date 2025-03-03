from setuptools import setup, find_packages

setup(
    name='eval_lib',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "statsmodels",
    ],
)