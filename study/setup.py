from setuptools import setup, find_packages

setup(
    name="StatMS",
    version="1.0.1",
    packages=find_packages(),
    install_requires=[
        "seaborn==0.13.2",
        "PyQt5==5.15.11",
        "factor_analyzer==0.5.1",
        "matplotlib==3.7.2",
        "scipy==1.10.1",
        "scikit-learn==1.3.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "mplcursors==0.6"
        
    ],
)