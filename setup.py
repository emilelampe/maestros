from setuptools import setup, find_packages

setup(
    name="maestros",
    version="0.1.1",
    long_description="A package for splitting multilabel datasets into train and test sets, while preserving the distribution of labels and keeping samples from the same group together. Includes a report and chart for visualizing the stratification. Multi-label stratificatin is done through the iterative-stratification package.",
    description="A package for splitting multilabel datasets into train and test sets, while preserving the distribution of labels and keeping samples from the same group together. Includes a report and chart for visualizing the stratification.",
    author="Emile Lampe",
    author_email="emilelampe@outlook.com",
    url="https://github.com/emilelampe/maestros",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.5",
        "pandas>=1.1.5",
        "scikit-learn>=0.24.2",
        "iterative-stratification>=0.1.6",
        "matplotlib>=3.3.4",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
)