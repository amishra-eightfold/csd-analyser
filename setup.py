"""Setup script for CSD Analyzer."""

from setuptools import setup, find_packages

setup(
    name="csd-analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.18.0",
        "python-pptx>=0.6.21",
        "openpyxl>=3.1.2",
        "nltk>=3.8.1",
        "scikit-learn>=1.3.0",
        "textblob>=0.17.1",
        "python-dotenv>=1.0.0",
        "streamlit>=1.31.0",
        "simple-salesforce>=1.12.5",
        "openai>=1.12.0"
    ],
    extras_require={
        "test": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "pytest-asyncio>=0.21.1"
        ]
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for analyzing customer support data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/csd-analyzer",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 