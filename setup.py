"""Setup script for CSD Analyzer."""

from setuptools import setup, find_packages

setup(
    name="csd-analyser",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.0.0",
        "pandas>=1.3.0",
        "plotly>=5.0.0",
        "numpy>=1.20.0",
        "python-pptx>=0.6.21",
        "Pillow>=8.0.0",
        "wordcloud>=1.8.1",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "openai>=1.0.0",
        "scikit-learn>=0.24.0",
        "joblib>=1.0.0",
        "tiktoken>=0.3.0",
        "textblob>=0.15.3",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Streamlit application for analyzing customer support ticket data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/csd-analyser",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Business/Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 