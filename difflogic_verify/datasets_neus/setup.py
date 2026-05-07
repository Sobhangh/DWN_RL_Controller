from setuptools import setup, find_packages

setup(
    name="dataset_neus",  # Package name
    version="0.1.0",  # Initial version
    packages=find_packages(),  # Auto-detect packages
    install_requires=[],  # Add dependencies if needed
    author="Your Name",
    author_email="fabian.kresse@gmail.com",
    description="A package for handling dataset_neus data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dataset_neus",  # Replace with repo URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Specify Python version
)
