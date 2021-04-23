import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deep-net",
    version="0.1.0",
    author="Nadun De Silva",
    author_email="nadunrds@gmail.com",
    description="A Simple Library for Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nadundesilva/deep-net",
    project_urls={
        "Bug Tracker": "https://github.com/nadundesilva/deep-net/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
