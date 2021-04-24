"""Copyright (c) 2021, Deep Net. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deep-net",
    version="0.2.0",
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
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=["numpy >= 1.19.5"],
    extras_require={"dev": ["pytest", "black"]},
)
