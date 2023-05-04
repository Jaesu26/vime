from setuptools import find_packages, setup

setup(
    name="vime",
    version="0.0.1",
    description="Unofficial Lightning implementation of VIME (Value Imputation and Mask Estimation).",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Jaesu Han",
    author_email="gkswotn9753@gmail.com",
    url="https://github.com/Jaesu26/vime",
    packages=find_packages(exclude=["examples"]),
    license="MIT",
    zip_safe=False,
    include_package_data=True,
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.8",
)
