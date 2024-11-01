from setuptools import setup, find_packages

setup(
    name="trainer",
    version="0.1.0",
    package_dir={"": "src/trainer"},
    packages=find_packages(where="src/trainer"),
    description="Layout-Corrector Trainer Package",
    author="Dummy User",
    author_email="dummy_user@gmail.com",
    license="MIT",
    python_requires=">=3.10, <3.11"
)
