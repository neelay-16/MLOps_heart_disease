from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
name = "MLOPS-Assignment",
version = "0.1",
author = "Neelay",
packages=find_packages(),         #in whichever directory, we have create the file "__init__.py", this line will automatically detect all the packages created by us i.e the custom packages
install_requires = requirements,
)

#command to run setup.py - pip install -e . -----------> automatically detects setup.py file to install the package, requirements
#creates a folder called "MLOPS_Assignment.egg-info" which contains some of the project information which we passed in the above code