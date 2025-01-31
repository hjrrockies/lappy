"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    author="Hayden Ringer",
    author_email='hringer@vt.edu',
    description="A package for computing Laplacian eigenvalues using the Method \
                 of Particular Solutions (MPS)",
    long_description=readme + '\n\n',# + history,
    include_package_data=True,
    name='pymps',
    packages=find_packages(),
    #test_suite='tests',
    url='https://github.com/hjrrockies/pymps',
    version='0.1.0',
)
