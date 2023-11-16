"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

# setup_requirements = ['pytest-runner', ]
#
# test_requirements = ['pytest', ]

setup(
    author="Hayden Ringer",
    author_email='hringer@vt.edu',
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7'
    ],
    description="A package for computing Laplacian eigenvalues using the Method \
                 of Particular Solutions (MPS)",
    #install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n',# + history,
    include_package_data=True,
    keywords='pymps',
    name='pymps',
    packages=find_packages(),
    #test_suite='tests',
    url='https://github.com/hjrrockies/pymps',
    version='0.0.1',
    zip_safe=False,
)
