from setuptools import setup, find_packages
import sys
from pathlib import Path

CURRENT_DIRECTORY = Path(__file__).parent.absolute()

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 8)
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of iblutil requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)

with open("README.md", 'r') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    require = [x.strip() for x in f.readlines() if not x.startswith('git+')]

setup(
    name='brainwidemap',
    version='0.1.0',
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    description='Github repo for code associated with IBL brainwide map paper',
    license="MIT",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='IBL',
    url="https://www.internationalbrainlab.com/",
    packages=find_packages(exclude=['scratch', 'tests']),  # same as name
    include_package_data=True,
    # external packages as dependencies
    install_requires=require,
    entry_points={},
    scripts={}
)
