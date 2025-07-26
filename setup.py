import os
from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    HYPEN_E_DOT = '-e .'
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

current_dir = os.path.dirname(__file__)
requirements_path = os.path.join(current_dir, 'requirements.txt')

setup(
    name='crop_recommendation',
    version='0.0.1',
    author='Ratnam',
    author_email='ratnamojha71@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(requirements_path),
)