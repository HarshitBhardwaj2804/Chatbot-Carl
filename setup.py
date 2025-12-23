from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    '''
    This function returns a list of strings representing the packages required to run the project.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "")for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)    
    
    return requirements


setup(
    name = "CARL",
    version = "0.2.7",
    author = "Harshit",
    author_email= "hb28042004@gmail.com",
    packages=find_packages(),
    include_dirs=get_requirements('requirements.txt')
)