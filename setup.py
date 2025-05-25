from setuptools import find_packages, setup



def get_requirements(file_path:str) -> list[str]:
    """
    This function reads the requirements from a file and returns a list of packages.
    :param file_path: str - path to the requirements file
    :return: list - list of required packages
    """
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements

setup(
    name = "mlproject",
    version= "0.0.1",
    author="Sanand",
    author_email="sanandkishan713@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),


)