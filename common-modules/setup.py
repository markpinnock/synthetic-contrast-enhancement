from setuptools import find_namespace_packages, setup


setup(
    name="common-sce",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    version="0.1.0",
    description=(
        "Modules common to all synthetic contrast enhancement projects"
    ),
    author="Mark Pinnock",
    license="MIT"
)
