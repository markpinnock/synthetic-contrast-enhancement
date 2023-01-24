from setuptools import find_namespace_packages, setup


setup(
    name="multi-phase-sce",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    version="0.1.0",
    description=(
        "Multi-phase synthetic contrast enhancement: comparison"\
        "of U-Net, Pix2Pix and CycleGAN for generating two contrast phases"\
        "from non-contrast-enhanced computed tomography images"
    ),
    author="Mark Pinnock",
    license="MIT"
)
