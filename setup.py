from setuptools import setup, find_packages


with open("requirements.txt", "r") as f:
    requirements = f.readlines()
    setup(
        name="sensation",
        version="0.0.1",
        author="Hakan Calim",
        author_email="hakan.calim@fau.de",
        description="In the project entitled Sidewalk Environment Detection System for Assistive NavigaTION (hereinafter referred to as SENSATION), our research team is meticulously advancing the development of the components of SENSATION. The primary objective of this venture is to enhance the mobility capabilities of blind or visually impaired persons (BVIPs) by ensuring safer and more efficient navigation on pedestrian pathways.",
        packages=find_packages(),
        install_requires=requirements,
        license="MIT",
        entry_points={
            "console_scripts": [
                "sensation = sensation.cli:main",
            ],
        },
    )
