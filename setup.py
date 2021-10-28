from setuptools import setup, find_packages


setup(name="mpseas",
      version="0.0.2",
      description="Model-based Per Set Efficient Algorithm Selection (MPSEAS)",
      author="Théo Matricon",
      author_email="theomatricon@gmail.com",
      packages=find_packages(),
      install_requires=[
          "numpy",
          "scipy",
          "pyyaml",
          "liac-arff",
          "pandas",
          "tqdm",
          "matplotlib",
          "seaborn",
          "pyrfr"
      ],
      license="MIT")
