from setuptools import setup, find_packages

setup(
    name="med_image",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "Pillow",
        "matplotlib",
        "opencv-python",
        "scikit-learn",
        "flask",
    ],
)
