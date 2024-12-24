from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os
import glob
import shutil


class BuildSharedLibrary(build_ext):
    """
    Custom command to build the shared library using clang++.
    """
    def run(self):
        source_dir = "src"
        output_dir = "tensorlite"
        shared_library = "libtensor.so"
        # Ensures the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        cpp_files = glob.glob("src/*.cpp")
        if not cpp_files:
            raise FileNotFoundError(f"No .cpp files found in '{source_dir}' directory.")

        # Builds the shared library using clang++
        subprocess.check_call(
            [
                "clang++",
                "-std=c++20",
                "-shared",
                "-o",
                shared_library,
                "-fPIC",
            ]
            + cpp_files
        )

        self.announce(f"Moving {shared_library} to {output_dir}...", level=3)
        shutil.move(shared_library, os.path.join(output_dir, shared_library))

setup(
    name="tensorlite",
    version="0.1.0",
    author="Liiban Mohamud",
    author_email="liibanmohamud12@gmail.com",
    description="A lightweight tensor library with a Python wrapper for C/C++.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LiibanMo/TensorLite",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["tensorlite"],
    package_data={"tensorlite": ["libtensor.so"]},
    ext_modules=[],
    cmdclass={"build_ext": BuildSharedLibrary},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "numpy",
        "pytest",
    ],
    zip_safe=False,
)