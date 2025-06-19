import glob
import os
import shutil
import subprocess

from setuptools import setup
from setuptools.command.build_ext import build_ext


class BuildSharedLibrary(build_ext):
    """
    Custom command to build the shared library using clang++.
    """

    def run(self):
        source_dir = "src"
        output_dir = "jetdl/tensor"
        shared_library = "libtensor.so"
        # Ensures the tensorlite directory exists
        os.makedirs(output_dir, exist_ok=True)

        cpp_files = glob.glob("src/*.cpp")
        if not cpp_files:
            raise FileNotFoundError(f"No .cpp files found in '{source_dir}' directory.")

        # Compilation and linking flags
        flags = [
            "clang++",
            "-std=c++20",
            "-O3",
            "-fopenmp",
            "-shared",
            "-fPIC",
            "-I/opt/homebrew/opt/openblas/include",
            "-L/opt/homebrew/opt/openblas/lib",
            "-lopenblas",
            "-o", shared_library,
        ]

        # Compile all .cpp files directly to shared library
        subprocess.check_call(flags + cpp_files)

        self.announce(f"Moving {shared_library} to {output_dir}...", level=3)
        shutil.move(shared_library, os.path.join(output_dir, shared_library))


setup(
    name="jetdl",
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
    packages=["jetdl"],
    package_data={"jetdl": ["libtensor.so"]},
    ext_modules=[],
    cmdclass={"build_ext": BuildSharedLibrary},
    python_requires=">=3.7",
    install_requires=[
        "pytest",
    ],
    zip_safe=False,
)
