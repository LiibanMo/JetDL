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

        # Compilation flags
        compilation_flags = [
            "clang++",
            "-std=c++17",
            "-O3",
            "-fopenmp",
            "-c",  
            "-fPIC",
            "-I/opt/homebrew/include/eigen3",
            "-I/opt/homebrew/Cellar/openblas/0.3.29/include",
        ]

        # Linking flags
        linking_flags = [
            "clang++",
            "-shared",
            "-o", shared_library,
            "-fopenmp",
            "-I/opt/homebrew/opt/libomp/include", 
            "-flto",  # Enable link-time optimization
            "-L/opt/homebrew/Cellar/openblas/0.3.29/lib",
            "-lopenblas",
        ]

        # Compile each .cpp file to .o
        object_files = []
        for cpp_file in cpp_files:
            obj_file = cpp_file.replace('.cpp', '.o')
            subprocess.check_call(compilation_flags + [cpp_file, "-o", obj_file])
            object_files.append(obj_file)

        # Link all object files into final shared library
        subprocess.check_call(linking_flags + object_files)

        # Clean up object files
        for obj_file in object_files:
            os.remove(obj_file)

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
