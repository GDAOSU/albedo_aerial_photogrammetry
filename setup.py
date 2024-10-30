import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def run(self):
        # Ensure the build directory exists
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Run CMake
        subprocess.check_call(["cmake", self.get_source_dir(), "-B", self.build_temp])
        subprocess.check_call(["cmake", "--build", self.build_temp])

    def get_source_dir(self):
        return os.path.abspath(os.path.dirname(__file__))


setup(
    name="aerial_albedo",
    version="0.1.0",
    description="A package for aerial albedo photogrammetry.",
    author="Shuang Song",
    author_email="sxsong1207@gmail.com",
    license="MIT",
    packages=find_packages(where="src_python"),
    package_dir={"": "src_python"},
    cmdclass={
        "build_ext": CMakeBuild,
    },
)
