from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "agents.houlang.my_agent_demo", 
        ["agents/houlang/my_agent_demo.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="houlang_agent",
    ext_modules=cythonize(extensions),
    zip_safe=False,
)