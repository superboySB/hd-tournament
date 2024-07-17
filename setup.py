from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "agents.houlang.agent", 
        ["agents/houlang/agent.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="houlang_agent",
    ext_modules=cythonize(extensions),
    zip_safe=False,
)