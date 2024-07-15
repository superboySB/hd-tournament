from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "agents.team_houlang_0715.houlang_agent", 
        ["agents/team_houlang_0715/houlang_agent.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="houlang_agent",
    ext_modules=cythonize(extensions),
    zip_safe=False,
)