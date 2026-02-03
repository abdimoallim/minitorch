from setuptools import setup, find_packages

setup(
  name='minitorch',
  version='0.1.0',
  description='A minimal PyTorch-like library implementation',
  author='minitorch',
  packages=find_packages(),
  install_requires=[
    'numpy>=1.19.0',
  ],
  extras_require={
    'cuda': ['cupy-cuda11x>=9.0.0'],
  },
  python_requires='>=3.7',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
)
