from setuptools import setup, find_packages


setup(
    name='bgunfolding',
    author='Lars Poppe',
    author_email='lars.poppe@tu-dortmund.de',
    description='',
    license='MIT',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'emcee',
        'astropy',
        'numpy',
        'matplotlib',
        'pandas',
        'scipy',
        'pyfact',
        'h5py',
        'sklearn',
        'numba'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
