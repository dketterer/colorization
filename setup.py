from setuptools import setup

setup(name='Colorization',
      version='1.1',
      description='Train and test a Colorization model with PyTorch',
      author='Daniel Ketterer',
      author_email='',
      url='https://github.com/dketterer/colorization',
      packages=['colorization'],
      entry_points={'console_scripts': ['colorization=colorization.main:main']}
      )
