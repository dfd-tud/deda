from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='deda',
      version='1.0-beta1',
      python_requires='>=3.3',
      description='tracking Dots Extraction, Decoding and Anonymisation toolkit',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/dfd-tud/deda',
      author='Timo Richter',
      author_email='timo.juez@gmail.com',
      license='GNU GPL 3',
      packages=['libdeda'],
      install_requires=[
          'numpy', 'opencv-python', 'argparse', 'scipy', 'Pillow'
      ],
      entry_points={'console_scripts': [
        'deda_anonmask_apply = bin.deda_anonmask_apply:main',
        'deda_anonmask_create = bin.deda_anonmask_create:main',
        'deda_clean_document = bin.deda_clean_document:main',
        'deda_compare_prints = bin.deda_compare_prints:main',
        'deda_parse_print = bin.deda_parse_print:main',
      ]},
      include_package_data=True,
      zip_safe=False
)

