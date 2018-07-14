from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='deda',
      version='1.0-beta2',
      python_requires='>=3.3',
      description='tracking Dots Extraction, Decoding and Anonymisation toolkit',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/dfd-tud/deda',
      author='Timo Richter',
      author_email='timo.juez@gmail.com',
      license='GNU General Public License v3 or later (GPLv3+)',
      packages=['libdeda', 'deda_bin'],
      install_requires=[
          'numpy', 'opencv-python', 'argparse', 'scipy', 'Pillow'
      ],
      entry_points={'console_scripts': [
        'deda_anonmask_apply = deda_bin.deda_anonmask_apply:main',
        'deda_anonmask_create = deda_bin.deda_anonmask_create:main',
        'deda_clean_document = deda_bin.deda_clean_document:main',
        'deda_compare_prints = deda_bin.deda_compare_prints:main',
        'deda_parse_print = deda_bin.deda_parse_print:main',
        'deda_extract_yd = libdeda.extract_yd:main',
      ]},
      include_package_data=True,
      zip_safe=False,
      classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Printing",
      ),
)

