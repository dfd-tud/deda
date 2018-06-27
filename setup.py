from setuptools import setup

"""
      scripts=[
        'bin/deda_anonmask_apply.py',
        'bin/deda_anonmask_create.py',
        'bin/deda_clean_document.py',
        'bin/deda_compare_prints.py',
        'bin/deda_parse_print.py'
      ],
"""
      
setup(name='deda',
      version='1.0-beta1',
      description='tracking Dots Extraction, Decoding and Anonymisation toolkit',
      url='https://github.com/dfd-tud/deda',
      author='Timo Richter',
      author_email='timo.juez@gmail.com',
      license='GNU GPL 3',
      packages=['libdeda'],
      install_requires=[
          'numpy', 'opencv-python', 'argparse', 'scipy'
      ],
      entry_points={'console_scripts': [
        'dedax = bin.test:Main()()'
      ]},
      include_package_data=True,
      zip_safe=False)

