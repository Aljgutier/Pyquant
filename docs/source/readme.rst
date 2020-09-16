=======================
About the Documentation
=======================

This file located at ./docs/source/readme.rst

The documentation is built using Python Docstrings and Sphinx. The key documenation files are as follows

	**./docs/source/index.rst**  ... includes the doctree and includes the key .rst files

	**./docs/source/readme.rst** ... this file. Includes human understandable explanation of the documentation process and key files.

How to build the documentation::

 from the ./docs directory issue the following command
    sphinx-build -b html ./source ./build


How to view the documentation::

  from a browser open the file ./pyquant/docs/build/index.html


Inlude the file Readme.rst::
  include:: ../../README.rst

.. include:: ../../README.rst


