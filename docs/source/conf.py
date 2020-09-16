# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# https://shunsvineyard.info/2019/09/19/use-sphinx-for-python-documentation/ << follow this example
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'Pyquant'
copyright = '2020, Alberto Gutierrez'
author = 'Alberto Gutierrez'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.


# autodoc extension 
#  https://www.sphinx-doc.org/en/master/usage/quickstart.html
#  When documenting Python code, it is common to put a lot of 
#  documentation in the source files, in documentation strings. Sphinx supports the 
#  inclusion of docstrings from your modules with an extension (an extension is a Python
#   module that provides additional features for Sphinx projects) called autodoc.

# numpy style extension ... Napoleon
#   https://shunsvineyard.info/2019/09/19/use-sphinx-for-python-documentation/
#   sometimes use NmPy style docstrings
#   for now use googlestyle docstrings so do not include this
#  extensions = [
#    'sphinx.ext.napoleon'
#   ]
extensions = ['sphinx.ext.autodoc','sphinx.ext.napoleon']
# 

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
