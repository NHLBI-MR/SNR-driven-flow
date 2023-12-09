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
import os
import os.path as op
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../../'))

# -- Project information -----------------------------------------------------

project = 'SNR-driven-flow'
copyright = '2023, Pierre Daudé'
author = 'Daudé P., Ramasawmy R., Javed A., Lederman R.J., Chow K., Campbell-Washburn A.E.'

# The full version, including alpha/beta/rc tags
release = 'v0.0.0'


# -- General configuration ---------------------------------------------------

extensions = ['sphinx.ext.autodoc',
'sphinx.ext.intersphinx',
'sphinx.ext.ifconfig',
'sphinx.ext.viewcode',
'sphinx_rtd_theme',
]

templates_path = ['_templates']

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

htmlhelp_basename='SNR-driven-flow'

html_static_path = ['_static']
