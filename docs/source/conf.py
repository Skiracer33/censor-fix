# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
#import sphinx_rtd_theme
import nbsphinx

sys.path.insert(0, os.path.abspath('../../censorfix'))
#print(os.path.abspath('../../censorfix'))

# -- Project information -----------------------------------------------------

project = 'censor-fix'
copyright = '2019, Rowan Swiers'
author = 'Rowan Swiers'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['nbsphinx','sphinx.ext.autodoc','sphinx_rtd_theme','sphinx.ext.mathjax','sphinx.ext.napoleon']

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['**.ipynb_checkpoints','_build']


master_doc = 'index' # to stop read the docs complaining

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme='sphinx_rtd_theme'

html_logo='logo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
