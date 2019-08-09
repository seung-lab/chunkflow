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
#sys.path.insert(0, os.path.abspath('.'))
#sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'chunkflow'
copyright = '2019, Jingpeng Wu'
author = 'Jingpeng Wu'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
with open('../../VERSION.txt', "r") as f:
    version = f.read()

# The full version, including alpha/beta/rc tags.
#release = chunkflow.__release__

# -- General configuration ---------------------------------------------------
#autodoc_mock_imports = ["numpy"]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    #'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode', # This will add links to source code to autodoc
    'sphinx.ext.githubpages',
    #'sphinx.ext.linkcode', # This is similar to viewcode but links to external source -> need to define a function for this
    'sphinx.ext.napoleon',
    #'sphinx.ext.mathjax', # mathjax is interactive and configurable but can also misbehave when rendering - switched to imgmath instead
    #'sphinx.ext.imgmath',
    #'matplotlib.sphinxext.plot_directive',
    'sphinx_autodoc_typehints',
]

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
html_theme = 'default'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

master_doc = 'index'

# enable the documentation of __init__ function for classes
autoclass_content = 'both'

# enable the typehints for Numpy style documentation
napoleon_use_param = True
