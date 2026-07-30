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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
import sys
import os
import numpy
sys.path.insert(0, os.path.abspath('..'))
import flip 

project = 'flip'
copyright = '2023, C. Ravoux, B. Carreres'
author = 'C. Ravoux, B. Carreres'

# The full version, including alpha/beta/rc tags
release = flip.__version__

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'astropy': ('http://docs.astropy.org/en/stable/', None),
    'sncosmo': ('https://sncosmo.readthedocs.io/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None)}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = ['myst_parser', 'sphinx.ext.napoleon', 'sphinx_markdown_tables', 'sphinx.ext.autosectionlabel',
              'sphinx.ext.linkcode', 'sphinx.ext.intersphinx', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary']
extensions.append('autoapi.extension')

# Auto-API config
autoapi_dirs = ['../flip', '../flip/covariance']

def skip_util_classes(app, what, name, obj, skip, options):
    if name == "__init__":
       skip = False
    return skip

def setup(sphinx):
   sphinx.connect("autoapi-skip-member", skip_util_classes)
   
   

napoleon_google_docstring = True
myst_enable_extensions = ["dollarmath"]
myst_dmath_double_inline = True


autosectionlabel_prefix_document = True
# Add any paths that contain templates here, relative to this directory.
autosummary_generate = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_templates']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = "_static/flip_logo.webp"
html_title = "FLIP"
html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/corentinravoux/flip",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ]
}

import inspect
from os.path import relpath


# Copied on sncosmo documentation
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.findsource(obj)
    except:
        lineno = None

    if lineno:
        linespec = "#L%d" % (lineno + 1)
    else:
        linespec = ""

    fn = relpath(fn, start=flip.__flip_dir_path__)
    
    return "https://github.com/corentinravoux/flip/tree/main/flip/%s%s" % (fn, linespec)
