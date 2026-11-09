"""Sphinx configuration."""

from importlib.metadata import version as get_version

project = "spaceodyssey"
release = get_version("spaceodyssey")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

autoapi_dirs = ["../src/spaceodyssey"]
autoapi_ignore = ["**/*version.py"]
# AutoAPI cannot resolve the ignored generated version module or constants
# assigned by tuple unpacking (studies.toy.RECORD_AUXILIARY).
suppress_warnings = ["autoapi.python_import_resolution"]
# Document each object at its defining module to keep re-export links unambiguous.
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
myst_enable_extensions = ["amsmath", "dollarmath"]

html_theme = "sphinx_book_theme"
html_context = {"default_mode": "dark"}
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}

nb_execution_mode = "auto"
nb_execution_timeout = 120
nb_execution_raise_on_error = True

copyright = "2026, Corey Spohn"
author = "Corey Spohn"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "jupyter_execute"]
autodoc_typehints = "description"
# AutoAPI already indexes fields; Napoleon should render their descriptions only.
napoleon_use_ivar = True
html_title = "spaceodyssey - Astronomical analysis and observing campaigns"
html_theme_options = {
    "repository_url": "https://github.com/CoreySpohn/spaceodyssey",
    "repository_branch": "main",
    "use_repository_button": True,
    "show_toc_level": 2,
}
