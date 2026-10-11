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
html_title = "spaceodyssey - Astronomical analysis and observing campaigns"
html_theme_options = {
    "repository_url": "https://github.com/CoreySpohn/spaceodyssey",
    "repository_branch": "main",
    "use_repository_button": True,
    "show_toc_level": 2,
}
