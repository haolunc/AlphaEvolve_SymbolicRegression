# Configuration file for the Sphinx documentation builder.

project = "AlphaEvolve SR"
version = "0.1.0"
author = ""

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]
myst_heading_anchors = 4

todo_include_todos = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
