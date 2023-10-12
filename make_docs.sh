#!/bin/bash

# Copy documents
cp section-2-data-science-and-ml-tools/introduction-to-numpy.ipynb docs/
cp section-2-data-science-and-ml-tools/introduction-to-pandas.ipynb docs/
cp section-2-data-science-and-ml-tools/introduction-to-matplotlib.ipynb docs/
cp section-2-data-science-and-ml-tools/introduction-to-scikit-learn.ipynb docs/
cp images/* docs/images/

# Launch mkdocs server
mkdocs serve