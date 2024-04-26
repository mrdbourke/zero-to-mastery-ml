#!/bin/bash

# Copy documents
cp section-2-data-science-and-ml-tools/introduction-to-numpy.ipynb docs/
cp section-2-data-science-and-ml-tools/introduction-to-pandas.ipynb docs/
cp section-2-data-science-and-ml-tools/introduction-to-matplotlib.ipynb docs/
cp section-2-data-science-and-ml-tools/introduction-to-scikit-learn.ipynb docs/
cp section-4-unstructured-data-projects/end-to-end-dog-vision-v2.ipynb docs/
cp communicating-your-work.md docs/
cp images/* docs/images/

# Remove .ipynb_checkpoints from docs/
rm -rf docs/.ipynb_checkpoints

# Launch mkdocs server
mkdocs serve