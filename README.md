# polyseq

polyseq is an open source python package for single cell RNA sequencing analysis and visualization. polyseq starts with a gene by cell matrix, which is then filtered for high quality cells, normalized, regressed, reduced, and clustered.visualization can then be performed with tSNE or umap for the full dataset. polyseq includes inbuilt functionality for violin plots and heatmaps. the data remains in a form that is easy to integrate with the vast community of python packages for further visualization and analysis

### installation 
pip install -r requirements.txt --user <br>
pip install -e .

### demo jupyter notebook
check out a demo inside the pacakge here:
examples/demo.ipynb

### contributors
the code was written by jason wittenbach for the purpose of analyzing data collected by ben cocanougher at janelia research campus. it is useful for any single cell genomics data.
