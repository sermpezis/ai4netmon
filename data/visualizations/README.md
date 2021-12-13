# Data related to network/monitor visualizations. 


* **`dataset_2D_visualization_<set>_<distance or similarity>_<method>_<date[YYYYMMDD]>.csv`**: Datasets with TSNE visualizations calculated based on the similarity dataset (see `../similarity/` folder for more info). Load the file with `pandas.read_csv(filename, header=0, index_col=0)`. The first column (index) is the monitor id (e.g., RIPE RIS peer IP), the 2nd and 3rd columns the TSNE x and y coordinates, and the remaining columns extra meta-data for each monitor