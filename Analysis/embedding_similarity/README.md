Through these scripts, the user can display a two-dimension plot for a given similarity matrix. We calculated the similarity
matrix by taking the Euclidean distance of each AS with all the other ASes.

The aforementioned plot is generated using the [UMAP](https://umap-learn.readthedocs.io/en/latest/) library.
Uniform Manifold Approximation and Projection UMAP is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction.

Python versions:
- The visualize.py script runs in python 3.7.11 version (UMAP library has not yet supported in 3.9 python version).
- The other two scripts run in python 3.9.2 version (As the whole project).

How to run the scripts:
- As a first step, the user should run the scripts placed in calculate_distance_and_similarity folder in order to contain the similarity matrix csv.
- As a second step, the user should run the get_similarity_csv_with_iso.py script
  - In this script, the user should first choose the model that user wishes to run.
  - Furthermore, the user should determine the dimensions (64 or 128).
  - The output of this script will be a .csv file, which will be given as input to visualize.py script.
- As a third step, the user should run the visualize_similarity.py script
 