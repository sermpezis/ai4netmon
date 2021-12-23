In this script, we calculate Euclidean distance and similarity matrix.

- First, we run the calculate_monitors_euclidean_distance.py (Just run the script, no parameters need to be given).

- After running the script, we obtain a file in csv format (named: RIPE_RIS_distance_embeddings_20211221.csv) which is given as input to the second script.
Specifically, we run the calculate_monitors_similarity.py in order to convert Euclidean distance to similarity matrix.

- Finally, we obtain the RIPE_RIS_similarity_20211221.csv, which will be given as input in get_similarity_csv_with_iso.py