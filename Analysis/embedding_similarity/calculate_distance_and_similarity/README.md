In this script, we calculate Euclidean distance and similarity matrix.

- First, we run the calculate_monitors_euclidean_distance.py, keeping as header the IP_Addresses from monitors (Just run the script, no parameters need to be given).
If we need to create a csv, having as header the ASNs instead of IP_ADDRESSES, we run the calculate_monitors_euclidean_distance_keep_ASns.py, exactly as the aforementioned script.

- After running the one of the above scripts, we obtain a file in csv format (named: RIPE_RIS_distance_embeddings_20211221.csv) which is given as input to the second script.
Specifically, we run the calculate_monitors_similarity.py in order to convert Euclidean distance to similarity matrix.

- Finally, we obtain the RIPE_RIS_similarity_20211221.csv, which will be given as input in get_similarity_csv_with_iso.py