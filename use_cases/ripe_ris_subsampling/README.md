### Use case: Select a subset of RIPE RIS peers

Files:
* **`dataset_selected_monitors_ripe_ris_{similarity method}_{clustering method}.json`**: Files containing an _ordered list_ (by order of selection) of RIPE RIS peers; can be used for subset selection, e.g., for selection of 20 peers, keep the first 20 items in the list. ("full" indicates that only full feeding peers have been taken into account)


The files are generated with the script `example_script_plot_proximity_improvement_vs_similarity_methods.py`, which also produces the following plot that shows the performance of the selected monitors when the objective is the "proximity" of the selected set of RIPE RIS peers to the origin ASes (the plot corresponds to only IPv4 peers and proximity values). 

![Subset selection methods and their efficiency wrt. the proximity metrix](./fig_ripe_ris_subset_selection_vs_proximity_v4.png?raw=true)