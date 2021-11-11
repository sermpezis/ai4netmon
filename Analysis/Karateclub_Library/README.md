**Karate Club** is an unsupervised machine learning extension library for [NetworkX](https://networkx.org/).

Karate Club consists of state-of-the-art methods to do unsupervised learning on graph structured data. 
To put it simply it is a Swiss Army knife for small-scale graph mining research. The library provides network embedding techniques at the node and graph level.

In the **call_library_method.py** script:
* We provide the dataset we want to convert to graph embeddings
* After the run of script we take a .csv file containing the graph embeddings.

<u>**Footnote:**</u>
Karate Club assumes that the NetworkX graph provided by the user for node embedding and community detection has the following important properties:
* Nodes are indexed with integers.
* The node indexing starts with zero and the indices are consecutive.