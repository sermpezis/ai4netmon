In this folder exists some preprocessing steps and the implementation of some Machine Learning algorithms.

# AS_improvement_scores
This folder contains 3 main scripts:

- **classification_test**: In this script, we apply standard Machine Learning algorithms for classification. Specifically, we use **classification_task.csv** dataset and classify our data based on the top_k feature. We notice that our dataset is highly biased so, we apply 5 different class imbalance techniques to mitigate bias. Finally, we give the user the convenience of choosing if he wants to use graph embeddings or not in the training process.

- **collect_info**: This script merges **asns.csv** and **improvements20210601.txt** datasets based on the ASn feature. Then it calls models_metrics script in order to estimate the impact_score feature. Finally, we should mention that users can train the ML algorithms with or without embeddings (produced by the Node2Vec algorithm).


- **models_metrics**: This python script calls the machine Learning algorithms for Regression, in order to calculate the improvement score. Furthermore, it shows the feature importance of each feature for each algorithm. 


# Aggregate_data
