In this folder exists some preprocessing steps and the implementation of some Machine Learning algorithms.

# AS_improvement_scores
This folder contains 3 main scripts:

- **classification_test**: In this script, we apply standard Machine Learning algorithms for classification. Specifically, we use **classification_task.csv** dataset and classify our data based on the top_k feature. We notice that our dataset is highly biased so, we apply 5 different class imbalance techniques to mitigate bias. Finally, we give the user the convenience of choosing if he wants to use graph embeddings or not in the training process.

- **collect_info**: This script merges **asns.csv** and **improvements20210601.txt** datasets based on the ASn feature. Then it calls models_metrics script in order to estimate the impact_score feature. Finally, we should mention that users can train the ML algorithms with or without embeddings (produced by the Node2Vec algorithm).

- **models_metrics**: This python script calls the machine Learning algorithms for Regression, in order to calculate the improvement score. Furthermore, it shows the feature importance of each feature for each algorithm. 


# Aggregate_data
This folder contains 5 main scripts:

- **get_dataframe**: Through this script the user will be able to request any dataset (or combination of datasets) or graph.

- **data_aggregation_tools**: This script is responsible for creating dataframes. Specifically, it concatanates heterogeneous data based on user preferences/needs and provides them to the user in a dataframe format. Moreover, by using NetworkX library it creates bipartite graphs based on our datasets.  
   
- **metric_BGP**: This script uses **impact__CAIDA20190801_sims2000_hijackType0_per_monitor_onlyRC_NEW_with_mon_ASNs.csv** datasets and it is responsible for creating a dataframe. Specifically, it selects randomly 50 monitors and for each of them checks if the monitor exists in **metric_data.csv** dataset (You can create this csv in line 173 in Analysis/AS_improvement_scores/classification_test.py). If the condition is true, it adds monitor's features to our dataframe (else just fills them with zeros). The final dataframe is saved in a csv named **Impact_&features.csv**. Then it calls call_models script. The user has the option to choose whether or not the final dataframe will contain embeddings.

- **call_models**: This scripts runs some Standar algorithms for Regression. 
- 
- **add_legitimate_hijacker_features**: It is similar to metric_BGP script, with the only addition being that in this script, we keep two additional features (that depict legitimate or hijacked ASns) except for 50 random monitors. For each of the 2 extra features, we check in the **metric_data.csv** dataset, and in case we find them there, we add their features to our dataframe (else we fill them with zeros).
