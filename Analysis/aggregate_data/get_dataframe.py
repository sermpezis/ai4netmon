import data_aggregation_tools as dat
import networkx as nx
import numpy as np
import pandas as pd

ALL_DATASETS = ['AS_rank', 'personal', 'PeeringDB']

if __name__ == "__main__":

    data = dat.create_dataframe_from_multiple_datasets(ALL_DATASETS)
    data = data.fillna(value=np.nan)
    # Drop the first row, by selecting all rows from first row onwards
    data = data.iloc[1:, :]

    # write dataframe to csv, with index False drop the first column where it enumarates the rows of the dataframe
    data.to_csv('final_dataframe.csv', index=False)

    # create a graph based on AS-relationships
    G = dat.create_bigraph_from_AS_relationships()
    print(nx.info(G))

    # create a graph based on netixlan
    H = dat.create_bigraph_from_netixlan()
    print(nx.info(H))

