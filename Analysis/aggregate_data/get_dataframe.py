from data_aggregation_tools import create_your_dataframe
import data_aggregation_tools as dat
import networkx as nx

if __name__ == "__main__":
    data = create_your_dataframe()

    # write dataframe to csv
    # with index False drop the first column where it enumarates the rows of the dataframe
    data.to_csv('final_dataframe.csv', index=False)
    G = dat.create_bigraph_from_AS_relationships()
    print(nx.info(G))
    H = dat.create_bigraph_from_netixlan()
    print(nx.info(H))

