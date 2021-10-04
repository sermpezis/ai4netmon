import data_aggregation_tools as dat
import networkx as nx

if __name__ == "__main__":
    data = dat.create_your_dataframe()

    # write dataframe to csv
    # with index False drop the first column where it enumarates the rows of the dataframe
    data.to_csv('final_dataframe.csv', index=False)

    # create a graph based on AS-relationships
    G = dat.create_bigraph_from_AS_relationships()
    print(nx.info(G))

    # create a graph based on netixlan
    H = dat.create_bigraph_from_netixlan()
    print(nx.info(H))

