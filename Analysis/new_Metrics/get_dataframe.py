from data_aggregation_tools import create_your_dataframe



if __name__ == "__main__":
    data = create_your_dataframe()

    # write dataframe to csv
    # with index False drop the first column where it enumarates the rows of the dataframe
    data.to_csv('final_dataframe.csv', index=False)
    print(data)