import pandas as pd
import json


def read_json_file():
    """
    :return: A dataframe containing for each AS the AS_hegemony score for every other AS which connects
    """
    f = open('../../Datasets/AS_hegemony/merged_file.json')
    data = json.load(f)
    df = pd.DataFrame(data)

    new_df = df[['originasn', 'asn', 'hege']]
    return new_df


def remove_all_duplicates(df):
    """
    :param df: It take as input the dataset containing the AS_hegemony scores
    :return: A dataframe where duplicates no longer exist
    """
    # Clean data from duplicate values
    # (example) if origin_Asn=1 and asn=19 exists 2 or more times in the dataset with different timestamp, we keep only the first appearance as the hege remains the same
    new_df = df.drop_duplicates(subset=['originasn', 'asn'], keep='first').reset_index(drop=True)

    # # to delete inverse column values (example origin_Asn=100 asn=70 and origin_Asn=70 and asn=100)
    # new_df['remove_inverse_duplicates'] = new_df.apply(lambda row: ''.join(sorted(str([row['originasn'], row['asn']]))),
    #                                                   axis=1)
    # new_df = new_df.drop_duplicates('remove_inverse_duplicates')

    # keep only the columns that we need
    new_df = new_df[['originasn', 'asn', 'hege']]

    return new_df


df = read_json_file()
cleaned_dataframe = remove_all_duplicates(df)

# For the same asn sum all hegemony scores
hege_df = cleaned_dataframe.groupby(['asn']).agg({'hege': 'sum'})
hege_df.reset_index(level=0, inplace=True)
hege_df.to_csv('AS_hegemony.csv', index=False)
