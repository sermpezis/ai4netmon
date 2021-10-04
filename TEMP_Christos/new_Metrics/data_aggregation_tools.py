import pandas as pd
import re

def keep_number(text):

    """
    :param text: example AS206924
    :return: 206924
    """
    num = re.findall(r'[0-9]+', text)
    return " ".join(num)

def exists_in_perso(perso, asn):

    """
    :param perso: personal AS
    :param asn: These ASn exist in AS_rank dataset
    :return: True if an ASn from AS_rank dataset exists in personal dataset else returns False
    """
    return asn in perso.asn.values

def create_df_from(dataset):

    if dataset =='AS_rank':
        """
        Change the column names in order to know the features origin
        :return: return the new dataframe
        """
        data = pd.read_csv('../../Datasets/As-rank/asns.csv', sep=",")
        new_columns = ['AS_rank_' + str(i) for i in data.columns]
        data = data.set_axis(new_columns, axis='columns', inplace=False)
        data = data.set_index('AS_rank_asn')

        return data
    elif dataset == 'personal':

        """
        :return: the a dataframe which contains only one column. This column has the ASn of personal dataset as integers
        """
        data = pd.read_csv('../../Datasets/bgp.tools/perso.txt', header=None)
        # name the column
        data.columns = ['asn']
        # keep only the digits of the ASns
        data['asn'] = data['asn'].apply(lambda x: keep_number(x))
        data['personal_is_matched'] = 1
        # needed to convert to a string first, then to an int.
        data['asn'] = data['asn'].astype(str).astype(int)
        data = data.set_index('asn')

        return data
    elif dataset == 'PeeringDB':

        """
        :return PeeringDB dataframe which contains only the features in the keep_keys list.
        """
        df = pd.read_json('../../Datasets/PeeringDB/peeringdb_2_dump_2021_07_01.json')
        data = []
        keep_keys = ['info_ratio', 'info_traffic', 'info_scope', 'info_type', 'info_prefixes4',
                     'info_prefixes6', 'policy_general', 'ix_count', 'fac_count', 'created']
        for row in df.net['data']:
            net_row = []
            for key in keep_keys:
                if key in row:
                    net_row.append(row[key])
                else:
                    net_row.append(None)
            data.append(net_row)
        df = pd.DataFrame(data, columns=keep_keys)
        data = df

        return data

def dfs_concatanate(data_AS, data_per):

    """
    :param data_AS: dataframe AS_rank
    :param data_per: dataframe AS_personal
    :return: the concatenation of the 2 dataframes
    """

    data = pd.concat([data_AS, data_per], axis=1)
    return data

def create_your_dataframe():
    # Create an empty DataFrame object
    data = pd.DataFrame()

    list_od_datasets = ['AS_rank', 'personal', 'PeeringDB']
    list_of_dataframes = []
    for i in list_od_datasets:
        list_of_dataframes.append(create_df_from(i))
    data_con = dfs_concatanate(list_of_dataframes[0], list_of_dataframes[1])

    return data_con

