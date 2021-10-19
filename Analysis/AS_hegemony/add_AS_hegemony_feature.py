from ihr.hegemony import Hegemony
import numpy as np
from Analysis.aggregate_data.data_aggregation_tools import create_dataframe_from_multiple_datasets
import json

ALL_DATASETS = ['AS_rank', 'personal', 'PeeringDB']
window_size = 10000


def call_hegemony(start, end, identifier):
    hege = Hegemony(originasns=list_of_All_ASns[start:end], start='2018-09-15 00:00', end='2018-09-15 23:59')
    buffer = []
    for r in hege.get_results():
        buffer.extend(r)

    file = open('output_{}'.format(identifier) + '.json', 'a')
    json.dump(buffer, file)
    file.close()


data = create_dataframe_from_multiple_datasets(ALL_DATASETS)
data = data.fillna(value=np.nan)
# Convert index=AS_rank_asn to column
data = data.reset_index(level='AS_rank_asn')

list_of_All_ASns = []
list_of_All_ASns = data['AS_rank_asn'].tolist()

dataframe_length = len(list_of_All_ASns)
for i in range(0, int(dataframe_length/window_size)):

    start_index = window_size * i
    end_index = window_size * (i + 1)
    call_hegemony(start_index, end_index, i)

remaining_data = dataframe_length % window_size
# call hegemony for the remaining elements
call_hegemony(dataframe_length-remaining_data, dataframe_length, int(dataframe_length/window_size))

