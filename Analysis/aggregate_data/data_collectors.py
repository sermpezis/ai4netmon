import requests
from datetime import date, datetime
from ihr.hegemony import Hegemony
import pandas as pd
import json
import csv
from tqdm import tqdm
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat


RIPE_STAT_RIS_PEERS_URL = 'https://stat.ripe.net/data/ris-peers/data.json?query_time={}'
RIPE_ATLAS_API_URL_PROBES = 'https://atlas.ripe.net/api/v2/probes'

def get_ripe_ris_data(query_time=None):
    '''
    Gets data about the RIPE RIS collectors from the RIPEstat API for the given date (if no date is given, it uses the date of today).
    :param  query_time:   (str) recommended format 'YYYY-MM-DD' (any ISO8601 date format would work; see API https://stat.ripe.net/docs/data_api#ris-peers)
    :return:              (dict) two dicts of format {(str)peer_ip:(int)asn} and {(str)peer_ip:(str)rrc}
    '''
    if query_time is None:
        query_time = str(date.today())
    data = requests.get(RIPE_STAT_RIS_PEERS_URL.format(query_time)).json()
    ris_dict = data['data']['peers']
    ris_peer_ip2asn = dict()
    ris_peer_ip2rrc = dict()
    for rrc in ris_dict.keys():
        for peer in ris_dict[rrc]:
            ris_peer_ip2asn[peer['ip']]  = int(peer['asn'])
            ris_peer_ip2rrc[peer['ip']]  = rrc

    return ris_peer_ip2asn, ris_peer_ip2rrc


def get_ripe_atlas_probes_data():
    '''
    Gets data about the RIPE Atlas probes from the RIPE Atlas API
    :return:              (list) of dicts, where each dict corresponds to the data of a probe
    '''
    data = requests.get(RIPE_ATLAS_API_URL_PROBES).json()
    probes = data['results']
    while data['next'] is not None:
        url = data['next']
        data = requests.get(url).json()
        probes.extend(data['results'])
    return probes



def get_AS_hegemony_scores_dict(list_of_asns, col_datetime):
    '''
    Request the AS hegemony API for the given list of ASNs for the given datetime
    :param      list_of_asns:   (list) list of ASNs (i.e., list of int) to be requested
    :parama     col_datetime:   (str) datetime for the collection of the data; format must be "YYYY-MM-DD HH:MM" (e.g., "2018-09-15 00:00");
                                The default value gets the current day ("today") at the time 00:00
    :return:                (dict) dict of dicts; with keys the ASNs and values the hegemony values 
                            for IPv4 (key: 'hege4'), IPv6 (key: 'hege6') and total (key: 'hege');
                            e.g., for AS123 and AS456 {123:{'hege4':10, 'hege6':1.5, 'hege':11.5}, 456:{'hege4':...}, ... }
    '''
    hege = Hegemony(asns=list_of_asns, start=col_datetime, end=col_datetime)

    HEGE_DICT = dict()
    for r in hege.get_results():
        asn = r[0].get('asn')
        hege4 = sum([rr['hege'] for rr in r if rr.get('af')==4])
        hege6 = sum([rr['hege'] for rr in r if rr.get('af')==6])
        HEGE_DICT[asn] = {'hege4':hege4, 'hege6':hege6, 'hege':hege4+hege6}
    return HEGE_DICT



def collect_AS_hegemony_dataset(save_filename, col_datetime=None):
    '''
    Loads the list of ASNs from the aggregated dataframe, request the hegemony scores from the AS hegemony API (in batches), 
    and saves the results in a file (json) and only the aggregate hegemony score to another file (csv).
    :param  col_datetime:   (str) datetime for the collection of the data; format must be "YYYY-MM-DD HH:MM" (e.g., "2018-09-15 00:00");
                            The default value gets the current day ("today") at the time 00:00
    :return:                (dict) dict of dicst; with keys the ASNs and values the hegemony values 
                            for IPv4 (key: 'hege4'), IPv6 (key: 'hege6') and total (key: 'hege');
                            e.g., for AS123 and AS456 {123:{'hege4':10, 'hege6':1.5, 'hege':11.5}, 456:{'hege4':...}, ... }

    '''
    # set datetime of collection to request from AS hegemony API
    if col_datetime is None:
        col_datetime = str(datetime.today().date())+" 00:00"

    # load list of ASNs
    df = dat.load_aggregated_dataframe()
    list_of_asns = [int(i) for i in df.index]

    # set window size to request from AS_hege API 
    window_size = 1000
    nb_asns = len(list_of_asns)
    nb_API_requests = int(nb_asns/window_size)

    TEMP_FILE_SAVENAME_FORMAT = 'AS_hege_TEMP{}.json'
    
    # make sequential requests to the API and save results (dicts) in temporary files
    HEGE_DICT = dict()
    for i in tqdm(range(nb_API_requests)):
        start_ind = i*window_size
        end_ind = min((i+1)*window_size, nb_asns)
        HD = get_AS_hegemony_scores_dict(list_of_asns[start_ind:end_ind], col_datetime)
        with open(TEMP_FILE_SAVENAME_FORMAT.format(i),'w') as f:
            json.dump(HD,f)

    # merge the temporary files with the partial results to one final file
    for i in range(nb_API_requests):
        with open(TEMP_FILE_SAVENAME_FORMAT.format(i), 'r') as f:
            HD = json.load(f)
            HEGE_DICT = {**HEGE_DICT, **HD}

    # save the data
    with open(save_filename, 'w') as f:
        json.dump(HEGE_DICT,f)

    #s save the data also to csv
    write_df = pd.DataFrame.from_dict(HEGE_DICT, orient='index')
    write_df.index.name = 'asn'
    write_df[['hege']].to_csv(save_filename.replace('.json','.csv'))



def collect_bgp_tools_dataset(save_filename):
    '''
    Collects data from the bgp.tool website and saves them in a txt
    '''
    BGP_TOOLS_URL = 'https://bgp.tools/tags/perso.txt'

    #fake an agent to retrieve data from bgp.tools
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    
    r = requests.get(url, headers=headers)
    list_of_perso_ASNs = [[i] for i in response.text.split('\n') if i.startswith('AS')]

    with open(save_filename, 'w') as f:
        w = csv.writer(f)
        w.writerows(list_of_perso_ASNs)