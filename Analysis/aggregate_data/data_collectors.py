import requests
from datetime import date, datetime, timedelta
import time
import os
from urllib.request import urlopen
import bz2
from ihr.hegemony import Hegemony
import pandas as pd
import json
import csv
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
from collections import defaultdict
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat


URL_ORIGINS = 'https://github.com/CAIDA/mapkit-cti-code/tree/master/PAM-Paper-Results/CTI/origin'
URL_TOP = 'https://github.com/CAIDA/mapkit-cti-code/tree/master/PAM-Paper-Results/CTI'
PATH_ORIGINS = 'https://raw.githubusercontent.com/CAIDA/mapkit-cti-code/master/PAM-Paper-Results/CTI/origin/{}'
PATH_TOP = 'https://raw.githubusercontent.com/CAIDA/mapkit-cti-code/master/PAM-Paper-Results/CTI/{}'
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



def AS_hegemony_collector(save_filename, col_datetime=None):
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



def bgp_tools_collector(save_filename):
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



def AS_rank_collector(save_filename, window=500, offset=0):
    """
    Collects data from the CAIDA AS rank API, and saves them in the given filename as csv file. 

    :param  save_filename:  (str) the file to be save the dataset
    :param  window:         (int) check the CAIDA API docs https://api.asrank.caida.org/v2/restful/doc
    :param  offset:         (int) check the CAIDA API docs https://api.asrank.caida.org/v2/restful/doc
    """

    AS_RANK_URL = 'https://api.asrank.caida.org/v2/restful/asns/?first={}&offset={}'

    basic_cols = ["asn", "rank", "source", "longitude", "latitude"]
    cone_cols = ["numberAsns", "numberPrefixes", "numberAddresses"]
    asn_degree_cols = ["total", "customer", "peer", "provider"]

    dict_ = defaultdict(dict)
    offset = 0
    has_next_page = True
    i = 0
    t = time.time()
    while has_next_page:
        print('step {} \t total time {} sec\r'.format(i, round(time.time()-t)), end='')
        r = requests.get(AS_RANK_URL.format(window, offset)).json()
        for e in r['data']['asns']['edges']:
            node = e['node']
            asn = node.get('asn')
            for col in basic_cols:
                dict_[asn][col] = node.get(col)
            for col in cone_cols:
                dict_[asn][col] = node.get('cone').get(col)
            for col in asn_degree_cols:
                dict_[asn][col] = node.get('asnDegree').get(col)
            dict_[asn]['iso'] = node.get("country").get("iso")

        has_next_page = r['data']['asns']['pageInfo']['hasNextPage']
        offset += window
        i += 1

    df = pd.DataFrame.from_dict(dict_, orient='index')
    df.to_csv(save_filename, index=False)

    
    
def data_collectors_peeringdb(save_filename):
    """
    Collects data from CAIDA peering db API and saves them to json file. The current date is 
    calculated, and the day before is used to extract data, because data of the same day 
    sometimes are not uploaded. If it is the first day of the month, we change the month
    by 1 too. Using those dates, the function gets the data from Peering db's api.
    :param save_filename: (str) the file to be save the dataset
    """
    today = date.today()
    month = today.strftime("%m")
    day = today.strftime("%d")
    # subtract some days because usually the data of today are not up yet
    d = today - timedelta(days=1)
    d = str(d)
    if str(day) == '01':
      m = int(month)-1
      m = '0' + str(m)
    else:
      m = month

    d = d[-2:]

    d1 = today.strftime("%Y/"+m)
    d2 = today.strftime("%Y_"+m+"_"+d)
    d3 = today.strftime("%Y"+m+d)

    PEERING_DB_URL = "https://publicdata.caida.org/datasets/peeringdb/{}/peeringdb_2_dump_{}.json"
    data = requests.get(PEERING_DB_URL.format(d1, d2)).json()
    with open('peeringdb_{}'.format(d3), 'w') as f:
        json.dump(data, f)
   


def as_rel_collector(save_filename):
    """
    Collects data from CAIDA's AS relationships API and saves them to json file.
    The current date is calculated and using urllib, the function
    unzips the file from this date in the API, and extracts the txt file. 
    :param save_filename: (str) the file to be save the dataset
    """
    AS_REL_URL = "https://publicdata.caida.org/datasets/as-relationships/serial-2/{}.as-rel2.txt.bz2"
    today = date.today()
    d1 = today.strftime("%Y%m") + "01"

    req = urlopen(AS_REL_URL.format(d1))
    CHUNK = 16 * 1024

    decompressor = bz2.BZ2Decompressor()
    with open(save_filename.format(d1), 'wb') as fp:
        while True:
            chunk = req.read(CHUNK)
            if not chunk:
                break
            fp.write(decompressor.decompress(chunk))
    req.close()
    
    

def remove_empty_lines(filename):
    """
    This function deletes the empty lines from the file of the route views that is collected
    from Route Views.
    :param filename
    :return:
    """
    if not os.path.isfile(filename):
        print("{} does not exist ".format(filename))
        return
    with open(filename) as filehandle:
        lines = filehandle.readlines()

    with open(filename, 'w') as filehandle:
        lines = filter(lambda x: x.strip(), lines)
        filehandle.writelines(lines)


        
def collect_route_views(save_filename):
    """
    Collects data from Route Views URL that contains an html file, and write them to a temporary text file. Then the empty lines
    and some rows that contain only bad words inside the data are cleaned. Finally, the txt file is opened and a new final txt
    is written, that in each line the
    data are seperated in the same way, using the character '|'.
    :param save_filename: (str) the file to be save the dataset
    :return: writes a txt file
    """
    ROUTE_VIEWS_URL = "http://www.routeviews.org/peers/peering-status.html"

    today = date.today()
    date_ = today.strftime("%Y%m%d")

    data = requests.get(ROUTE_VIEWS_URL)
    temp_file = 'temp.txt'
    with open(temp_file, 'w') as temp:
        temp.write(data.text)
    remove_empty_lines(temp_file)

    bad_words = ['ROUTEVIEWS COLLECTOR', '====================', '<pre>', '</pre>']

    with open(temp_file) as oldfile, open(save_filename.format(date_), 'w') as newfile:
        for line in oldfile:
            if not any(bad_word in line for bad_word in bad_words):
                newfile.write(line)

    with open(save_filename.format(date_), 'r') as fin:
        data = fin.read()
        data = data.replace("|", " ")
        data = data.splitlines(True)

    for line in data:
        print(' '.join(line.split()))

    with open(save_filename.format(date_), 'w') as fout:
        for line in data[5:]:
            fout.write(' '.join(line.split()) + "\n")
        
def list_of_filenames_for_top_and_origins(url):
    """
    Function that uses bs4 library to scrape html content from the github repositories where
    the csv files are, and returns a list of the names of those csv files.
    :param url: the url of the github repository
    :return: a list with csv files names
    """
    result = requests.get(url)

    soup = BeautifulSoup(result.text, 'html.parser')
    csvfiles = soup.find_all(title=re.compile("\.csv$"))

    filenames = []
    for i in csvfiles:
        filenames.append(i.extract().get_text())

    return filenames


def data_collector_cti(filenames, path):
    """
    Function that reads all the csv files in the list from list_of_filenames, and reads them to
    pandas Dataframe and saves them to a list of dataframes.
    :param filenames: the list of csv files names
    :param path: the path to read the csv files from
    :return: list of dataframes of the csv files
    """
    dfs = []
    for i in range(len(filenames)):

        df = pd.read_csv(path.format(filenames[i]), encoding='ISO-8859-1', header=None)
        dfs.append(df)

    return dfs


def create_origins_df(dfs):
    """
    Creates the origins dataframe (by concatenating the input list), after throwing unneeded columns
    and take only the names of the ASNs, and writes the dataframe to csv file.
    :param dfs: list of dataframes of origin csv files
    """
    all_dfs_origins = pd.concat(dfs)
    all_dfs_origins = all_dfs_origins.drop([0], axis=0)
    all_dfs_origins.columns = ["ASN", "OriginASName", "%country", "prefix"]
    all_dfs_origins = all_dfs_origins.drop(['OriginASName', 'prefix'], axis=1)

    all_dfs_origins["ASN"] = all_dfs_origins["ASN"].str.split('-', n=1).str.get(0)
    all_dfs_origins = all_dfs_origins.sort_values('%country').drop_duplicates('ASN', keep='last')
    all_dfs_origins.to_csv('origins.csv')


def create_top_df(dfs):
    """
    Creates the top dataframe (by concatenating the input list), after keeping only the name of the ASNs
    :param dfs: list of dataframes of top csv files
    """
    all_dfs_top = pd.concat(dfs)
    all_dfs_top[0] = all_dfs_top[0].str.split('-', n=1).str.get(0)
    all_dfs_top = all_dfs_top.drop([0], axis=0)
    all_dfs_top.columns = ["ASN", "prefix"]
    all_dfs_top = all_dfs_top.sort_values('prefix').drop_duplicates('ASN', keep='last')
    all_dfs_top.to_csv('top.csv')


