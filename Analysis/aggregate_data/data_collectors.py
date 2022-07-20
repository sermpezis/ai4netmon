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
import wget



URL_ORIGINS = 'https://github.com/CAIDA/mapkit-cti-code/tree/master/PAM-Paper-Results/CTI/origin'
URL_TOP = 'https://github.com/CAIDA/mapkit-cti-code/tree/master/PAM-Paper-Results/CTI'
PATH_ORIGINS = 'https://raw.githubusercontent.com/CAIDA/mapkit-cti-code/master/PAM-Paper-Results/CTI/origin/{}'
PATH_TOP = 'https://raw.githubusercontent.com/CAIDA/mapkit-cti-code/master/PAM-Paper-Results/CTI/{}'



def get_ripe_ris_data(query_time=None):
    '''
    Gets data about the RIPE RIS collectors from the RIPEstat API for the given date (if no date is given, it uses the date of today).
    :param  query_time:   (str) recommended format 'YYYY-MM-DD' (any ISO8601 date format would work; see API https://stat.ripe.net/docs/data_api#ris-peers)
    :return:              (dict) two dicts of format {(str)peer_ip:(int)asn} and {(str)peer_ip:(str)rrc}
    '''
    ris_dict = ripe_ris_collector(query_time=query_time, save_filename=None)
    ris_peer_ip2asn = dict()
    ris_peer_ip2rrc = dict()
    for rrc in ris_dict.keys():
        for peer in ris_dict[rrc]:
            ris_peer_ip2asn[peer['ip']]  = int(peer['asn'])
            ris_peer_ip2rrc[peer['ip']]  = rrc

    return ris_peer_ip2asn, ris_peer_ip2rrc


def ripe_ris_collector(query_time=None, save_filename=None):
    '''
    Gets data about the RIPE RIS collectors from the RIPEstat API for the given date (if no date is given, it uses the date of today)
    and saves them in a file (if the "save_filename" argument is gievn) and returns the data (json/dict).
    :param  query_time:   (str) recommended format 'YYYY-MM-DD' (any ISO8601 date format would work; 
                                see API https://stat.ripe.net/docs/data_api#ris-peers)
    :param  save_filename:(str) filename to save. If none is given the data is only returned
    :return:              (dict) the data in a dict of {rrc_id: list_of_peers}
    '''
    RIPE_STAT_RIS_PEERS_URL = 'https://stat.ripe.net/data/ris-peers/data.json?query_time={}'
    if query_time is None:
        query_time = str(date.today())
    data = requests.get(RIPE_STAT_RIS_PEERS_URL.format(query_time)).json()
    ris_dict = data['data']['peers']

    if save_filename is not None:
        with open (save_filename, 'w') as f:
            json.dump(ris_dict, f)

    return ris_dict



def ripe_atlas_probes_collector(save_filename=None):
    '''
    Gets data about the RIPE Atlas probes from the RIPE Atlas API and saves them in a file and returns them
    :param  save_filename:(str) filename to save. If none is given the data is only returned
    :return:              (list) of dicts, where each dict corresponds to the data of a probe
    '''
    RIPE_ATLAS_API_URL_PROBES = 'https://atlas.ripe.net/api/v2/probes'
    data = requests.get(RIPE_ATLAS_API_URL_PROBES).json()
    probes = data['results']
    while data['next'] is not None:
        url = data['next']
        data = requests.get(url).json()
        probes.extend(data['results'])

    if save_filename is not None:
        with open(save_filename, 'w') as f:
            json.dump(probes, f)

    return probes




def get_AS_hegemony_scores_dict(list_of_asns, col_datetime):
    '''
    #################################################
    #### DEPRECATED FUNCTION - NOT USED ANYMORE #####
    #################################################

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



def AS_hegemony_collector_detailed(save_filename, col_datetime=None):
    '''
    #################################################
    #### DEPRECATED FUNCTION - NOT USED ANYMORE #####
    #################################################

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



def AS_hegemony_collector(save_filename, col_datetime=None):
    '''
    Requests the **global** hegemony scores from the AS hegemony API, and saves the results in a file.
    :param  save_filename:(str) filename to save.
    :param  col_datetime:   (str) datetime for the collection of the data; format must be "YYYY-MM-DD HH:MM" (e.g., "2018-09-15 00:00");
                            The default value gets the current day ("today") at the time 00:00
    :return:                (dict) dict of dicst; with keys the ASNs and values the hegemony values 
                            for IPv4 (key: 'hege4'), IPv6 (key: 'hege6') and total (key: 'hege');
                            e.g., for AS123 and AS456 {123:{'hege4':10, 'hege6':1.5, 'hege':11.5}, 456:{'hege4':...}, ... }

    '''
    # set datetime of collection to request from AS hegemony API
    if col_datetime is None:
        col_datetime = str(datetime.today().date())+" 00:00"

    hege = Hegemony(originasns=0, start=col_datetime, end=col_datetime)
    r = next(hege.get_results())
    df = pd.DataFrame().from_dict(r)
    df = df[['asn','hege']]
    df.set_index('asn',inplace=True)
    df = df.drop(-1) # drop an ASN "-1" that appears in the data

    df.to_csv(save_filename)



def bgp_tools_collector(save_filename):
    '''
    Collects data from the bgp.tool website and saves them in a txt
    '''
    BGP_TOOLS_URL = 'https://bgp.tools/tags/perso.txt'
    #fake an agent to retrieve data from bgp.tools
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    
    r = requests.get(BGP_TOOLS_URL, headers=headers)
    list_of_perso_ASNs = [[i] for i in r.text.split('\n') if i.startswith('AS')]

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
        r = requests.get(AS_RANK_URL.format(window, offset)).json()
        print('\t step {} ({}) \t total time {} sec\r'.format(i, round(r['data']['asns']['totalCount']/window), round(time.time()-t)), end='')
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

    
    
def peeringdb_collector(save_filename):
    """
    Collects data from CAIDA peering db API and saves them to json file. The current date is 
    calculated, and the day before is used to extract data, because data of the same day 
    sometimes are not uploaded. If it is the first day of the month, we change the month
    by 1 too. Using those dates, the function gets the data from Peering db's api.
    :param save_filename: (str) the file to be save the dataset
    """
    PEERING_DB_URL = "https://publicdata.caida.org/datasets/peeringdb/{}/peeringdb_2_dump_{}.json"
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

    data = requests.get(PEERING_DB_URL.format(d1, d2)).json()
    with open(save_filename, 'w') as f:
        json.dump(data, f)
   


def AS_rel_collector(save_filename):
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


        
def RouteViews_collector(save_filename):
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
    data = requests.get(ROUTE_VIEWS_URL)
    temp_file = 'temp.txt'
    with open(temp_file, 'w') as temp:
        temp.write(data.text)
    remove_empty_lines(temp_file)

    bad_words = ['ROUTEVIEWS COLLECTOR', '====================', '<pre>', '</pre>']

    with open(temp_file) as oldfile, open(save_filename, 'w') as newfile:
        for line in oldfile:
            if not any(bad_word in line for bad_word in bad_words):
                newfile.write(line)
    os.remove(temp_file)

    with open(save_filename, 'r') as fin:
        data = fin.read()
        data = data.replace("|", " ")
        data = data.splitlines(True)

    # for line in data:
    #     print(' '.join(line.split()))

    with open(save_filename, 'w') as fout:
        for line in data[5:]:
            fout.write(' '.join(line.split()) + "\n")

def ASDB_collector(save_filename):
    '''
    Downloads the ASDB dataset and saves it in the save_filename
    '''
    ASDB_PATH = 'https://asdb.stanford.edu/data/2022-05_categorized_ases.csv'
    wget.download(ASDB_PATH, out=save_filename)


def CTI_collector(save_filename, dataset):
    if dataset == 'top':
        CTI_URL = 'https://github.com/CAIDA/mapkit-cti-code/tree/master/PAM-Paper-Results/CTI'
        CTI_RAW_URL_FORMAT = 'https://raw.githubusercontent.com/CAIDA/mapkit-cti-code/master/PAM-Paper-Results/CTI/{}'
    elif dataset == 'origin':
        CTI_URL = 'https://github.com/CAIDA/mapkit-cti-code/tree/master/PAM-Paper-Results/CTI/origin'
        CTI_RAW_URL_FORMAT = 'https://raw.githubusercontent.com/CAIDA/mapkit-cti-code/master/PAM-Paper-Results/CTI/origin/{}'
    else:
        raise ValueError
    
    filenames = cti_list_of_filenames_for_top_and_origins(CTI_URL)
    dfs = cti_get_dfs(filenames, CTI_RAW_URL_FORMAT)
    if dataset == 'top':
        df = cti_create_top_df(dfs)
    elif dataset == 'origin':
        df = cti_create_origins_df(dfs)    
    df.to_csv(save_filename)


def cti_list_of_filenames_for_top_and_origins(url):
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


def cti_get_dfs(filenames, path):
    """
    Function that reads all the csv files in the list from list_of_filenames, and reads them to
    pandas Dataframe
    :param filenames: the list of csv files names
    :param path: the path to read the csv files from
    :return: list of dataframes of the csv files
    """
    dfs = []
    for i in range(len(filenames)):

        df = pd.read_csv(path.format(filenames[i]), encoding='ISO-8859-1', header=None)
        dfs.append(df)

    return dfs


def cti_create_origins_df(dfs):
    """
    Creates the origins dataframe (by concatenating the input list), after throwing unneeded columns
    and take only the names of the ASNs.
    :param dfs: list of dataframes of origin csv files
    """
    all_dfs_origins = pd.concat(dfs)
    all_dfs_origins = all_dfs_origins.drop([0], axis=0)
    all_dfs_origins.columns = ["ASN", "OriginASName", "%country", "prefix"]
    all_dfs_origins = all_dfs_origins.drop(['OriginASName', 'prefix'], axis=1)

    all_dfs_origins["ASN"] = all_dfs_origins["ASN"].str.split('-', n=1).str.get(0)
    all_dfs_origins = all_dfs_origins.sort_values('%country').drop_duplicates('ASN', keep='last')

    return all_dfs_origins


def cti_create_top_df(dfs):
    """
    Creates the top dataframe (by concatenating the input list), after keeping only the name of the ASNs
    :param dfs: list of dataframes of top csv files
    """
    all_dfs_top = pd.concat(dfs)
    all_dfs_top[0] = all_dfs_top[0].str.split('-', n=1).str.get(0)
    all_dfs_top = all_dfs_top.drop([0], axis=0)
    all_dfs_top.columns = ["ASN", "prefix"]
    all_dfs_top = all_dfs_top.sort_values('prefix').drop_duplicates('ASN', keep='last')

    return all_dfs_top




#### Super script data collection ####

def print_wrapper(i, I, dset, func):
    def wrapper(*args, **kwargs):
        # try:
        print('#{}/{} Collecting dataset: {}'.format(i,I,dset))
        if os.path.exists(kwargs['save_filename']):
            print('\t WARNING: dataset already exists; collection skipped')
            print('\t existng dile: '+kwargs['save_filename'])
        else:
            t0 = time.time()
            func(*args, **kwargs)
            print('\t saved data in: '+kwargs['save_filename'])
            print('\t total time : {}sec.'.format(round(time.time()-t0,2)))
        except Exception as e:
            print('ERROR: Failed to execute !!!',e)
    return wrapper

def super_collector(save_folder):
    '''
    Runs all collectors for this date and saves datasets in the given folder.
    '''
    current_date_str = str(date.today()).replace('-','')
    tjoin = lambda x: os.path.join(save_folder, x.format(current_date_str))


    d = {'RIPE RIS collectors': {'func':ripe_ris_collector, 'kwargs':{'save_filename':'RIPE_RIS_collectors_{}.json'}},
         'RIPE Atlas probes': {'func':ripe_atlas_probes_collector, 'kwargs':{'save_filename':'RIPE_Atlas_probes_{}.json'}},
         'IHR AS Hegemony': {'func':AS_hegemony_collector, 'kwargs':{'save_filename':'AS_hegemony_{}.csv'}},
         'BGPtools personal AS': {'func':bgp_tools_collector, 'kwargs':{'save_filename':'bgptools_perso_{}.txt'}},
         'CAIDA AS rank': {'func':AS_rank_collector, 'kwargs':{'save_filename':'ASrank_{}.csv'}},
         'CAIDA PeeringDB': {'func':peeringdb_collector, 'kwargs':{'save_filename':'PeeringDB_{}.json'}},
         'CAIDA AS relationships': {'func':AS_rel_collector, 'kwargs':{'save_filename':'AS_relationships_{}.txt'}},
         'RouteViews': {'func':RouteViews_collector, 'kwargs':{'save_filename':'RouteViews_{}.txt'}},
         'ASDB': {'func':ASDB_collector, 'kwargs':{'save_filename':'ASDB_{}.csv'}},
         'CTI top': {'func':CTI_collector, 'kwargs':{'save_filename':'CTI_top_{}.csv', 'dataset':'top'}},
         'CTI origin': {'func':CTI_collector, 'kwargs':{'save_filename':'CTI_origin_{}.csv', 'dataset':'origin'}},
         }

    nb_datasets = len(d)
    i = 1
    for k, v in d.items():
        v['kwargs']['save_filename'] = tjoin(v['kwargs']['save_filename'])
        print_wrapper(i, nb_datasets, k, v['func'])(**v['kwargs'])
        i += 1

