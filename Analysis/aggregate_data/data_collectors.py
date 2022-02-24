import requests
from datetime import date

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