#!/usr/bin/env python3
#
# Author: Pavlos Sermpezis (https://sites.google.com/site/pavlossermpezis/)
#

import pymongo

MONGO_HOST = 'mongodb://localhost:27017/'


def get_mongo_db(db_name, host=MONGO_HOST):
    client = pymongo.MongoClient(host)
    if db_name in client.list_database_names():
        return client[db_name]
    else:
        print('WARNING: DB does not exist')
        return None


def create_new_db(db_name, host=MONGO_HOST):
    return pymongo.MongoClient(host)[db_name]
    


