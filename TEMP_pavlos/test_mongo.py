#!/usr/bin/env python3
#
# Author: Pavlos Sermpezis (https://sites.google.com/site/pavlossermpezis/)
#

import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

mydb = myclient["mydatabase"]
mycol = mydb["customers"]

# mydict1 = { "name": "John", "address": "Highway 37" }
# # x = mycol.insert_one(mydict)

# mydict2 = { "name": "Peter", "address": "Lowstreet 27" }
# # x = mycol.insert_one(mydict)

# x = mycol.insert_many([mydict1, mydict2])


# print(x.inserted_ids) 

# print(myclient.list_database_names())
# print(mydb.list_collection_names())


# dblist = myclient.list_database_names()
# if "mydatabase" in dblist:
#     print("The database exists.")


# collist = mydb.list_collection_names()
# if "customers" in collist:
#     print("The collection exists.")



# x = mycol.find_one()
# print(x)



for x in mycol.find():
# for x in mycol.find({},{ "_id": 0, "name": 1, "address": 1 }):
# for x in mycol.find({},{"name": 1, "address": 0 }):
# for x in mycol.find({},{ "name": 0}):
    print(x) 


myquery = { "address": "Park Lane 38" }
myquery = { "address": { "$gt": "H" } }
mydoc = mycol.find(myquery,{ "_id": 0, "name": 1, "address": 1 }).sort("name",-1).limit(1)

# mydoc = mycol.find().sort("name")
# sort("name", 1) #ascending
# sort("name", -1) #descending 

# for x in mydoc:
#     print(x) 




myquery = { "address": "Valley 1" }
newvalues = { "$set": { "address": "Canyon 123" } }

mycol.update_one(myquery, newvalues,upsert=True)



for x in mycol.find():
# for x in mycol.find({},{ "_id": 0, "name": 1, "address": 1 }):
# for x in mycol.find({},{"name": 1, "address": 0 }):
# for x in mycol.find({},{ "name": 0}):
    print(x) 