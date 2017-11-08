#coding=utf-8

import redis
import pymongo

pool=redis.ConnectionPool(host='localhost',port=6379,decode_responses=True)
r=redis.Redis(connection_pool=pool)

conn=pymongo.MongoClient('localhost',27017)
db=conn.geetest_python
collection=db.names

with open('name','r') as f:
	result=list()
	for line in f.readlines():
		line=line.strip()
		r.rpush('nnn',line)
mylist=[]
for i in range(r.llen('nnn')):
	coll=r.rpop('nnn')
	mydict={"name":coll}
	mylist.append(mydict)
collection.insert_many(mylist)

