#coding=utf-8

import redis

pool=redis.ConnectionPool(host='localhost',port=6379,decode_responses=True)
r=redis.Redis(connection_pool=pool)
with open('name','r') as f:
	result=list()
	for line in f.readlines():
		line=line.strip()
		r.rpush('nnn',line)

print(r.lrange('nnn',0,-1))
	

