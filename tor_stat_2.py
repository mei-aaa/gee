#coding=utf-8
import tornado.web
import tornado.ioloop
import tornado.options
import multiprocessing
from tornado.options import define,options
import os,sys
import pymongo
import json
import redis

pool=redis.ConnectionPool(host='localhost',port=6379,decode_responses=True)
r=redis.Redis(connection_pool=pool)

conn=pymongo.MongoClient('localhost',27017)
db=conn.tor_python
table=db.visti_times

define("port", default=9000, help="run on the given port", type=int)

r.set("visit_time",0)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        r.incr("visit_time")
        v_time=int(r.get("visit_time"))
        print(v_time)
        self.write("hello")

class SHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("world")
        v_time=int(r.get("visit_time"))
        r.rpush("visit_num",v_time)
        t=int(r.rpop("visit_num"))
        data='{"visit_num":"%d"}'%t
        data_tojson=json.loads(data)
        table.insert(data_tojson)
        
        r.set("visit_time",0)
        print(v_time)

app=tornado.web.Application([(r'/',MainHandler),(r"/s",SHandler),])

if __name__ == "__main__":
    tornado.options.parse_command_line()
    def run(mid,port):
        print("Process %d start" % mid)
        sys.stdout.flush()
        app.listen(port)
        tornado.ioloop.IOLoop.instance().start()
    jobs=list()
    for mid,port in enumerate(range(9010,9014)):
        p=multiprocessing.Process(target=run,args=(mid,port))
        jobs.append(p)
        p.start()