#coding=utf-8
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import redis
import pymongo
import json

pool=redis.ConnectionPool(host='localhost',port=6379,decode_responses=True)
r=redis.Redis(connection_pool=pool)

conn=pymongo.MongoClient('localhost',27017)
db=conn.tor_python
table=db.visti_times

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)


r.set("visit_time",0)
class MHandler(tornado.web.RequestHandler):
   def get(self):
        r.incr("visit_time")
        v_time=int(r.get("visit_time"))
        
        self.write("You request the main page")
class SHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("hello")
        v_time=int(r.get("visit_time"))
        r.rpush("visit_num",v_time)
        t=int(r.rpop("visit_num"))
        data='{"visit_num":"%d"}'%t
        data_tojson=json.loads(data)
        table.insert(data_tojson)
        
        r.set("visit_time",0)
        print(v_time)

app = tornado.web.Application([(r"/",MHandler),(r"/s",SHandler),])


if __name__ == "__main__":
    
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()


