from flask import Flask, render_template, request
from MnistPredict import MnistPredict as Mnist
from cassandra.query import SimpleStatement
from cassandra.cluster import Cluster
import os
import datetime
import random
import json
import logging
import time

log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))


@app.route('/', methods=['get'])
def hello_world():
    try:
        cluster = Cluster(contact_points=['192.168.2.100'], port=9042)
        session = cluster.connect()
        KEYSPACE = "mnistimages"
        session.execute("""
                CREATE KEYSPACE %s
                WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
                """ % KEYSPACE)
            
        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)
        
        log.info("creating table...")
        session.execute("""
            CREATE TABLE images (
            image blob,
            Result int,
            time timestamp,
            PRIMARY KEY (image, time)
            )
            """)
    except Exception as e:
        log.error(e)
    return render_template('index.html')


@app.route('/', methods=['post'])
def up_photo():
    def get_time_stamp():
        ct = time.time()
        local_time = time.localtime(ct)
        data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        data_secs = (ct - int(ct)) * 1000
        time_stamp = "%s.%03d" % (data_head, data_secs)
        return time_stamp

    img = request.files.get('photo')
    print(img)
    path = basedir + "/static/photo/"
    if '.' in img.filename and img.filename.rsplit('.', 1)[-1] in set(['png', 'jpg', 'JPG', 'PNG']):
        nowtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        randomnum = random.randint(0, 100)
        if randomnum <= 10:
            randomnum = str(0) + str(randomnum)
        uniquenum = str(nowtime) + str(randomnum)
        file_path = path + uniquenum + img.filename
        img.save(file_path)
        bytearray1 = bytearray(Mnist.imageprepareStorage(file_path))
        result = Mnist(file_path).get_predict_result()

        cluster = Cluster(contact_points=['192.168.2.100'], port=9042)
        session = cluster.connect()
        try:
            session.set_keyspace("mnistimages")
        except Exception as e:
            log.error(e)
            return json.dumps({"number": "server error!"})
        params = [bytearray1, result, get_time_stamp()]
        query = SimpleStatement(
            "insert into images(image,Result,time)values(%s,%s,%s)")
        session.execute(query, params)
        os.remove(file_path)
        return json.dumps({"number": str(result)})
    else:
        return json.dumps({"number": "error! you did not upload an image in png or jpg format"})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
#app.run()
