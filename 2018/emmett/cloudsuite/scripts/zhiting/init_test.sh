#!/bin/bash

WEB_SERVER_IP=127.0.0.1
DATABASE_SERVER_IP=127.0.0.1
MEMCACHED_SERVER_IP=127.0.0.1
MAX_PM_CHILDREN=80

docker run -dt --net=host --name=mysql_server cloudsuite/web-serving:db_server ${WEB_SERVER_IP}
docker run -dt --net=host --name=memcache_server cloudsuite/web-serving:memcached_server
docker run -dt --net=host --name=web_server cloudsuite/web-serving:web_server /etc/bootstrap.sh ${DATABASE_SERVER_IP} ${MEMCACHED_SERVER_IP} ${MAX_PM_CHILDREN}
