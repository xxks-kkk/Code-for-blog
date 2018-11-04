#!/bin/bash -vx

docker pull cloudsuite/web-serving:db_server
docker pull cloudsuite/web-serving:memcached_server
docker pull cloudsuite/web-serving:web_server
docker pull cloudsuite/web-serving:faban_client
