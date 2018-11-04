#!/bin/sh -vx

docker run --security-opt seccomp:unconfined -dt --net=host --name=mysql_server cloudsuite/web-serving:db_server localhost
docker run --security-opt seccomp:unconfined -dt --net=host --name=memcache_server cloudsuite/web-serving:memcached_server
docker run --cap-add SYS_PTRACE --security-opt seccomp:unconfined -dt --net=host --name=web_server cloudsuite/web-serving:web_server /etc/bootstrap.sh


# Wait for other servers to come up to avoid curl errors
sleep 10
docker run --security-opt seccomp:unconfined --net=host --name=faban_client cloudsuite/web-serving:faban_client localhost
