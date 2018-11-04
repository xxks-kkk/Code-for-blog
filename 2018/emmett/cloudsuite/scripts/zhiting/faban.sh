#!/bin/bash
FABAN_IP=127.0.0.1

sudo docker run  --net host -v "/home/zhitingz/Documents/kernel_enclave/docker_files/output:/faban/output" --name faban_client cloudsuite/web-serving:faban_client ${FABAN_IP} 7
