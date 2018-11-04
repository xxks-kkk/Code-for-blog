#!/bin/bash

# This script makes sure the system has the requisite programs, especially a
# late enough version of git. It also downloads and untars the necessary linux
# repo.

set -x

apt-get update > /dev/null

# script dependencies
apt-get -y install blktrace > /dev/null

# git dependencies
apt-get -y install make git autoconf libcurl4-gnutls-dev libexpat1-dev gettext libz-dev libssl-dev > /dev/null

# filesystems
apt-get -y install xfsprogs f2fs-tools btrfs-tools > /dev/null

# git related
git config --global gc.autodetach False
git config --global gc.auto 0
git config --global uploadpack.allowReachableSHA1InWant True
git config --global user.name xxks-kkk
git config --global user.email ferrishu3886@gmail.com
