#!/bin/bash

# ./format.sh target
# target is the name of a test partition profile in config.sh
# generally this means "clean" or "aged"

DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi

profile=$1
. "$DIR/config.sh"

$DIR/destroy.sh $profile

echo "formatting $profile partition of type $fs_type"

case "$fs_type" in
	ext4)
		set -x
		#mkfs -t ext4 -E lazy_itable_init=0,lazy_journal_init=0 $partition > /dev/null
                mkfs -t ext4 $partition > /dev/null
                tune2fs -O ^has_journal $partition > /dev/null
		mkdir -p $mntpnt
		mount -t ext4 $partition $mntpnt 
		chown -R $user $mntpnt 
		;;

	xfs)
		set -x
		mkfs.xfs -f $partition > /dev/null
		mkdir -p $mntpnt
		mount -t xfs $partition $mntpnt 
		chown -R $user $mntpnt 
		;;

	btrfs)
		set -x
		mkfs.btrfs -f $partition > /dev/null
		mkdir -p $mntpnt
		mount -t btrfs $partition $mntpnt 
		chown -R $user $mntpnt 
		;;

	f2fs)
		set -x
		mkfs.f2fs $partition > /dev/null
		mkdir -p $mntpnt
		mount -t f2fs $partition $mntpnt
		chown -R $user $mntpnt
		;;

	*)
		echo "Unknown filesystem type $fs_type"
		exit 1
		;;
esac
