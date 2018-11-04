# Configuration file for git test

################################################################################
# General Test Configuration Parameters

test_name=testy_test
total_pulls=20
pulls_per_grep=10
gc_on=False
keep_traces=False
clear_cache=True
grep_random=8TCg8BVMrUz4xoaU

################################################################################
# System Parameters

user=iamzeyuanhu:iamzeyuanhu

################################################################################
# Profiles
# set mntpnt to '' to disable in the test

case "$profile" in
	aged)
		partition=/dev/sdb1
		mntpnt=/home/iamzeyuanhu/hzy
		fs_type=ext4
		# ext4 only
		datastore=agedstore 
		;;

	# clean)
	# 	partition=/dev/sdb2
	# 	mntpnt=/mnt/clean
	# 	fs_type=ext4
	# 	# ext4 only
	# 	datastore=cleanstore
	# 	;;

	# cleaner)
	# 	partition=/dev/sdb3
	# 	mntpnt=/mnt/cleaner
	# 	fs_type=ext4
	# 	# ext4 only
	# 	datastore=cleanerstore
	# 	;;

	*)
		echo "Unknown profile $profile"
		exit 1
		;;
esac
