#!/bin/sh

# This script is used to measure the space/write amplification reported in the writeup

# root path
ROOT_PATH=/home/iamzeyuanhu/hzy
# The location of the filebench binary
FILEBENC_BIN=$ROOT_PATH/filebench_install_custom_latest/bin/filebench
# The path the workload. In our case, we use varmail.f
WORKLOAD_PATH=$ROOT_PATH/varmail.f
#WORKLOAD_PATH=$ROOT_PATH/varmail-2.f
# The directory to save the reuslt
RES_PATH=$ROOT_PATH/filebench_res
# The directory to save filebench generated files
# WARNING: don't modify this value. The modified filebench hardcode this value
FILEBENC_FILE_PATH=$ROOT_PATH/empty

gather_data() {
    # We get the file system type
    fs_type=`df -Th | awk '$1 ~ /\dev\/sdb1/ {print $2}'`
    
    # We use this variable to keep track of the `kB_wrtn' number before we run the
    # the workload
    # Unit: KB
    kB_wrtn_before=`iostat -d /dev/sdb1 | awk '$1 ~ /sdb1/ {print $6}'`

    # We set the filebench output file name
    filebench_output=$RES_PATH/varmail_$fs_type

    # We run the filebench varmail workload with strace
    printf "==== Executing filebench: $WORKLOAD_PATH ====\n"
    sudo strace -f -e trace=write -o strace.txt "$FILEBENC_BIN" -f "$WORKLOAD_PATH"

    # We use now track the `kB_wrtn` number after we run the workload
    # Unit: KB
    kB_wrtn_after=`iostat -d /dev/sdb1 | awk '$1 ~ /sdb1/ {print $6}'`

    # We caclulate the difference between two kB_wrtn values
    # Unit: KB
    kB_wrtn_diff=`echo "$kB_wrtn_after - $kB_wrtn_before" | bc`

    # We get the number of bytes written during the preallocation phase of 
    # the fileset of the filebench (i.e., `prealloc=80` in `define fileset` command)
    # If we use strace with -f, there can be multiple of outputs of our command
    # Since the marker is unique (:diff:), all the output should be the same. We
    # use the last one (END).
    # Unit: KB
    kB_wrtn_preallocation=`grep 'diff' strace.txt | awk -F ':' 'END {print $3}'`

    # We parse the strace output and get the number of bytes written from
    # write() syscall. We exclude all the write() to fd=1
    # Unit: B
    # In the strace.txt, we notice that if there is no workload for the filebench,
    # the only write() for fd that is other than 1,2 is 4. Since we're only interested in
    # the write() happens only during running the workload, we exclude fd=4 as well.
    actual_write=`grep '^write([5-9][0-9]*' strace.txt | awk -F '=' '{sum += $2} END {print sum}'`
    if [ -z "$actual_write" ]; then
        actual_write=0
    fi

    # Since actual_write is in bytes, we translate it into KB to have the unit
    # with the kB_wrtn_before (note we use integer division here)
    # Unit: KB
    actual_write=`echo "$actual_write / 1024" | bc`

    # We fetch the result from the filebench printout, which is saved in the strace.txt
    # Unit: KB
    actual_write=`grep 'tot' strace.txt | awk -F ':' '{print $3}'`

    # We calculate the bytes written during the workload, which equals to 
    # = (iostat after filebench - iostat before filebench) - (iostat after the preallocation - iostat before filebench)
    # = kB_wrtn_diff - kB_wrtn_preallocation
    # Unit: KB
    kB_wrtn_workload=`echo "$kB_wrtn_diff - $kB_wrtn_preallocation" | bc`

    # We calculate the write amplification
    write_amplification=`echo "$kB_wrtn_workload / $actual_write" | bc -l`

    # Now we have everything we need, let's construct the result printout string
    printf "==== Resulting Statistics  ====\n"
    printf "%s : %s\n" "fs_type" "$fs_type" 
    printf "%s: %s\n" "kB_wrtn_before (kb)" "$kB_wrtn_before"
    printf "%s: %s\n" "kB_wrtn_after (kb)" "$kB_wrtn_after"
    printf "%s : %s\n" "kB_wrtn_diff (kb):" "$kB_wrtn_diff"
    printf "%s : %s\n" "kB_wrtn_preallocation (kb):" "$kB_wrtn_preallocation"
    printf "%s : %s\n" "kB_wrtn_workload (kb):" "$kB_wrtn_workload"
    printf "%s : %s\n" "actual_write (kb):" "$actual_write"
    printf "%s : %s\n" "write_amplification:" "$write_amplification" 
}

main(){
    # create the directory to save the results
    mkdir -p "$RES_PATH"
    # create the directory to save the filebench-generated files
    mkdir -p $FILEBENC_FILE_PATH

    gather_data
    sudo rm -rf $FILEBENC_FILE_PATH
}

main


