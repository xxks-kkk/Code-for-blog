#!/bin/sh -e

# This script is used to measure the space/write amplification reported in the writeup

# The location of the filebench binary
FILEBENC_BIN=$HOME/filebench_install/bin/filebench
# The path the workload. In our case, we use varmail.f
WORKLOAD_PATH=$HOME/varmail.f
# The directory to save the reuslt
RES_PATH=$HOME/filebench_res
# The directory to save filebench generated files
FILEBENC_FILE_PATH=empty

gather_data() {
    # We get the file system type
    fs_type=`df -Th | awk '$1 ~ /\dev\/sda1/ {print $2}'`
    
    # We use this variable to keep track of the `kB_wrtn' number before we run the
    # the workload
    # Unit: KB
    kB_wrtn_before=`iostat -d /dev/sda1 | awk '$1 ~ /sda1/ {print $6}'`

    # We get the space of the device before running the workload
    # Unit: KB
    space_before=`du -k "$FILEBENC_FILE_PATH" | awk 'END {print $1}'`

    # We set the filebench output file name
    filebench_output=$RES_PATH/varmail_$fs_type

    # We run the filebench varmail workload with strace
    printf "==== Executing filebench: $WORKLOAD_PATH ====\n"
    sudo strace -e trace=write -o strace.txt "$FILEBENC_BIN" -f "$WORKLOAD_PATH"

    # We use now track the `kB_wrtn` number after we run the workload
    # Unit: KB
    kB_wrtn_after=`iostat -d /dev/sda1 | awk '$1 ~ /sda1/ {print $6}'`

    # After the workload, the space difference
    # Unit: KB
    space_after=`du -k "$FILEBENC_FILE_PATH" | awk 'END {print $1}'`

    # We caclulate the difference between two kB_wrtn values
    # Unit: KB
    kB_wrtn_diff=`echo "$kB_wrtn_after - $kB_wrtn_before" | bc`

    # We calculate the space difference
    # Unit: KB
    space_diff=`echo "$space_after - $space_before" | bc`

    # We parse the strace output and get the number of bytes written from
    # write() syscall. We exclude all the write() to fd=1
    # Unit: B
    actual_write=`grep '^write([2-9][0-9]*' strace.txt | awk -F '=' '{sum += $2} END {print sum}'`

    # Since actual_write is in bytes, we translate it into KB to have the unit
    # with the kB_wrtn_before (note we use integer division here)
    # Unit: KB
    actual_write=`echo "$actual_write / 1000" | bc`

    # We calculate the write amplification
    write_amplification=`echo "$kB_wrtn_diff / $actual_write" | bc -l`

    # We calculate the space amplfication
    space_amplification=`echo "$space_diff / $actual_write" | bc -l`

    # Now we have everything we need, let's construct the result printout string
    printf "==== Resulting Statistics  ====\n"
    printf "%s : %s\n" "fs_type" "$fs_type" 
    printf "%s: %s\n" "kB_wrtn_before (kb)" "$kB_wrtn_before"
    printf "%s: %s\n" "kB_wrtn_after (kb)" "$kB_wrtn_after"
    printf "%s : %s\n" "kB_wrtn_diff (kb):" "$kB_wrtn_diff"
    printf "%s : %s\n" "actual_write (kb):" "$actual_write"
    printf "%s : %s\n" "space_before (kb)" "$space_before" 
    printf "%s : %s\n" "space_after (kb)" "$space_after"
    printf "%s : %s\n" "space_diff (kb):" "$space_diff" 
    printf "%s : %s\n" "write_amplification:" "$write_amplification" 
    printf "%s : %s\n" "space_amplification:" "$space_amplification"
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


