#!/bin/sh -e

# This script is used to measure the space/write amplification reported in the writeup

# The location of the filebench binary
FILEBENC_BIN=$HOME/filebench_install/bin/filebench
# The path the workload. In our case, we use varmail.f
WORKLOAD_PATH=$HOME/varmail.f
# The directory to save the reuslt
RES_PATH=$HOME/filebench_res

gather_data() {
    # We get the file system type
    fs_type=`df -Th | awk '$1 ~ /\dev\/sda1/ {print $2}'`
    
    # We use this variable to keep track of the `kB_wrtn' number before we run the
    # the workload
    # Unit: KB
    kB_wrtn_before=`iostat -d /dev/sda1 | awk '$1 ~ /sda1/ {print $6}'`
    printf "%s: %s\n", "kB_wrtn_before", "$kB_wrtn_before"

    # We set the filebench output file name
    filebench_output=$RES_PATH/varmail_$fs_type
    
    # We run the filebench varmail workload with strace
    sudo strace -e trace=write -o strace.txt "$FILEBENC_BIN" -f "$WORKLOAD_PATH" | tee "$filebench_output"
    #sudo strace -e trace=write -o strace.txt "$FILEBENC_BIN" -f "$WORKLOAD_PATH"

    # We parse the filebench output to get the file size we work with
    # Unit: MB (We assume the varmail worload file size is MB)
    filesize=`awk '$2 ~ /bigfileset/ {print $18}' "$filebench_output"  | sed s'/MB//'`
    #filesize=1601.035

    # We transform the filesize into KB for the proper comparison
    # Unit: KB
    filesize=`echo "$filesize * 1000" | bc`
    printf "%s: %s\n", "filesize", "$filesize"
    
    # We use now track the `kB_wrtn` number after we run the workload
    # Unit: KB
    kB_wrtn_after=`iostat -d /dev/sda1 | awk '$1 ~ /sda1/ {print $6}'`
    printf "%s: %s\n", "kB_wrtn_after", "$kB_wrtn_after"

    # We caclulate the difference between two kB_wrtn values
    # Unit: KB
    kB_wrtn_diff=`echo "$kB_wrtn_after - $kB_wrtn_before" | bc`
    printf "%s: %s\n", "kB_wrtn_diff", "$kB_wrtn_diff"    

    # We parse the strace output and get the number of bytes written from
    # write() syscall
    # Unit: B
    actual_write=`awk -F '=' '{sum += $2} END {print sum}' strace.txt`

    # Since actual_write is in bytes, we translate it into KB to have the unit
    # with the kB_wrtn_before (note we use integer division here)
    # Unit: KB
    actual_write=`echo "$actual_write / 1000" | bc`
    printf "%s: %s\n", "actual_write", "$actual_write"    

    # We calculate the write amplification
    write_amplification=`echo "$actual_write / $filesize" | bc -l`
    printf "%s: %s\n", "write_amplification", "$write_amplification"

    # We calculate the space amplfication
    space_amplification=`echo "$kB_wrtn_diff / $filesize" | bc -l`
    printf "%s: %s\n", "space_amplification", "$space_amplification"

    # Now we have everything we need, let's construct the result printout string
    # to be displayed later
    printf "%s %s %s %s %s %s\n" "fs_type", "kB_wrtn_diff (kb)", "actual_write(kb)", "filesize(kb)", "write_amplification", "space_amplification"
    printf "%s\n" "========================================================================================================================"
    result_str="$fs_type \t $kB_wrtn_diff \t $actual_write \t $filesize \t $write_amplification \t $space_amplification\n"

    echo "$result_str"
}

main(){
    # create the directory to save the results
    mkdir -p "$RES_PATH"

    gather_data
}

main


