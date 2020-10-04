#!/bin/bash

# Display the content of all log files in a folder

if [ $# -lt 1 ]
then
    echo "Please provide a folder from which all the log files will be displayed";
    exit 1;
fi
	
log_folder=$1
log_files=`ls $log_folder/*.log 2> /dev/null`

if [ -z "${log_files}" ]
then
    echo "No log files to display in folder $log_folder"
else
    echo "Displaying content of log files in folder $log_folder"
    for log_file in $log_files
    do
        echo File: "${log_file##*/}";
        cat $log_file;
        echo;
    done
fi
