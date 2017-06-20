#!/bin/bash
file=$1
com=$2
a=`md5sum $file`

while [ 1 ]
do
	if [ "$a" != "`md5sum $file`" ]
	then
		echo "file has changed"
		$com
		a=`md5sum $file`
	fi
	sleep 1
done
