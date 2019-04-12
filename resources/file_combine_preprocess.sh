#!/bin/bash

# Bash shell script for combining several .article files into a single article file.

echo "Processing Files..."
cd article

for i in $(seq $2)
do
        x=$(expr $i - 1)
	if [ $x -gt $1 ]
	then
		MAX="$(ls -l $x-* | wc -l)"
		COUNT=$(expr $MAX - 1)
		for y in $(seq $COUNT)
		do
                        echo "Processing File: $x-$y-src-$3-txt.article"
		        printf "\n" >> "$x-0-src-$3-txt.article"
			cat "$x-$y-src-$3-txt.article" >> "$x-0-src-$3-txt.article"
			rm "$x-$y-src-$3-txt.article"
		done
	fi
done 
