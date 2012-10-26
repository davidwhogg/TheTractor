#!/bin/bash
for file in $(ls *.pdf);
do
prefix=$(basename $file .pdf)
convert -density 300 $file $prefix.png
convert $prefix.png $prefix.eps
done
