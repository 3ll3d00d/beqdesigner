#!/bin/bash
for ui in $(ls *.ui)
do
    echo "Compiling ${ui}"
    /usr/bin/pyuic5 "${ui}" -o "${ui%.ui}.py"
done