#!/bin/bash
for ui in $(ls *.ui)
do
    echo "Compiling ${ui}"
    pyuic5 "${ui}" -o "${ui%.ui}.py"
done