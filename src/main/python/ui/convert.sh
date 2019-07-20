#!/bin/bash
for ui in $(ls *.ui)
do
    echo "Compiling ${ui}"
    /home/matt/.virtualenvs/beqdesigner-entpycF3/bin/pyuic5 "${ui}" -o "${ui%.ui}.py"
done