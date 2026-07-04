#!/bin/bash
for ui in $(ls *.ui)
do
    echo "Compiling ${ui}"
    ../../../../.venv/bin/pyuic6 "${ui}" -o "${ui%.ui}.py"
done