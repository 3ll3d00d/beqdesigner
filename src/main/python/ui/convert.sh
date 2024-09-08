#!/bin/bash
for ui in $(ls *.ui)
do
    echo "Compiling ${ui}"
    /home/matt/.cache/pypoetry/virtualenvs/beqdesigner-IpT5f2Ps-py3.12/bin/pyuic6 "${ui}" -o "${ui%.ui}.py"
done