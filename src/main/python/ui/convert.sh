#!/bin/bash
for ui in $(ls *.ui)
do
    pyuic5 "${ui}" -o "${ui%.ui}.py"
done