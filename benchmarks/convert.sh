#!/usr/bin/env bash

BASEPATH=$1
SCRIPTDIR=$(cd $(dirname $0); pwd)

for nb in $(find ${BASEPATH} -iname "*.ipynb") ; do
    jupyter nbconvert --to python --template ${SCRIPTDIR}/templates/benchmark.tpl "${nb}"
done
