#! /bin/bash

model_name=${1}
dataset=${2}

case ${model_name} in
    "mcnn")
        python train.py mcnn 4 ${dataset}
    ;;
    "csr_net")
        python train.py csr_net 8 ${dataset}
    ;;
    "sa_net")
        python train.py sa_net 1 ${dataset}
    ;;
    "tdf_net")
        python train.py tdf_net 4 ${dataset}
    ;;
esac