#! /bin/bash

model_name=${1}

case ${model_name} in
    "mcnn")
        python train.py mcnn 4 shtu_dataset
    ;;
    "csr_net")
        python train.py csr_net 8 shtu_dataset
    ;;
    "sa_net")
        python train.py sa_net 1 shtu_dataset
    ;;
    "tdf_net")
        python train.py tdf_net 4 shtu_dataset
    ;;
esac