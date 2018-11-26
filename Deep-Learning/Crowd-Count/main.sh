#! /bin/bash

model_name=${1}
dataset=${2}

case ${model_name} in
    "mcnn")
        python train.py mcnn 4 ${dataset} 1e-5 Adam
    ;;
    "csr_net")
        python train.py csr_net 8 ${dataset} 1e-6 SGD
    ;;
    "sa_net")
        python train.py sa_net 1 ${dataset} 1e-5 Adam
    ;;
    "tdf_net")
        python train.py tdf_net 4 ${dataset} 1e-5 Adam
    ;;
    "inception")
        python train.py inception 16 ${dataset} 1e-5 Adam
    ;;
esac