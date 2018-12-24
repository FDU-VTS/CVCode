#! /bin/bash

model_name=${1}
dataset=${2}

case ${model_name} in
    "mcnn")
        python crowd_count.py mcnn 4 ${dataset} 1e-6 Adam
    ;;
    "csr_net")
        python crowd_count.py csr_net 8 ${dataset} 1e-7 SGD
    ;;
    "sa_net")
        python crowd_count.py sa_net 1 ${dataset} 1e-6 SGD
    ;;
    "tdf_net")
        python crowd_count.py tdf_net 4 ${dataset} 1e-7 SGD
    ;;
    "inception")
        python crowd_count.py inception 16 ${dataset} 1e-6 Adam
    ;;
    "aspp")
        python crowd_count.py aspp 8 ${dataset} 1e-7 SGD
    ;;
    "pad_net")
        python crowd_count.py pad_net 8 ${dataset} 1e-7 SGD
esac