#!/bin/bash
version="bash_test"
if [ "$1" == "run2" ]; then
    config=run2_UL2018_nano_tau_v10
    datasets=(
        'data_ul2018_a_single_mu'
        'data_ul2018_b_single_mu'
        'data_ul2018_c_single_mu'
        'data_ul2018_d_single_mu'
        )

elif [ "$1" == "run2lim" ]; then
    config=run2_UL2018_nano_tau_v10_limited
    datasets=('data_ul2018_a_single_mu')
    processes=('data_mu')

elif [ "$1" == "run3" ]; then
    config=run3_2022_postEE_nano_tau_v12
    datasets=(
        'dy_incl'
        'wj_incl'
        'data_mu_f'
        'data_mu_g'
        )
    workflow=htcondor
elif [ "$1" == "run3lim" ]; then
    config=run3_2022_postEE_nano_tau_v12_limited
    datasets=('wj_incl')
    workflow=local     
else
    echo "You need to choose [run2, run3] as the first argument"
    exit
fi

if [ "$2" == "run_all" ]; then
    for dataset_inst in ${datasets[@]}; do
        args=(
            --config $config
            --version $version
            --dataset $dataset_inst
            --branch -1  
            --workflow $workflow  
            "${@:3}"
            )
        law run cf.ReduceEvents "${args[@]}"
    done
else
    args=(
        --config $config
        --version $version
        --dataset "$2"
        --branch -1  
        --workflow $workflow     
        "${@:3}"
        )
        law run cf.ReduceEvents ${args[@]}
fi