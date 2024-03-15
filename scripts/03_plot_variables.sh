#!/bin/bash
version="bash_test"
if [ "$1" == "run2" ]; then
    config=run2_UL2018_nano_tau_v10
    datasets="data_ul2018_a_single_mu,data_ul2018_b_single_mu,data_ul2018_c_single_mu,data_ul2018_d_single_mu"
    processes="data"
elif [ "$1" == "run3" ]; then
    config="run3_2022_postEE_nano_tau_v12"
    datasets="wj_incl,dy_incl,data_mu_f,data_mu_g"
    processes="wj,dy_lep,data"

elif [ "$1" == "run3lim" ]; then
    config="run3_2022_postEE_nano_tau_v12_limited"
    datasets="wj_incl" #,data_mu_f,data_mu_g"
    processes="wj" #,data"
else
    echo "You need to choose [run2, run3] as the first argument"
    exit
fi
shift
args=(
        --config $config
        --processes $processes
        --version $version
        --datasets $datasets
        --branch -1
        --variables electron_pt,electron_eta,electron_phi,tau_pt,tau_eta,tau_phi
        #--workflow htcondor
        --skip-ratio
        "$@"
    )
law run cf.PlotVariables1D "${args[@]}"