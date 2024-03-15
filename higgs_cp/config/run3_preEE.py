# coding: utf-8

"""
Configuration of the higgs_cp analysis.
"""

import functools

import law
import order as od
from scinum import Number

from columnflow.util import DotDict, maybe_import, dev_sandbox
from columnflow.config_util import (
    get_root_processes_from_campaign, 
    add_category,
    verify_config_processes,
)

ak = maybe_import("awkward")


def add_run3_preEE (ana: od.Analysis,
                      campaign: od.Campaign,
                      config_name           = None,
                      config_id             = None,
                      limit_dataset_files   = None,) -> od.Config :

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)
    
    # create a config by passing the campaign, so id and name will be identical
    cfg = ana.add_config(campaign,
                        name  = config_name,
                        id    = config_id)

    # gather campaign data
    year = campaign.x.year
    
    # add processes we are interested in
    process_names = [
        "data",
        #Drell-Yan
        "dy_lep",
        #W + jets
        "wj",
        #diboson
        "ww",
        "wz",
        "zz",
        #ttbar & single top
        "tt_sl",
        "tt_dl",
        "tt_fh",
        "st_tchannel_t",
        "st_tchannel_tbar",
        
    ]
    for process_name in process_names:
        # add the process
        proc = cfg.add_process(procs.get(process_name))
        if proc.is_mc:
            if proc.name == "dy_lep": proc.color1 = (255,204,102)
            if proc.name == "h_ggf_tautau": proc.color1 = (51,53,204)
            if proc.name == "wj": proc.color1 = (201,89,84)
            if proc.name == "tt_sl": proc.color1 = (153,153,204)
            if proc.name == "tt_dl": proc.color1 = (184,184,227)
            if proc.name == "tt_fh": proc.color1 = (87,87,141)
            if proc.name == "ww" : proc.color1 = (191,130,119)
            if proc.name == "wz" : proc.color1 = (153,83,70)
            if proc.name == "zz" : proc.color1 = (212,167,160)

        # configuration of colors, labels, etc. can happen here
        

    # add datasets we need to study
    dataset_names = [
        #data
        "data_egamma_c",
        "data_egamma_d",
        "data_egamma_e",
        #Drell-Yan
        "dy_incl",
        #W+jets
        "wj_incl",
        #Diboson
        "ww",
        "wz",
        "zz",
        #ttbar & single top
        "tt_sl",
        "tt_dl",
        "tt_fh",
        "st_tchannel_t",
        "st_tchannel_tbar",
        ]
    
    for dataset_name in dataset_names:
        # add the dataset
        dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

        # for testing purposes, limit the number of files to 1
        for info in dataset.info.values():
            if limit_dataset_files:
                info.n_files = min(info.n_files, limit_dataset_files) #<<< REMOVE THIS FOR THE FULL DATASET

    # verify that the root process of all datasets is part of any of the registered processes
    verify_config_processes(cfg, warn=True)


    # triggers required, sorted by primary dataset tag for recorded data
    # cfg.x.trigger_matrix = [
    #     (
    #         "SingleMuon", {
    #             "IsoMu24",
    #             "IsoMu27",
    #         },
    #     ),
    # ]
    # # union of all triggers for use in MC
    # cfg.x.all_triggers = {
    #     trigger
    #     for _, triggers in cfg.x.trigger_matrix
    #     for trigger in triggers
    # }
    
    from higgs_cp.config.triggers import add_triggers_run3_2022_preEE
    add_triggers_run3_2022_preEE(cfg)

    # default objects, such as calibrator, selector, producer, ml model, inference model, etc
    cfg.x.default_calibrator = "example"
    cfg.x.default_selector = "default"
    cfg.x.default_producer = "default"
    cfg.x.default_ml_model = None
    cfg.x.default_inference_model = None
    cfg.x.default_categories = ("incl",)
    cfg.x.default_variables = ("n_jet", "jet1_pt")

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    cfg.x.process_groups = {}

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    cfg.x.dataset_groups = {}

    # category groups for conveniently looping over certain categories
    # (used during plotting)
    cfg.x.category_groups = {}

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    cfg.x.variable_groups = {}

    # shift groups for conveniently looping over certain shifts
    # (used during plotting)
    cfg.x.shift_groups = {}

    # selector step groups for conveniently looping over certain steps
    # (used in cutflow tasks)
    cfg.x.selector_step_groups = {
        "default": ["muon", "jet"],
    }
    #  cfg.x.selector_step_labels = {"Initial":0, 
    #                                "Trigger": , "Muon"}
     
    # whether to validate the number of obtained LFNs in GetDatasetLFNs
    # (currently set to false because the number of files per dataset is truncated to 2)
    cfg.x.validate_dataset_lfns = False

    # lumi values in inverse pb
    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2?rev=2#Combination_and_correlations
    #https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis#DATA_AN2
    #Only F and G eras
    cfg.x.luminosity = Number(13960, {
        "lumi_13p6TeV_2022": 0.022j,
        
    })
    

    # names of muon correction sets and working points
    # (used in the muon producer)
    #cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", f"{year}")

    # register shifts
    cfg.add_shift(name="nominal", id=0)

    cfg.x.external_files = DotDict.wrap({
        # lumi files
        "lumi": {
            "golden": ("/eos/user/c/cmsdqm/www/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
            "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
        },
        "pileup":{
            #"json": ("/eos/user/c/cmsdqm/www/CAF/certification/Collisions22/PileUp/EFG/pileup_JSON.txt", "v1")
            "data" : "/afs/cern.ch/work/d/dwinterb/public/Run3_corrections/pu_data_2022_preEE.root",
            "mc"   : "/afs/cern.ch/work/d/dwinterb/public/Run3_corrections/pu_mc_2022.root"
        },
        "muon_correction" : "/afs/cern.ch/work/d/dwinterb/public/Run3_corrections/muon_SFs_2022_preEE.root",
        "electron_correction" : "/afs/cern.ch/work/d/dwinterb/public/Run3_corrections/electron_SFs_2022_preEE.root"

    })

    # target file size after MergeReducedEvents in MB
    cfg.x.reduced_file_size = 512.0
    
    from higgs_cp.config.variables import keep_columns
    keep_columns(cfg)

    # event weight columns as keys in an OrderedDict, mapped to shift instances they depend on
    #get_shifts = functools.partial(get_shifts_from_sources, cfg)
    cfg.x.event_weights = DotDict({
        "normalization_weight": [],
        "pu_weight": [],
        # "muon_weight": [],
        "electron_weight": [],
        
    })

    # versions per task family, either referring to strings or to callables receving the invoking
    # task instance and parameters to be passed to the task family
    def set_version(cls, inst, params):
        # per default, use the version set on the command line
        version = inst.version 
        return version if version else 'dev1'
            
        
    cfg.x.versions = {
        "cf.CalibrateEvents"    : set_version,
        "cf.SelectEvents"       : set_version,
        "cf.MergeSelectionStats": set_version,
        "cf.MergeSelectionMasks": set_version,
        "cf.ReduceEvents"       : set_version,
        "cf.MergeReductionStats": set_version,
        "cf.MergeReducedEvents" : set_version,
    }
    # channels
    # (just one for now)
    # cfg.add_channel(name="mutau", id=1)
    cfg.add_channel(name="etau", id=1)
    
    if cfg.campaign.x("custom").get("creator") == "desy":  
        def get_dataset_lfns(dataset_inst: od.Dataset, shift_inst: od.Shift, dataset_key: str) -> list[str]:
            # destructure dataset_key into parts and create the lfn base directory
            dataset_id = dataset_key.split("/", 1)[1]
            print(f"Creating custom get_dataset_lfns for {config_name}")   
            campagn_name = cfg.campaign.x("custom").get("name")
            lfn_base = law.wlcg.WLCGDirectoryTarget(
                f"{dataset_id}",
                fs=f"wlcg_fs_{campagn_name}",
            )
            
            # loop though files and interpret paths as lfns
            return [
                lfn_base.child(basename, type="f").path
                for basename in lfn_base.listdir(pattern="*.root")
            ]
        # define the lfn retrieval function
        cfg.x.get_dataset_lfns = get_dataset_lfns

        # define a custom sandbox
        cfg.x.get_dataset_lfns_sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/cf.sh")

        # define custom remote fs's to look at
        campagn_name = cfg.campaign.x("custom").get("name")
        print(campagn_name)
        cfg.x.get_dataset_lfns_remote_fs = lambda dataset_inst: f"wlcg_fs_{campagn_name}"
        
        # add categories using the "add_category" tool which adds auto-generated ids
    # the "selection" entries refer to names of selectors, e.g. in selection/example.py
    add_category(
        cfg,
        name="incl",
        selection="cat_incl",
        label="inclusive",
    )
    add_category(
        cfg,
        name="2j",
        selection="cat_2j",
        label="2 jets",
    )
    
    from higgs_cp.config.variables import add_variables
    add_variables(cfg)
    
    