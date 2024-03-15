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


def add_run3_postEE(ana: od.Analysis,
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
        "h_ggf_tautau",
        # "dy_lep",
        # "wj"
    ]
    for process_name in process_names:
        # add the process
        proc = cfg.add_process(procs.get(process_name))
        if proc.is_mc:
           if proc.name == "dy_lep": proc.color1 = (51,153,204)
           if proc.name == "h_ggf_tautau": proc.color1 = (51,53,204)
           if proc.name == "wj": proc.color1 = (204,51,51)

        # configuration of colors, labels, etc. can happen here
        

    # add datasets we need to study
    dataset_names = ["data_egamma_f",
                    #  "data_egamma_g",
                    #  "signal",
                    #  "dy_incl",
                    #  "wj_incl",
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
    
    from higgs_cp.config.triggers import add_triggers_run3_2022_postEE
    add_triggers_run3_2022_postEE(cfg)

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
    cfg.x.luminosity = Number(20655, {
        "lumi_13p6TeV_2022FG": 0.022j,
        
    })
    

    # names of muon correction sets and working points
    # (used in the muon producer)
    #cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", f"{year}")

    # register shifts
    cfg.add_shift(name="nominal", id=0)

    # tune shifts are covered by dedicated, varied datasets, so tag the shift as "disjoint_from_nominal"
    # (this is currently used to decide whether ML evaluations are done on the full shifted dataset)
    #cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
    #cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})

    # fake jet energy correction shift, with aliases flaged as "selection_dependent", i.e. the aliases
    # affect columns that might change the output of the event selection
    #cfg.add_shift(name="jec_up", id=20, type="shape")
    #cfg.add_shift(name="jec_down", id=21, type="shape")
    #add_shift_aliases(
    #     cfg,
    #     "jec",
    #     {
    #         "Jet.pt": "Jet.pt_{name}",
    #         "Jet.mass": "Jet.mass_{name}",
    #         "MET.pt": "MET.pt_{name}",
    #         "MET.phi": "MET.phi_{name}",
    #     },
    # )

    # event weights due to muon scale factors
    # cfg.add_shift(name="mu_up", id=10, type="shape")
    # cfg.add_shift(name="mu_down", id=11, type="shape")
    # add_shift_aliases(cfg, "mu", {"muon_weight": "muon_weight_{direction}"})

    #external files
    # json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-849c6a6e"
    # cfg.x.external_files = DotDict.wrap({
    #     # jet energy correction
    #     "jet_jerc": (f"{json_mirror}/POG/JME/{year}{corr_postfix}_UL/jet_jerc.json.gz", "v1"),

    #     # tau energy correction and scale factors
    #     "tau_sf": (f"{json_mirror}/POG/TAU/{year}{corr_postfix}_UL/tau.json.gz", "v1"),

    #     # electron scale factors
    #     "electron_sf": (f"{json_mirror}/POG/EGM/{year}{corr_postfix}_UL/electron.json.gz", "v1"),

    #     # muon scale factors
    #     "muon_sf": (f"{json_mirror}/POG/MUO/{year}{corr_postfix}_UL/muon_Z.json.gz", "v1"),

    #     # btag scale factor
    #     "btag_sf_corr": (f"{json_mirror}/POG/BTV/{year}{corr_postfix}_UL/btagging.json.gz", "v1"),

    #     # met phi corrector
    #     "met_phi_corr": (f"{json_mirror}/POG/JME/{year}{corr_postfix}_UL/met.json.gz", "v1"),
    # })

    
    
    cfg.x.external_files = DotDict.wrap({
        # lumi files
        "lumi": {
            "golden": ("/eos/user/c/cmsdqm/www/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
            "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
        },
        "pileup":{
            #"json": ("/eos/user/c/cmsdqm/www/CAF/certification/Collisions22/PileUp/EFG/pileup_JSON.txt", "v1")
            "data" : "/afs/cern.ch/user/s/stzakhar/work/run3_taufw/CMSSW_10_6_13/src/TauFW/PicoProducer/data/pileup/Data_PileUp_2022_postEE.root",
            "mc"   : "/afs/cern.ch/user/s/stzakhar/work/run3_taufw/CMSSW_10_6_13/src/TauFW/PicoProducer/data/pileup/MC_PileUp_2022.root"
        },
        "muon_correction" : "/afs/cern.ch/user/s/stzakhar/work/run3_taufw/CMSSW_10_6_13/src/TauFW/PicoProducer/data/lepton/MuonPOG/Run2022postEE/muon_SFs_2022_postEE.root"

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
        "muon_weight": [],
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
    cfg.add_channel(name="mutau", id=1)
    cfg.add_channel(name="etau", id=2)
    
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
    
    