# coding: utf-8

"""
Definition of triggers
"""

import order as od

from higgs_cp.config.util import Trigger, TriggerLeg

def add_triggers_run3_2022_preEE(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    """
    config.x.triggers = od.UniqueObjectIndex(Trigger, [
        # #
        # # single electron
        # #      
        # Trigger(
        #     name="HLT_Ele30_WPTight_Gsf",
        #     id=201,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=11,
        #             min_pt=31,
        #             trigger_bits=1,
        #         ),
        #     ],
        #     applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
        #     tags={"single_trigger", "single_e", "channel_e_tau"},
        # ),        
        
        # # # # Single Tau
        # Trigger(
        #     name="HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1",
        #     id=301,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=180.0,
        #             trigger_bits=8,
        #         ),
        #     ],
        #     applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
        #     tags={"single_trigger", "single_tau", "channel_e_tau"},
        # ),        
        
        # # e tauh
        # Trigger(
        #     name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
        #     id=401,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=11,
        #             min_pt=26.0,
        #             trigger_bits= 7,
        #         ),
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=35.0,
        #             trigger_bits=11,
        #         ),
        #     ],
        #     applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
        #     tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        # ),
            Trigger(
            name="HLT_Ele32_WPTight_Gsf",
            id=201,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=35.0,
                    # filter names:
                    # hltEle32WPTightGsfTrackIsoFilter
                    trigger_bits=2,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),

        Trigger(
            name="HLT_Ele35_WPTight_Gsf",
            id=203,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=38.0,
                    # filter names:
                    # hltEle35noerWPTightGsfTrackIsoFilter
                    trigger_bits=2,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),
        # Trigger(
        #     name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1",
        #     id=401,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=11,
        #             min_pt=27.0,
        #             # filter names:
        #             # hltEle24erWPTightGsfTrackIsoFilterForTau
        #             # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
        #             trigger_bits=2 + 64,
        #         ),
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=35.0,
        #             # filter names:
        #             # hltSelectedPFTau30LooseChargedIsolationL1HLTMatched
        #             # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
        #             trigger_bits=1024 + 256,
        #         ),
        #     ],
        #     applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
        #     tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        # ),
    ])



def add_triggers_run3_2022_postEE(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    """
    config.x.triggers = od.UniqueObjectIndex(Trigger, [
        # #
        # # single electron
        # #      
        # Trigger(
        #     name="HLT_Ele30_WPTight_Gsf",
        #     id=201,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=11,
        #             min_pt=31,
        #             trigger_bits=1,
        #         ),
        #     ],
        #     applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
        #     tags={"single_trigger", "single_e", "channel_e_tau"},
        # ),        
        
        # # # # Single Tau
        # Trigger(
        #     name="HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1",
        #     id=301,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=180.0,
        #             trigger_bits=8,
        #         ),
        #     ],
        #     applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
        #     tags={"single_trigger", "single_tau", "channel_e_tau"},
        # ),        
        
        # # e tauh
        # Trigger(
        #     name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
        #     id=401,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=11,
        #             min_pt=26.0,
        #             trigger_bits= 7,
        #         ),
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=35.0,
        #             trigger_bits=11,
        #         ),
        #     ],
        #     applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
        #     tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        # ),
            Trigger(
            name="HLT_Ele32_WPTight_Gsf",
            id=201,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=35.0,
                    # filter names:
                    # hltEle32WPTightGsfTrackIsoFilter
                    trigger_bits=2,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),

        Trigger(
            name="HLT_Ele35_WPTight_Gsf",
            id=203,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=38.0,
                    # filter names:
                    # hltEle35noerWPTightGsfTrackIsoFilter
                    trigger_bits=2,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),
        # Trigger(
        #     name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1",
        #     id=401,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=11,
        #             min_pt=27.0,
        #             # filter names:
        #             # hltEle24erWPTightGsfTrackIsoFilterForTau
        #             # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
        #             trigger_bits=2 + 64,
        #         ),
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=35.0,
        #             # filter names:
        #             # hltSelectedPFTau30LooseChargedIsolationL1HLTMatched
        #             # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
        #             trigger_bits=1024 + 256,
        #         ),
        #     ],
        #     applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
        #     tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        # ),
    ])

 
def add_triggers_run2_2018_UL(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    """
    
    config.x.triggers = od.UniqueObjectIndex(Trigger, [
        #
        # single electron
        #
        Trigger(
            name="HLT_Ele32_WPTight_Gsf",
            id=201,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=35.0,
                    # filter names:
                    # hltEle32WPTightGsfTrackIsoFilter
                    trigger_bits=2,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),

        Trigger(
            name="HLT_Ele35_WPTight_Gsf",
            id=203,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=38.0,
                    # filter names:
                    # hltEle35noerWPTightGsfTrackIsoFilter
                    trigger_bits=2,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
            tags={"single_trigger", "single_e", "channel_e_tau"},
        ),
        # Trigger(
        #     name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1",
        #     id=401,
        #     legs=[
        #         TriggerLeg(
        #             pdg_id=11,
        #             min_pt=27.0,
        #             # filter names:
        #             # hltEle24erWPTightGsfTrackIsoFilterForTau
        #             # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
        #             trigger_bits=2 + 64,
        #         ),
        #         TriggerLeg(
        #             pdg_id=15,
        #             min_pt=35.0,
        #             # filter names:
        #             # hltSelectedPFTau30LooseChargedIsolationL1HLTMatched
        #             # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
        #             trigger_bits=1024 + 256,
        #         ),
        #     ],
        #     applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "A"),
        #     tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        # ),
      
    ])
