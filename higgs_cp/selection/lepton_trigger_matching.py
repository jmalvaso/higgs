# coding: utf-8

"""
Lepton selection methods.
"""
# law run cf.PlotCutflow --config run2_UL2018_nano_tau_v10_limited --version dev --branch 0 --dataset data_egamma_ul2018_a --selector-steps "trigger,Electron_pt_33,Electron_eta_2p4,Electron_dxy_0p045,Electron_dz_0p2,Electron_iso_0p15,Electron_mvaIso_WP90,HLT_path_matched"
# law run cf.PlotCutflow --config run3_2022_postEE_nano_tau_v12_limited  --version dev --branch 0 --dataset data_egamma_f --selector-steps "trigger,Electron_pt_33,Electron_eta_2p4,Electron_dxy_0p045,Electron_dz_0p2,Electron_iso_0p15,Electron_mvaIso_WP90,HLT_path_matched"


from __future__ import annotations

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column
from columnflow.util import DotDict, maybe_import
from columnflow.production.util import attach_coffea_behavior




from higgs_cp.config.util import Trigger, process_e_tau_pairs_single_trigger, process_e_tau_pairs_cross_trigger, mT, veto_os_leptons

np = maybe_import("numpy")
ak = maybe_import("awkward")

@selector(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.dxy", "Electron.dz",
        "Electron.pfRelIso03_all", "Electron.mvaIso_WP90",
        "TrigObj.pt", "TrigObj.eta", "TrigObj.phi","Electron.charge"
    },
    exposed=False,
)
def electron_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    leg_masks: list[ak.Array],
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Electron selection returning two sets of indidces for default and veto electrons.
    See https://twiki.cern.ch/twiki/bin/view/CMS/EgammaNanoAOD?rev=4
    """
    is_single = trigger.has_tag("single_e")
    is_cross = trigger.has_tag("cross_e_tau")
    
    # default electron mask, only required for single and cross triggers with electron leg
    electron_mask = None
    selection_steps = {}


    electron_selections = {
        "e_pt_33"          : events.Electron.pt > 33,
        "e_eta_2p4"        : abs(events.Electron.eta) < 2.1,
        "e_dxy_0p045"      : abs(events.Electron.dxy) < 0.045,
        "e_dz_0p2"         : abs(events.Electron.dz) < 0.2,
        "e_iso_0p15"       : events.Electron.pfRelIso03_all < 0.15,
        "e_mvaIso_WP90"    : events.Electron.mvaIso_WP90 == 1,
    }
    electron_mask = abs(events.event) > 0
    
    index = ak.local_index(events.Electron)
    selected_electron_idxs = []
    for cut_name in electron_selections.keys():
        electron_mask = electron_mask & electron_selections[cut_name]
        # (selection_idx, _) = ak.broadcast_arrays(ak.values_astype(electron_mask, np.int32),electron_mask)
        selection_steps[cut_name] = np.array(ak.any(electron_mask,axis=1), dtype=np.bool_)
    
    selected_electron_idxs.append(index[electron_mask]) #These are the indices of the electrons that passes the electron selection 
    selected_electron_idxs = ak.values_astype(selected_electron_idxs,'uint16')

    return electron_mask, selected_electron_idxs,  SelectionResult(  
        steps = selection_steps,
        objects={
            "Electron": {
                "Electron": selected_electron_idxs #selection_idx
            },
        },
    )
    
@selector(
    uses={
        # nano columns
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.dz", "Tau.idDeepTau2018v2p5VSe",
        "Tau.idDeepTau2018v2p5VSmu", "Tau.idDeepTau2018v2p5VSjet",
        "TrigObj.pt", "TrigObj.eta", "TrigObj.phi",
        "Electron.pt", "Electron.eta", "Electron.phi","Electron.cutBased",
        "Muon.pt", "Muon.eta", "Muon.phi","Muon.mediumId","Muon.pfRelIso04_all",
        "MET.phi","MET.pt","PuppiMET.phi","PuppiMET.pt"
    },

    exposed=False,
)

def tau_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    leg_masks: list[ak.Array],
    selected_electron_idxs: ak.Array,
    electron_mask: ak.Array,
    # muon_indices: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:

    is_single_e = trigger.has_tag("single_e")
    is_cross_e = trigger.has_tag("cross_e_tau")
    is_single_tau = trigger.has_tag("single_tau")

    tau_vs_e = DotDict(VVVLoose=1, VVLoose=2, VLoose=3, Loose=4, Medium=5, Tight=6, VTight=7, VVTight=8)
    tau_vs_mu = DotDict(VLoose=1, Loose=2, Medium=3, Tight=4)
    tau_vs_jet = DotDict(VVVLoose=1, VVLoose=2, VLoose=3, Loose=4, Medium=5, Tight=6, VTight=7, VVTight=8)   
   
    # default electron mask, only required for single and cross triggers with electron leg
    tau_mask = None
    selection_steps = {}
    
    # base tau mask for default and qcd sideband tau
    tau_selections = {   
            "Tau_pt_30"          : events.Tau.pt > 30,
            "Tau_eta_2p3"        : abs(events.Tau.eta) < 2.3,
            "Tau_dz_0p2"         : abs(events.Tau.dz) < 0.2,
            "Tau_vs_e"    : events.Tau.idDeepTau2018v2p5VSe >= tau_vs_e.Tight, 
            "Tau_vs_mu"    : events.Tau.idDeepTau2018v2p5VSmu >= tau_vs_mu.VLoose,
            "Tau_vs_jet"    : events.Tau.idDeepTau2018v2p5VSjet >= tau_vs_jet.Medium,
    }
    tau_mask = abs(events.event) > 0

    index = ak.local_index(events.Tau)
    selected_tau_idxs = []
    for cut_name in tau_selections.keys():
        tau_mask = tau_mask & tau_selections[cut_name]
        selection_idx = ak.values_astype(tau_mask, np.int32)
        # (selection_idx, _) = ak.broadcast_arrays(ak.values_astype(tau_mask, np.int32),tau_mask)
        selection_steps[cut_name] = np.array(ak.any(tau_mask,axis=1), dtype=np.bool_)
    
    selected_tau_idxs.append(index[tau_mask]) #These are the indices of the taus that passes the tau selection  

          
    if is_single_e or is_single_tau:
        #print("Single trigger:", trigger.hlt_field)
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 1
        assert abs(trigger.legs[0].pdg_id) == 11 or abs(trigger.legs[0].pdg_id) == 15 
        pairs_by_idx, mask_event_selected_couple,selected_e,selected_tau = process_e_tau_pairs_single_trigger(events,selected_electron_idxs, selected_tau_idxs,leg_masks)
          
    elif is_cross_e:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 2
        assert abs(trigger.legs[0].pdg_id) == 11
        assert abs(trigger.legs[1].pdg_id) == 15
        pairs_by_idx, mask_event_selected_couple,selected_e,selected_tau = process_e_tau_pairs_cross_trigger(events,selected_electron_idxs, selected_tau_idxs,leg_masks)
    
    (selected_couple, _) = ak.broadcast_arrays(ak.firsts(mask_event_selected_couple)[:,np.newaxis],tau_mask)
    tau_selections.update({"Etau_couples": ak.fill_none(selected_couple,False)})
    tau_mask = tau_mask & tau_selections["Etau_couples"]
    # selection_idx_e = ak.values_astype(selected_e,np.int32)
    # selection_idx_tau = ak.values_astype(selected_tau,np.int32)
    # (selection_idx, _) = ak.broadcast_arrays(ak.values_astype(tau_mask, np.int32),tau_mask)
    selection_steps["Etau_couples"] = np.array(ak.any(tau_mask,axis=1), dtype=np.bool_)


    mT_dict, mask_mT, selected_e,selected_tau = mT(events, pairs_by_idx) 
    (mT_cut, _) = ak.broadcast_arrays(ak.firsts(mask_mT)[:,np.newaxis],tau_mask)
    tau_selections.update({"mT_less_70" :ak.fill_none(mT_cut,False)})
    selection_idx_e = ak.values_astype(selected_e,np.int32)
    selection_idx_tau = ak.values_astype(selected_tau,np.int32)
    tau_mask = tau_mask & tau_selections["mT_less_70"]
    # # selection_idx = ak.values_astype(tau_mask, np.int32)
    selection_steps["mT_less_70"] = np.array(ak.any(tau_mask,axis=1), dtype=np.bool_)
     

    # veto emuon mask 
    veto_mu_mask_requirements = (events.Muon.mediumId == 1) & (events.Muon.pfRelIso04_all < 0.3) &(abs(events.Muon.eta) < 2.4) & (events.Muon.pt > 10.0)
    veto_mu_mask = ak.sum(veto_mu_mask_requirements, axis=1) == 0
    (veto_mu, _) = ak.broadcast_arrays(veto_mu_mask[:,np.newaxis],tau_mask)
    tau_selections.update({"veto_mu" : ak.fill_none(veto_mu,False)})
    tau_mask = tau_mask & tau_selections["veto_mu"]
    selection_steps["veto_mu"] = np.array(ak.any(tau_mask,axis=1), dtype=np.bool_)      
    
    empty_indices = ak.zeros_like(1 * events.event, dtype=np.uint16)[..., None][..., :0]
    
    sel_e_veto_mu = ak.where(veto_mu_mask, selected_e, empty_indices)
    sel_tau_veto_mu = ak.where(veto_mu_mask, selected_tau, empty_indices)

    # veto extra electon mask 
    veto_e_mask = (
        (events.Electron.mvaIso_WP90 == 1) & 
        (events.Electron.pfRelIso03_all < 0.3) &
        (abs(events.Electron.eta) < 2.5) &
        (abs(events.Electron.dxy) < 0.045) &
        (abs(events.Electron.dz) < 0.2) &
        (events.Electron.pt > 10.0)
    )
    veto_e_mask =  ak.sum(veto_e_mask, axis=1) == 1
    (veto_e, _) = ak.broadcast_arrays(veto_e_mask[:,np.newaxis],tau_mask)
    tau_selections.update({"veto_e" : ak.fill_none(veto_e,False)})
    tau_mask = tau_mask & tau_selections["veto_e"]
    selection_steps["veto_e"] = np.array(ak.any(tau_mask,axis=1), dtype=np.bool_)    
    
    sel_e_veto_mu_extra_e = ak.where(veto_e_mask, sel_e_veto_mu , empty_indices)
    sel_tau_veto_mu_extra_e = ak.where(veto_e_mask, sel_tau_veto_mu , empty_indices)
    
    # veto os electron couples 
    Electron = events.Electron
    pairs_by_idx_after_veto, mask_event_after_veto = veto_os_leptons(events, Electron, pairs_by_idx)
    (veto_os, _) = ak.broadcast_arrays(ak.firsts(mask_event_after_veto)[:,np.newaxis],tau_mask)
    tau_selections.update({"veto_os" :ak.fill_none(veto_os,False)})
    tau_mask = tau_mask & tau_selections["veto_os"]
    selection_steps["veto_os"] = np.array(ak.any(tau_mask,axis=1), dtype=np.bool_)
    # from IPython import embed
    # embed()
    sel_e_veto_mu_extra_e_os = ak.where(ak.any(mask_event_after_veto,axis=1), sel_e_veto_mu_extra_e , empty_indices)
    sel_tau_veto_mu_extra_e_os = ak.where(ak.any(mask_event_after_veto,axis=1), sel_tau_veto_mu_extra_e , empty_indices)
    
    sel_e_veto_mu_extra_e_os_uint16 = ak.values_astype(sel_e_veto_mu_extra_e_os, 'uint16')
    sel_tau_veto_mu_extra_e_os_uint16 = ak.values_astype(sel_tau_veto_mu_extra_e_os, 'uint16')


    return tau_mask, SelectionResult(  
        steps = selection_steps,
        objects={
            "Electron": {
                "Electron": sel_e_veto_mu_extra_e_os_uint16
            },
            "Tau": {
                "Tau": sel_tau_veto_mu_extra_e_os_uint16
            },
        },
    ) 
    
@selector(
    uses={
        electron_selection, tau_selection,
        # nano columns
        "event", "Electron.charge", "Muon.charge", "Tau.charge", "Electron.mass", "Muon.mass",
        "Tau.mass",
    },
    produces={
        electron_selection, tau_selection,
    },
)

def lepton_selection(
    self: Selector,
    events: ak.Array,
    trigger_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    
    # Lepton_Results = SelectionResult()
    # perform each lepton election step separately per trigger
    for trigger, trigger_fired, leg_masks in trigger_results.x.trigger_data:

        # electron selection
        electron_mask, selected_electron_idxs, electron_Results = self[electron_selection]( 
            events,
            trigger,
            leg_masks,
            # call_force=True,
            **kwargs,
        )
        
        # tau selection
        tau_mask, tau_Results = self[tau_selection]( 
            events,
            trigger,
            leg_masks,
            selected_electron_idxs,
            electron_mask,
            # call_force=True,
            **kwargs,
        )
 
    Lepton_Results = electron_Results + tau_Results

    return events, Lepton_Results


# empty_events = ak.zeros_like(1 * events.event, dtype=np.uint16)
# ak.values_astype(single_tau_indices, np.int32)

    
