# coding: utf-8

"""
Lepton selection methods.
"""

from __future__ import annotations

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column
from columnflow.util import DotDict, maybe_import

from higgs_cp.config.util import Trigger

np = maybe_import("numpy")
ak = maybe_import("awkward")


def trigger_object_matching(
    vectors1: ak.Array,
    vectors2: ak.Array,
    threshold: float = 0.25,
    axis: int = 2,
) -> ak.Array:
    """
    Helper to check per object in *vectors1* if there is at least one object in *vectors2* that
    leads to a delta R metric below *threshold*. The final reduction is applied over *axis* of the
    resulting metric table containing the full combinatorics. When *return_all_matches* is *True*,
    the matrix with all matching decisions is returned as well.
    """
    # delta_r for all combinations
    dr = vectors1.metric_table(vectors2)

    # check per element in vectors1 if there is at least one matching element in vectors2
    any_match = ak.any(dr < threshold, axis=axis)

    return any_match


@selector(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.dxy", "Electron.dz","Electron.pfRelIso03_all","Electron.mvaIso_WP90",
        "TrigObj.pt", "TrigObj.eta", "TrigObj.phi",
    },
    exposed=False,
)

def study_electron_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    leg_masks: list[ak.Array],
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    
    is_single = trigger.has_tag("single_e")
    # is_cross = trigger.has_tag("cross_e_tau")
    # is_2016 = self.config_inst.campaign.x.year == 2016

    # start per-electron mask with trigger object matching
    if is_single:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 1
        assert abs(trigger.legs[0].pdg_id) == 11
        # match leg 0 
    matches_leg0 = trigger_object_matching(events.Electron, events.TrigObj[leg_masks[0]])
    # elif is_cross:
    #     # catch config errors
    #     assert trigger.n_legs == len(leg_masks) == 2
    #     assert abs(trigger.legs[0].pdg_id) == 11
    #     # match leg 0
    #     matches_leg0 = trigger_object_matching(events.Electron, events.TrigObj[leg_masks[0]])
    
    # pt sorted indices for converting masks to indices
    # sorted_idx = ak.argsort(events.Electron.pt, axis=-1, ascending=False)
    selections = {
        "Electron_pt_36"          : events.Electron.pt > 36,
        "Electron_eta_2p4"        : abs(events.Electron.eta) < 2.1,
        "Electron_dxy_0p045"      : abs(events.Electron.dxy) < 0.045,
        "Electron_dz_0p2"         : abs(events.Electron.dz) < 0.2,
        "Electron_iso_0p15"       : events.Electron.pfRelIso03_all < 0.15,
        "Electron_mvaIso_WP90"    : events.Electron.mvaIso_WP90 == 1,
        "HLT_path_matched"        : matches_leg0 
    }
    electron_mask = abs(events.event) > 0

    selection_idxs = {}
    selection_steps = {}

   
    for cut_name in selections.keys():
        electron_mask = electron_mask & selections[cut_name]
        # selection_idx = sorted_idx[buffer_mask[sorted_idx]]  selection_idx
        selection_idx = ak.values_astype(electron_mask, np.int32)
        selection_idxs[cut_name] = selection_idx
        selection_steps[cut_name] = np.array(ak.num(selection_idx, axis=1) > 0, dtype=np.bool_)
           
    return events,SelectionResult(
        steps= selection_steps,
        objects={
            "Electron": {
                "Electron": electron_mask,
            },
        },
    )
@selector(
    uses={
        study_electron_selection,
        # nano columns
        "event", "Electron.charge", "Muon.charge", "Tau.charge", "Electron.mass", "Muon.mass",
        "Tau.mass",
    },
    produces={
        study_electron_selection, 
        # new columns: , "leptons_os", "tau2_isolated", "cross_triggered",
        "channel_id",
        "single_triggered", 
    },
)

def lepton_selection(
    self: Selector,
    events: ak.Array,
    trigger_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Combined lepton selection.
    """


    # prepare vectors for output vectors
    false_mask = (abs(events.event) < 0)
    channel_id = np.uint8(1) * false_mask

    single_triggered = false_mask

    empty_indices = ak.zeros_like(1 * events.event, dtype=np.uint16)[..., None][..., :0]
    sel_electron_indices = empty_indices

    # perform each lepton election step separately per trigger
    for trigger, trigger_fired, leg_masks in trigger_results.x.trigger_data:
        is_single = trigger.has_tag("single_trigger")

        # electron selection, electron_veto_indices
        electron_indices = self[study_electron_selection](
            events,
            trigger,
            leg_masks,
            # call_force=True,
            **kwargs,
        )

        # lepton pair selecton per trigger via lepton counting
        if trigger.has_tag({"single_e"}):
        #store global variables
            where = (channel_id == 0) 
            single_triggered = ak.where(where & is_single, True, single_triggered)
            sel_electron_indices = ak.where(where, electron_indices, sel_electron_indices)
    
    # some final type conversions
    channel_id = ak.values_astype(channel_id, np.uint8)
    leptons_os = ak.fill_none(leptons_os, False)
    sel_electron_indices = ak.values_astype(sel_electron_indices, np.int32)
    
    # save new columns
    events = set_ak_column(events, "channel_id", channel_id)
    events = set_ak_column(events, "single_triggered", single_triggered)

    return events, SelectionResult(
        steps={
            "Lepton": channel_id != 0,
        },
        objects={
            "Electron": {
                "Electron": sel_electron_indices,
            },
            # "Muon": {
            #     "Muon": sel_muon_indices,
            # },
            # "Tau": {
            #     "Tau": sel_tau_indices,
            # },
        },
        aux={
            # save the selected lepton pair for the duration of the selection
            # multiplication of a coffea particle with 1 yields the lorentz vector
            "lepton_pair": ak.concatenate(
                [
                    events.Electron[sel_electron_indices] * 1,
                    # events.Muon[sel_muon_indices] * 1,
                    # events.Tau[sel_tau_indices] * 1,
                ],
                axis=1,
            ),
        },
    )