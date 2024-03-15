# coding: utf-8

"""
Config-related object definitions and utils.
"""

from __future__ import annotations

from typing import Callable, Any, Sequence
from columnflow.util import DotDict, maybe_import
from order import UniqueObject, TagMixin
from order.util import typed

np = maybe_import("numpy")
ak = maybe_import("awkward")

class TriggerLeg(object):
    """
    Container class storing information about trigger legs:

        - *pdg_id*: The id of the object that should have caused the trigger leg to fire.
        - *min_pt*: The minimum transverse momentum in GeV of the triggered object.
        - *trigger_bits*: Integer bit mask or masks describing whether the last filter of a trigger fired.
          See https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
          Per mask, any of the bits should match (*OR*). When multiple masks are configured, each of
          them should match (*AND*).

    For accepted types and conversions, see the *typed* setters implemented in this class.
    """

    def __init__(
        self,
        pdg_id: int | None = None,
        min_pt: float | int | None = None,
        trigger_bits: int | Sequence[int] | None = None,
    ):
        super().__init__()

        # instance members
        self._pdg_id = None
        self._min_pt = None
        self._trigger_bits = None

        # set initial values
        self.pdg_id = pdg_id
        self.min_pt = min_pt
        self.trigger_bits = trigger_bits

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"'pdg_id={self.pdg_id}, min_pt={self.min_pt}, trigger_bits={self.trigger_bits}' "
            f"at {hex(id(self))}>"
        )

    @typed
    def pdg_id(self, pdg_id: int | None) -> int | None:
        if pdg_id is None:
            return None

        if not isinstance(pdg_id, int):
            raise TypeError(f"invalid pdg_id: {pdg_id}")

        return pdg_id

    @typed
    def min_pt(self, min_pt: int | float | None) -> float | None:
        if min_pt is None:
            return None

        if isinstance(min_pt, int):
            min_pt = float(min_pt)
        if not isinstance(min_pt, float):
            raise TypeError(f"invalid min_pt: {min_pt}")

        return min_pt

    @typed
    def trigger_bits(
        self,
        trigger_bits: int | Sequence[int] | None,
    ) -> list[int] | None:
        if trigger_bits is None:
            return None

        # cast to list
        if isinstance(trigger_bits, tuple):
            trigger_bits = list(trigger_bits)
        elif not isinstance(trigger_bits, list):
            trigger_bits = [trigger_bits]

        # check bit types
        for bit in trigger_bits:
            if not isinstance(bit, int):
                raise TypeError(f"invalid trigger bit: {bit}")

        return trigger_bits


class Trigger(UniqueObject, TagMixin):
    """
    Container class storing information about triggers:

        - *name*: The path name of a trigger that should have fired.
        - *id*: A unique id of the trigger.
        - *run_range*: An inclusive range describing the runs where the trigger is to be applied
          (usually only defined by data).
        - *legs*: A list of :py:class:`TriggerLeg` objects contraining additional information and
          constraints of particular trigger legs.
        - *applies_to_dataset*: A function that obtains an ``order.Dataset`` instance to decide
          whether the trigger applies to that dataset. Defaults to *True*.

    For accepted types and conversions, see the *typed* setters implemented in this class.

    In addition, a base class from *order* provides additional functionality via mixins:

        - *tags*: Trigger objects can be assigned *tags* that can be checked later on, e.g. to
          describe the type of the trigger ("single_mu", "cross", ...).
    """

    allow_undefined_data_source = True

    def __init__(
        self,
        name: str,
        id: int,
        run_range: Sequence[int] | None = None,
        legs: Sequence[TriggerLeg] | None = None,
        applies_to_dataset: Callable | bool | Any = True,
        tags: Any = None,
    ):
        UniqueObject.__init__(self, name, id)
        TagMixin.__init__(self, tags=tags)

        # force the id to be positive
        if self.id < 0:
            raise ValueError(f"trigger id must be positive, but found {self.id}")

        # instance members
        self._run_range = None
        self._leg = None
        self._applies_to_dataset = None

        # set initial values
        self.run_range = run_range
        self.legs = legs
        self.applies_to_dataset = applies_to_dataset

    def __repr__(self):
        data_source = "" if self.data_source is None else f", {self.data_source}-only"
        return (
            f"<{self.__class__.__name__} 'name={self.name}, nlegs={self.n_legs}{data_source}' "
            f"at {hex(id(self))}>"
        )

    @typed
    def name(self, name: str) -> str:
        if not isinstance(name, str):
            raise TypeError(f"invalid name: {name}")
        if not name.startswith("HLT_"):
            raise ValueError(f"invalid name: {name}")

        return name

    @typed
    def run_range(
        self,
        run_range: Sequence[int] | None,
    ) -> tuple[int] | None:
        if run_range is None:
            return None

        # cast list to tuple
        if isinstance(run_range, list):
            run_range = tuple(run_range)

        # run_range must be a tuple with to integers
        if not isinstance(run_range, tuple):
            raise TypeError(f"invalid run_range: {run_range}")
        if len(run_range) != 2:
            raise ValueError(f"invalid run_range length: {run_range}")
        if not isinstance(run_range[0], int):
            raise ValueError(f"invalid run_range start: {run_range[0]}")
        if not isinstance(run_range[1], int):
            raise ValueError(f"invalid run_range end: {run_range[1]}")

        return run_range

    @typed
    def legs(
        self,
        legs: (
            dict |
            tuple[dict] |
            list[dict] |
            TriggerLeg |
            tuple[TriggerLeg] |
            list[TriggerLeg] |
            None
        ),
    ) -> list[TriggerLeg]:
        if legs is None:
            return None

        if isinstance(legs, tuple):
            legs = list(legs)
        elif not isinstance(legs, list):
            legs = [legs]

        _legs = []
        for leg in legs:
            if isinstance(leg, dict):
                leg = TriggerLeg(**leg)
            if not isinstance(leg, TriggerLeg):
                raise TypeError(f"invalid trigger leg: {leg}")
            _legs.append(leg)

        return _legs or None

    @typed
    def applies_to_dataset(self, func: Callable | bool | Any) -> Callable:
        if not callable(func):
            decision = True if func is None else bool(func)
            func = lambda dataset_inst: decision

        return func

    @property
    def has_legs(self):
        return bool(self._legs)

    @property
    def n_legs(self):
        return len(self.legs) if self.has_legs else 0

    @property
    def hlt_field(self):
        # remove the first four "HLT_" characters
        return self.name[4:]

###############################################################################################
###Function that creates the e-tau couples and the best one in which the electron is matched###
###############################################################################################
  
def process_e_tau_pairs_single_trigger(events, selected_electron_idxs, selected_tau_idxs, leg_masks):
    a = events.Electron[selected_electron_idxs[0]]
    num_e = ak.num(a)
    e_local_index = ak.local_index(a)
    b = events.Tau[selected_tau_idxs[0]]
    tau_local_index = ak.local_index(b)
    num_tau = ak.num(b)
    Trigger_Objs = events.TrigObj[leg_masks[0]]
    num_TrigObj = ak.num(Trigger_Objs)
    etau_couples = ak.cartesian([a, b])
    
    num_couples = ak.num(etau_couples)
    events_couple = ak.where(num_couples >= 1)
    
    # selected_tau_idxs[0][7640][ak.local_index(b)[7640][0]]
    charge_e = ak.fill_none(ak.firsts(events.Electron[selected_electron_idxs[0]].charge, axis=1),2)
    charge_tau = ak.fill_none(ak.firsts(events.Tau[selected_tau_idxs[0]].charge, axis=1),2)

    indices_for_couples = list(range(len(events_couple[0])))
    good_indices = []
    pairs_of_indices = []  # List to store pairs of indices

    for idx in events_couple[0]:
        i=0
        while i < num_e[idx]: 
            j=0
            while j < num_tau[idx]:
                is_os = a[idx].charge[i] == -b[idx].charge[j]
                Delta_R = np.sqrt((a[idx].eta[i] - b[idx].eta[j])**2 + (a[idx].phi[i] - b[idx].phi[j])**2)
                e_is_matched = False
                k = 0
                while k < num_TrigObj[idx]:
                    if len(Trigger_Objs[idx])==0:
                        break
                    else:
                        e_is_matched = np.sqrt((a[idx].eta[i] - Trigger_Objs[idx].eta[k])**2 + (a[idx].phi[i] - Trigger_Objs[idx].phi[k])**2) < 0.5
                        k+=1
                if is_os and Delta_R > 0.5 and e_is_matched:
                    good_indices.append(idx)
                    e_i = selected_electron_idxs[0][idx][ak.local_index(a)[idx][i]]
                    tau_j = selected_tau_idxs[0][idx][ak.local_index(b)[idx][j]]
                    pairs_of_indices.append((e_i, tau_j))
                
                j+=1
            i+=1
    
    # Create a dictionary to store pairs for each idx
    pairs_by_idx = {}

    # Populate the dictionary
    for idx, (i, j) in zip(good_indices, pairs_of_indices):
        current_pair = (i, j)

        if idx in pairs_by_idx:
            pairs_by_idx[idx].append(current_pair)
        else:
            pairs_by_idx[idx] = [current_pair]

    for idx, pairs_list in pairs_by_idx.items():
        if len(pairs_list) >= 2:
            a = events.Electron
            b = events.Tau
            for pair1 in pairs_list:
                for pair2 in pairs_list:
                    if pair1 != pair2:
                        i1, j1 = pair1
                        i2, j2 = pair2

                        if a[idx].pfRelIso03_all[i1] < a[idx].pfRelIso03_all[i2]:
                            pairs_by_idx[idx] = [(i1, j1)]
                        elif a[idx].pfRelIso03_all[i1] > a[idx].pfRelIso03_all[i2]:
                            pairs_by_idx[idx] = [(i2, j2)]
                        elif a[idx].pfRelIso03_all[i1] == a[idx].pfRelIso03_all[i2]:
                            if a[idx].pt[i1] > a[idx].pt[i2]:
                                pairs_by_idx[idx] = [(i1, j1)]
                            elif a[idx].pt[i1] < a[idx].pt[i2]:
                                pairs_by_idx[idx] = [(i2, j2)]
                            if a[idx].pt[i1] == a[idx].pt[i2]:
                                if b[idx].idDeepTau2018v2p5VSjet[j1] > b[idx].idDeepTau2018v2p5VSjet[j2]:
                                    pairs_by_idx[idx] = [(i1, j1)]
                                elif b[idx].idDeepTau2018v2p5VSjet[j1] < b[idx].idDeepTau2018v2p5VSjet[j2]:
                                    pairs_by_idx[idx] = [(i2, j2)]
                                if b[idx].idDeepTau2018v2p5VSjet[j1] == b[idx].idDeepTau2018v2p5VSjet[j2]:
                                    if b[idx].pt[j1] > b[idx].pt[j2]:
                                        pairs_by_idx[idx] = [(i1, j1)]
                                    elif b[idx].pt[j1] < b[idx].pt[j2]:
                                        pairs_by_idx[idx] = [(i2, j2)]

    index_events = list(pairs_by_idx.keys())
    length = len(events.event)  # Replace with your desired length
    my_list = [[False]] * length
    my_list_e = [[]] * length
    my_list_tau = [[]] * length

    for idx in index_events:
        my_list[idx] = [True]
        my_list_e[idx] = [pairs_by_idx[idx][0][0]]
        my_list_tau[idx] = [pairs_by_idx[idx][0][1]]
        
    mask_event_selected_couple = ak.Array(my_list)
    selected_e = ak.Array(my_list_e)
    selected_tau = ak.Array(my_list_tau)
    
    return pairs_by_idx, mask_event_selected_couple,selected_e,selected_tau
################################################################################################################
###Function that creates the e-tau couples and the best one in which the electron and the tau are HLT-matched###
################################################################################################################
def process_e_tau_pairs_cross_trigger(events, selected_electron_idxs, selected_tau_idxs, leg_masks):
    
    a = events.Electron[selected_electron_idxs[0]]
    num_e = ak.num(a)
    e_local_index = ak.local_index(a)
    b = events.Tau[selected_tau_idxs[0]]
    tau_local_index = ak.local_index(b)
    num_tau = ak.num(b)
    Trigger_Objs_e = events.TrigObj[leg_masks[0]]
    num_TrigObj_e = ak.num(Trigger_Objs_e)
    Trigger_Objs_tau = events.TrigObj[leg_masks[1]]
    num_TrigObj_tau = ak.num(Trigger_Objs_tau)
    etau_couples = ak.cartesian([a, b])
    
    num_couples = ak.num(etau_couples)
    events_couple = ak.where(num_couples >= 1)
    
    # selected_tau_idxs[0][7640][ak.local_index(b)[7640][0]]
    charge_e = ak.fill_none(ak.firsts(events.Electron[selected_electron_idxs[0]].charge, axis=1),2)
    charge_tau = ak.fill_none(ak.firsts(events.Tau[selected_tau_idxs[0]].charge, axis=1),2)

    indices_for_couples = list(range(len(events_couple[0])))
    good_indices = []
    pairs_of_indices = []  # List to store pairs of indices

    for idx in events_couple[0]:
        i=0
        while i < num_e[idx]: 
            j=0
            while j < num_tau[idx]:
                is_os = a[idx].charge[i] == -b[idx].charge[j]
                Delta_R = np.sqrt((a[idx].eta[i] - b[idx].eta[j])**2 + (a[idx].phi[i] - b[idx].phi[j])**2)
                e_is_matched = False
                k = 0
                while k < num_TrigObj_e[idx]:
                    if len(Trigger_Objs_e[idx])==0:
                        break
                    else:
                        e_is_matched = np.sqrt((a[idx].eta[i] - Trigger_Objs_e[idx].eta[k])**2 + (a[idx].phi[i] - Trigger_Objs_e[idx].phi[k])**2) < 0.5
                        k+=1
                tau_is_matched = False
                p=0
                while p < num_TrigObj_tau[idx]:
                    if len(Trigger_Objs_tau[idx])==0:
                        break
                    else : 
                        tau_is_matched = np.sqrt((b[idx].eta[j] - Trigger_Objs_tau[idx].eta[p])**2 + (b[idx].phi[j] - Trigger_Objs_tau[idx].phi[p])**2) < 0.5
                    if tau_is_matched:
                        break
                    p += 1
                if is_os and Delta_R > 0.5 and e_is_matched and tau_is_matched:
                    good_indices.append(idx)
                    e_i = selected_electron_idxs[0][idx][ak.local_index(a)[idx][i]]
                    tau_j = selected_tau_idxs[0][idx][ak.local_index(b)[idx][j]]
                    pairs_of_indices.append((e_i, tau_j))
                
                j+=1
            i+=1
    
    # Create a dictionary to store pairs for each idx
    pairs_by_idx = {}

    # Populate the dictionary
    for idx, (i, j) in zip(good_indices, pairs_of_indices):
        current_pair = (i, j)

        if idx in pairs_by_idx:
            pairs_by_idx[idx].append(current_pair)
        else:
            pairs_by_idx[idx] = [current_pair]

    for idx, pairs_list in pairs_by_idx.items():
        if len(pairs_list) >= 2:
            for pair1 in pairs_list:
                for pair2 in pairs_list:
                    if pair1 != pair2:
                        i1, j1 = pair1
                        i2, j2 = pair2

                        if a[idx].pfRelIso03_all[i1] < a[idx].pfRelIso03_all[i2]:
                            pairs_by_idx[idx] = [(i1, j1)]
                        elif a[idx].pfRelIso03_all[i1] > a[idx].pfRelIso03_all[i2]:
                            pairs_by_idx[idx] = [(i2, j2)]
                        elif a[idx].pfRelIso03_all[i1] == a[idx].pfRelIso03_all[i2]:
                            if a[idx].pt[i1] > a[idx].pt[i2]:
                                pairs_by_idx[idx] = [(i1, j1)]
                            elif a[idx].pt[i1] < a[idx].pt[i2]:
                                pairs_by_idx[idx] = [(i2, j2)]
                            if a[idx].pt[i1] == a[idx].pt[i2]:
                                if b[idx].idDeepTau2018v2p5VSjet[j1] > b[idx].idDeepTau2018v2p5VSjet[j2]:
                                    pairs_by_idx[idx] = [(i1, j1)]
                                elif b[idx].idDeepTau2018v2p5VSjet[j1] < b[idx].idDeepTau2018v2p5VSjet[j2]:
                                    pairs_by_idx[idx] = [(i2, j2)]
                                if b[idx].idDeepTau2018v2p5VSjet[j1] == b[idx].idDeepTau2018v2p5VSjet[j2]:
                                    if b[idx].pt[j1] > b[idx].pt[j2]:
                                        pairs_by_idx[idx] = [(i1, j1)]
                                    elif b[idx].pt[j1] < b[idx].pt[j2]:
                                        pairs_by_idx[idx] = [(i2, j2)]

    index_events = list(pairs_by_idx.keys())
    length = len(events.event)  # Replace with your desired length
    my_list = [[False]] * length
    my_list_e = [[]] * length
    my_list_tau = [[]] * length

    for idx in index_events:
        my_list[idx] = [True]
        my_list_e[idx] = [pairs_by_idx[idx][0][0]]
        my_list_tau[idx] = [pairs_by_idx[idx][0][1]]
        
    mask_event_selected_couple = ak.Array(my_list)
    selected_e = ak.Array(my_list_e)
    selected_tau = ak.Array(my_list_tau)
    
    return pairs_by_idx, mask_event_selected_couple,selected_e,selected_tau


###############################################################
def mT(events, pairs_by_idx):
    mT_dict = {}
    for key in list(pairs_by_idx.keys()):
        i = pairs_by_idx[key][0][0]
        mT = np.sqrt(2*events.Electron[key].pt[i]*events.MET[key].pt*(1-np.cos(events.Electron[key].phi[i] - events.MET[key].phi)))
        mT_dict[key] = mT
    length = len(events.event)  # Replace with your desired length
    my_list = [[False]] * length
    my_list_e = [[]] * length
    my_list_tau = [[]] * length

    for key in mT_dict.keys():
        if mT_dict[key] < 70:
            my_list[key] = [True]
            my_list_e[key] = [pairs_by_idx[key][0][0]]
            my_list_tau[key] = [pairs_by_idx[key][0][1]]
    mask_mT= ak.Array(my_list)
    selected_e = ak.Array(my_list_e)
    selected_tau = ak.Array(my_list_tau)
    return mT_dict, mask_mT, selected_e, selected_tau
###############################################################
###############################################################
def trigger_object_matching(
    vectors1: ak.Array,
    vectors2: ak.Array,
    threshold: float = 0.5,
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
###############################################################
def veto_os_leptons(events, Lepton, pairs_by_idx):
    number_of_leptons = ak.num(Lepton)
    pairs_by_idx_after_veto = pairs_by_idx
    keys_to_remove = []

    for key in pairs_by_idx_after_veto.keys():
        if number_of_leptons[key] > 1:
            i = 0
            veto_loops = False
            while i < number_of_leptons[key] and not veto_loops:
                j = 0
                while j < number_of_leptons[key]:
                    if i >= j:
                        j += 1
                        continue
                    pt_cut = 15
                    eta_cut = 2.5
                    dz_cut = 0.2
                    dxy_cut = 0.045
                    Iso_rel_cut = 0.3
                    Delta_R_cut = 0.15
                    veto = 1
                    charge_i = Lepton[key][i].charge
                    charge_j = Lepton[key][j].charge
                    os = charge_i * charge_j == -1
                    pt_i = Lepton[key][i].pt
                    pt_j = Lepton[key][j].pt
                    eta_i = abs(Lepton[key][i].eta)
                    eta_j = abs(Lepton[key][j].eta)
                    dz_i = Lepton[key][i].dz
                    dz_j = Lepton[key][j].dz
                    dxy_i = Lepton[key][i].dxy
                    dxy_j = Lepton[key][j].dxy
                    Iso_rel_i = Lepton[key][i].pfRelIso03_all
                    Iso_rel_j = Lepton[key][j].pfRelIso03_all
                    cutBased_i = Lepton[key][j].cutBased
                    cutBased_j = Lepton[key][j].cutBased
                    Delta_R = np.sqrt((Lepton[key][i].eta - Lepton[key][j].eta)**2 + (Lepton[key][i].phi - Lepton[key][j].phi)**2)
                    os_e_pair = cutBased_i == veto and cutBased_j == veto and os and pt_i > pt_cut and pt_j > pt_cut and eta_i < eta_cut and eta_j < eta_cut and dz_i < dz_cut and dz_j < dz_cut and dxy_i < dxy_cut and dxy_j < dxy_cut and Iso_rel_i < Iso_rel_cut and Iso_rel_j < Iso_rel_cut and Delta_R > Delta_R_cut
                    if os_e_pair:
                        # print("Index:", key)
                        # print("i,j:",i,j)
                        # print("os_e_pair:",os_e_pair)
                        keys_to_remove.append(key)
                        veto_loops = True
                        break  # exit the inner loop 
                    
                    j += 1
                
                i += 1
   
    # Remove the keys outside of the loop
    for key in keys_to_remove:
            del pairs_by_idx_after_veto[key]

    
    index_events = list(pairs_by_idx_after_veto.keys())
    length = len(events.event)  # Replace with your desired length
    my_list = [[False]] * length

    for idx in index_events:
        my_list[idx] = [True]

    mask_event_after_veto = ak.Array(my_list)
    return pairs_by_idx_after_veto, mask_event_after_veto
###############################################################
###############################################################

# etau selection
    # events, etau_Results = self[etau_selection]( 
    #         events,
    #         electron_indices,
    #         tau_indices,
    #         # call_force=True,
    #         **kwargs,
    #     )   


    # @selector(
#     uses = 
#     {
#         f"Electron.{var}" for var in ["pt", "eta","phi", "charge", "dxy", "dz", 
#                                   "mvaIso_WP90", "pfRelIso03_all",]
#     } | {
#         f"Tau.{var}" for var in ["pt","eta","phi", "dz", "dxy", "charge",
#                                  "idDeepTau2018v2p5VSjet", "idDeepTau2018v2p5VSe", "idDeepTau2018v2p5VSmu",] 
#     } | { attach_coffea_behavior},
#     exposed=False,
# )

# def etau_selection(
#     self: Selector,
#     events: ak.Array,
#     e_idxs: ak.Array,
#     tau_idxs : ak.Array,
#     **kwargs,
# ) -> tuple[ak.Array, ak.Array]:
    
#     false_mask = (abs(events.event) < 0)
#     true_mask = (abs(events.event) > 0)
#     preselected_e   = events.Electron[e_idxs]
#     preselected_taus    = events.Tau[tau_idxs]
   
#     etau_pairs = ak.cartesian([preselected_e, preselected_taus], axis=1)
#     etau_e, etau_tau = ak.unzip(etau_pairs)
    
#     etau_selections = {}
    
#     is_os = ((etau_e.charge * etau_tau.charge) == -1)
    
#     deltaR_selection = etau_e.delta_r(etau_tau) > 0.5
    

#     n_e = ak.num(etau_e, axis=1)
#     n_taus = ak.num(etau_tau, axis=1)
    
#     etau_selections['etau_os'] = np.array(ak.num(is_os, axis=1) != 0 , dtype=np.bool_)
#     etau_selections['etau_dr_0p5'] = np.array(ak.num(deltaR_selection, axis=1) != 0, dtype=np.bool_)

#     # Select events where 
#     # events = self[mT](events, **kwargs) #produces leading electroon mT variable
#     # etau_selections['etau_mt_50'] = np.array(ak.any(events.Electron.mT < 70, axis=1), dtype=np.bool_)
#     etau_selections['signle_e'] = np.array(ak.num(events.Electron, axis=1) == 1, dtype=np.bool_)
#     etau_selections['signle_tau'] = np.array(ak.num(events.Tau, axis=1) >= 1, dtype=np.bool_)
    
    
#     return events, SelectionResult(
#         steps = etau_selections,
#     )  

# duplicati = {}
# elementi_duplicati = {}

# for idx, (i, j) in zip(good_indices, pairs_of_indices):
#     current_pair = (i, j)
    
#     if idx in duplicati:
#         duplicati[idx].append(current_pair)
#     else:
#         duplicati[idx] = [current_pair]

# # Estrai gli elementi duplicati dalla lista
# for idx, pairs_list in duplicati.items():
#     if len(pairs_list) > 1:
#         elementi_duplicati[idx] = pairs_list

# #print("Elementi duplicati in good_indices:", elementi_duplicati)

# Now pairs_of_indices contains the pairs (i, j) for each idx in good_indices
# #print("Good Indices:", good_indices)
# #print("Pairs of Indices:", pairs_of_indices) 

# # Create a dictionary to store pairs for each idx
# pairs_by_idx = {}

# # Populate the dictionary
# for idx, (i, j) in zip(good_indices, pairs_of_indices):
#     current_pair = (i, j)
    
#     if idx in pairs_by_idx:
#         pairs_by_idx[idx].append(current_pair)
#     else:
#         pairs_by_idx[idx] = [current_pair]
# #print("Pairs by idx:", pairs_by_idx)

# for idx, pairs_list in pairs_by_idx.items():
#     #print(f"idx: {idx}")
#     for i, j in pairs_list:
#         #print(a[idx].pt[i])
#         #print(f"  i: {i}, j: {j}")

# for idx, pairs_list in pairs_by_idx.items():
#     #print(f"idx: {idx}")
#     for i, j in pairs_list:
#         #print(f"  i: {i}, j: {j}")
#     # Check if the current key has reached the stop threshold
#     if len(pairs_list) >= 2:
#         #print(f"Stopping for idx {idx} with {len(pairs_list)} couples.")

# Removing the close electrons
# tau_selections.update({"Remove_close_e": ak.all(events.Tau[selected_tau_idxs[0]].metric_table(events.Electron[selected_electron_idxs[0]]) > 0.5, axis=1)})
# tau_mask = tau_mask & ak.any(tau_selections["Remove_close_e"],axis=1)
# selection_idx = ak.values_astype(tau_mask, np.int32)
# selection_steps["Remove_close_e"] = np.array(selection_idx, dtype=np.bool_)

#   Removing the close electrons
# tau_selections.update({"Remove_close_e": ak.all(events.Tau[selected_tau_idxs[0]].metric_table(events.Electron[selected_electron_idxs[0]]) > 0.5, axis=1)})
# tau_mask = tau_mask & ak.any(tau_selections["Remove_close_e"],axis=1)
# selection_idx = ak.values_astype(tau_mask, np.int32)
#  selection_steps["Remove_close_e"] = np.array(ak.any(selection_idx,axis=1), dtype=np.bool_)        
          
'''
    # 
    a = events.Electron[selected_electron_idxs[0]]
    b = events.Tau[selected_tau_idxs[0]]
    number_e = ak.num(a)
    number_tau = ak.num(b)
    etau_couples= ak.cartesian([a,b])
    cartesian_indices = ak.argcartesian([a,b], axis=1)
    number_of_couples_per_event = ak.num(etau_couples)
    events_0_couple = ak.where(number_of_couples_per_event >= 1)
    events_1_couple = ak.where(number_of_couples_per_event == 1)
    events_2_couple = ak.where(number_of_couples_per_event >= 2)
    
    number_of_electron_in_each_event_with_at_least_1_couples = ak.num(a[events_0_couple[0]])
    number_of_tau_in_each_event_with_at_least_1_couples = ak.num(b[events_0_couple[0]])

    number_of_electron_in_each_event_with_at_least_2_couples = ak.num(a[events_2_couple[0]])
    number_of_tau_in_each_event_with_at_least_2_couples = ak.num(b[events_2_couple[0]])
    # ak.where(number_of_couples_per_event >= 1 & is_os)
    charge_e = ak.firsts(events.Electron[selected_electron_idxs[0]].charge,axis=1)
    charge_tau = ak.firsts(events.Tau[selected_tau_idxs[0]].charge,axis=1)
                        
    indices_for_couples = []
    good_indices = []
    pairs_of_indices = []  # List to store pairs of indices

    n = 0
    while n < len(events_0_couple[0]):
        indices_for_couples.append(n)
        n += 1
    
    for idx, k in zip(events_0_couple[0], indices_for_couples):
        #print("#######")
        #print("idx:", idx)
        #print(k)
        i = 0
        while i < number_of_electron_in_each_event_with_at_least_1_couples[k]:
            #print("indice degli elettroni :", i)
            j = 0
            while j < number_of_tau_in_each_event_with_at_least_1_couples[k]:
                #print("indice dei tau :", j)
                is_os = a[idx].charge[i] == -b[idx].charge[j]
                #print("is_os:", is_os)
                Delta_R = np.sqrt((a[idx].eta[i] - b[idx].eta[j])**2 + (a[idx].phi[i] - b[idx].phi[j])**2)
                #print("Delta_R:", Delta_R)
                #print("#######")
                if is_os and Delta_R > 0.5:
                    good_indices.append(idx)
                    pairs_of_indices.append((i, j))  # Append the pair (i, j) to the list
                j += 1
            i += 1

    # Create a dictionary to store pairs for each idx
    pairs_by_idx = {}

    # Populate the dictionary
    for idx, (i, j) in zip(good_indices, pairs_of_indices):
        current_pair = (i, j)
    
        if idx in pairs_by_idx:
            pairs_by_idx[idx].append(current_pair)
        else:
            pairs_by_idx[idx] = [current_pair]
    #print("Pairs by idx:", pairs_by_idx)

    for idx, pairs_list in pairs_by_idx.items():
        #print(f"idx: {idx}")
        for i, j in pairs_list:
            #print(a[idx].pt[i])
            #print(f"  i: {i}, j: {j}")

        # Check if the current key has reached the stop threshold
            if len(pairs_list) >= 2:
            #print(f"Stopping for idx {idx} with {len(pairs_list)} couples.")

            # Access and compare indices of different couples
                for pair1 in pairs_list:
                    for pair2 in pairs_list:
                        if pair1 != pair2:
                            i1, j1 = pair1
                            i2, j2 = pair2
                        # Check the rel Isolation of the eletrcons 
                            if a[idx].pfRelIso03_all[i1] < a[idx].pfRelIso03_all[i2]:
                                pairs_by_idx[idx] = [(i1,j1)]
                            # stored_indices = (idx, i1, None, None, None)
                            elif a[idx].pfRelIso03_all[i1] > a[idx].pfRelIso03_all[i2]:
                                pairs_by_idx[idx] = [(i2,j2)]
                        # Check the pt of the eletrcons if the isolation is equal 
                            if a[idx].pfRelIso03_all[i1] == a[idx].pfRelIso03_all[i2]:
                                if a[idx].pt[i1] > a[idx].pt[i2]:
                                    pairs_by_idx[idx] = [(i1,j1)]
                                elif a[idx].pt[i1] < a[idx].pt[i2]:
                                    pairs_by_idx[idx] = [(i2,j2)]
                            # Check the Isolation of the tau if the pt of electron is equal
                                if a[idx].pt[i1] == a[idx].pt[i2]:
                                    if b[idx].idDeepTau2018v2p5VSjet[j1] > b[idx].idDeepTau2018v2p5VSjet[j2]:
                                        pairs_by_idx[idx] = [(i1,j1)]
                                    elif b[idx].idDeepTau2018v2p5VSjet[j1] < b[idx].idDeepTau2018v2p5VSjet[j2]: 
                                        pairs_by_idx[idx] = [(i2,j2)]
                            # Check the pt of the taus if the Isolation of taus are equal                            
                                    if b[idx].idDeepTau2018v2p5VSjet[j1] == b[idx].idDeepTau2018v2p5VSjet[j2]:
                                        if b[idx].pt[j1] > b[idx].pt[j2]:
                                            pairs_by_idx[idx] = [(i1,j1)]
                                        elif b[idx].pt[j1] < b[idx].pt[j2]:
                                            pairs_by_idx[idx] = [(i2,j2)]           

    print(pairs_by_idx)
    
# # Assuming empty_indices is already defined
# empty_indices = ak.zeros_like(1 * events.event, dtype=bool)

# # Assuming indices_to_modify is a list or array of indices you want to set to True
# indices_to_modify = []
# for key in pairs_by_idx.keys():
#     indices_to_modify.append(key)

# # Create a boolean mask with True at specified indices
# mask = np.zeros(len(empty_indices), dtype=bool)
# mask[indices_to_modify] = True

# # Apply the boolean mask using ak.mask
# empty_indices = ak.mask(empty_indices, False)
# empty_indices = ak.mask(empty_indices, mask)

# # Print the resulting array
# print(empty_indices)
'''

# Electrons = events.Electron[mask_event_selected_couple]
#         Trigger_Objects = events.TrigObj[leg_masks[0]]
#         trigger_couples = ak.cartesian([Electrons,Trigger_Objects])
#         number_of_couples_per_event_trigger = ak.num(trigger_couples)
#         events_at_least_1_couple_trigger = ak.where(number_of_couples_per_event_trigger >= 1)
#         number_of_electrons_per_event_trigger = ak.num(Electrons[events_at_least_1_couple_trigger])
#         number_of_trig_obj_per_event_trigger = ak.num(Trigger_Objects[events_at_least_1_couple_trigger])
#         indices_for_couples_trigger = []
#         good_indices_trigger = []
#         pairs_of_indices_trigger = []  # List to store pairs of indices

#         n = 0
#         while n < len(events_at_least_1_couple_trigger[0]):
#             indices_for_couples_trigger.append(n)
#             n += 1
        
#         for idx, k in zip(events_at_least_1_couple_trigger[0], indices_for_couples_trigger):
#             i = 0
#             while i < number_of_electrons_per_event_trigger[k]:
#                 j = 0
#                 while j < number_of_trig_obj_per_event_trigger[k]:
#                     pt_match = Trigger_Objects[idx].pt[j] - 1 < Electrons[idx].pt[i] and Trigger_Objects[idx].pt[j] + 1 > Electrons[idx].pt[i]
#                     Delta_R = np.sqrt((Electrons[idx].eta[i] - Trigger_Objects[idx].eta[j])**2 + (Electrons[idx].phi[i] - Trigger_Objects[idx].phi[j])**2)
#                     if pt_match and Delta_R < 0.5:
#                         good_indices_trigger.append(idx)
#                         pairs_of_indices_trigger.append((i, j))  # Append the pair (i, j) to the list
#                     j += 1
#                 i += 1

#         # Create a dictionary to store pairs for each idx
#         pairs_by_idx_trigger = {}

#         # Populate the dictionary
#         for idx, (i, j) in zip(good_indices_trigger, pairs_of_indices_trigger):
#             current_pair = (i, j)
        
#             if idx in pairs_by_idx_trigger:
#                 pairs_by_idx_trigger[idx].append(current_pair)
#             else:
#                 pairs_by_idx_trigger[idx] = [current_pair]
#         #print("Pairs by idx:", pairs_by_idx_trigger)           

#         print(pairs_by_idx_trigger)
#         index_events = list(pairs_by_idx_trigger.keys())        
#         length = len(events.event)  # Replace with your desired length
#         my_list = [[False]] * length

#         for idx in index_events:
#             my_list[idx] = [True]

#         mask_event_e_HLT_matched = ak.Array(my_list)   
########################################################################
############# The following code select the best etau pair #############
########################################################################
    # a = events.Electron[selected_electron_idxs[0]]
    # b = events.Tau[selected_tau_idxs[0]]
    # number_e = ak.num(a)
    # number_tau = ak.num(b)
    # etau_couples= ak.cartesian([a,b])
    # cartesian_indices = ak.argcartesian([a,b], axis=1)
    # number_of_couples_per_event = ak.num(etau_couples)
    # events_0_couple = ak.where(number_of_couples_per_event >= 1)
    # events_1_couple = ak.where(number_of_couples_per_event == 1)
    # events_2_couple = ak.where(number_of_couples_per_event >= 2)
    
    # number_of_electron_in_each_event_with_at_least_1_couples = ak.num(a[events_0_couple[0]])
    # number_of_tau_in_each_event_with_at_least_1_couples = ak.num(b[events_0_couple[0]])

    # number_of_electron_in_each_event_with_at_least_2_couples = ak.num(a[events_2_couple[0]])
    # number_of_tau_in_each_event_with_at_least_2_couples = ak.num(b[events_2_couple[0]])
    # # ak.where(number_of_couples_per_event >= 1 & is_os)
    # charge_e = ak.firsts(events.Electron[selected_electron_idxs[0]].charge,axis=1)
    # charge_tau = ak.firsts(events.Tau[selected_tau_idxs[0]].charge,axis=1)
                        
    # indices_for_couples = []
    # good_indices = []
    # pairs_of_indices = []  # List to store pairs of indices

    # n = 0
    # while n < len(events_0_couple[0]):
    #     indices_for_couples.append(n)
    #     n += 1
    
    # for idx, k in zip(events_0_couple[0], indices_for_couples):
    #     i = 0
    #     while i < number_of_electron_in_each_event_with_at_least_1_couples[k]:
    #         j = 0
    #         while j < number_of_tau_in_each_event_with_at_least_1_couples[k]:
    #             is_os = a[idx].charge[i] == -b[idx].charge[j]
    #             Delta_R = np.sqrt((a[idx].eta[i] - b[idx].eta[j])**2 + (a[idx].phi[i] - b[idx].phi[j])**2)
    #             if is_os and Delta_R > 0.5:
    #                 good_indices.append(idx)
    #                 pairs_of_indices.append((i, j))  # Append the pair (i, j) to the list
    #             j += 1
    #         i += 1

    # # Create a dictionary to store pairs for each idx
    # pairs_by_idx = {}

    # # Populate the dictionary
    # for idx, (i, j) in zip(good_indices, pairs_of_indices):
    #     current_pair = (i, j)
    
    #     if idx in pairs_by_idx:
    #         pairs_by_idx[idx].append(current_pair)
    #     else:
    #         pairs_by_idx[idx] = [current_pair]
    # #print("Pairs by idx:", pairs_by_idx)

    # for idx, pairs_list in pairs_by_idx.items():
    #     #print(f"idx: {idx}")
    #     for i, j in pairs_list:
    #         #print(a[idx].pt[i])
    #         #print(f"  i: {i}, j: {j}")

    #     # Check if the current key has reached the stop threshold
    #         if len(pairs_list) >= 2:
    #         #print(f"Stopping for idx {idx} with {len(pairs_list)} couples.")

    #         # Access and compare indices of different couples
    #             for pair1 in pairs_list:
    #                 for pair2 in pairs_list:
    #                     if pair1 != pair2:
    #                         i1, j1 = pair1
    #                         i2, j2 = pair2
    #                     # Check the rel Isolation of the eletrcons 
    #                         if a[idx].pfRelIso03_all[i1] < a[idx].pfRelIso03_all[i2]:
    #                             pairs_by_idx[idx] = [(i1,j1)]
    #                         # stored_indices = (idx, i1, None, None, None)
    #                         elif a[idx].pfRelIso03_all[i1] > a[idx].pfRelIso03_all[i2]:
    #                             pairs_by_idx[idx] = [(i2,j2)]
    #                     # Check the pt of the eletrcons if the isolation is equal 
    #                         if a[idx].pfRelIso03_all[i1] == a[idx].pfRelIso03_all[i2]:
    #                             if a[idx].pt[i1] > a[idx].pt[i2]:
    #                                 pairs_by_idx[idx] = [(i1,j1)]
    #                             elif a[idx].pt[i1] < a[idx].pt[i2]:
    #                                 pairs_by_idx[idx] = [(i2,j2)]
    #                         # Check the Isolation of the tau if the pt of electron is equal
    #                             if a[idx].pt[i1] == a[idx].pt[i2]:
    #                                 if b[idx].idDeepTau2018v2p5VSjet[j1] > b[idx].idDeepTau2018v2p5VSjet[j2]:
    #                                     pairs_by_idx[idx] = [(i1,j1)]
    #                                 elif b[idx].idDeepTau2018v2p5VSjet[j1] < b[idx].idDeepTau2018v2p5VSjet[j2]: 
    #                                     pairs_by_idx[idx] = [(i2,j2)]
    #                         # Check the pt of the taus if the Isolation of taus are equal                            
    #                                 if b[idx].idDeepTau2018v2p5VSjet[j1] == b[idx].idDeepTau2018v2p5VSjet[j2]:
    #                                     if b[idx].pt[j1] > b[idx].pt[j2]:
    #                                         pairs_by_idx[idx] = [(i1,j1)]
    #                                     elif b[idx].pt[j1] < b[idx].pt[j2]:
    #                                         pairs_by_idx[idx] = [(i2,j2)]           

    # #print(pairs_by_idx)
    # index_events = list(pairs_by_idx.keys())
    
    # # 
    # # empty_indices = ak.zeros_like(1 * events.event, dtype=np.uint16)[..., None][..., :0]
    # length = len(events.event)  # Replace with your desired length
    # my_list = [[False]] * length

    # for idx in index_events:
    #     my_list[idx] = [True]

    # mask_event_selected_couple = ak.Array(my_list)
    # if is_single_e:
    #     #print("Single trigger:", trigger.hlt_field)
    #     # catch config errors
    #     assert trigger.n_legs == len(leg_masks) == 1
    #     assert abs(trigger.legs[0].pdg_id) == 11
    #     # matching the HLT paths
    #     matches_leg0_e = trigger_object_matching(events.Electron[selected_electron_idxs[0]], events.TrigObj[leg_masks[0]])
    #     tau_selections.update({"HLT_path_matched" : matches_leg0_e})
    #     tau_mask = tau_mask & ak.any(tau_selections["HLT_path_matched"],axis=1)
    #     selection_idx = ak.values_astype(tau_mask, np.int32)
    #     selection_steps["HLT_path_matched"] = np.array(selection_idx, dtype=np.bool_)
        
    # elif is_cross_e:
    #     # catch config errors
    #     assert trigger.n_legs == len(leg_masks) == 2
    #     assert abs(trigger.legs[0].pdg_id) == 11
    #     assert abs(trigger.legs[1].pdg_id) == 15
        
    #     #  matching the HLT paths
    #     matches_leg0_e = trigger_object_matching(events.Electron[selected_electron_idxs[0]], events.TrigObj[leg_masks[0]])
    #     matches_leg1_tau = trigger_object_matching(events.Tau[selected_tau_idxs[0]], events.TrigObj[leg_masks[1]])
    #     tau_selections.update({"HLT_path_matched" : ak.any(matches_leg1_tau,axis=1) & ak.any(matches_leg0_e,axis=1) })
    #     tau_mask = tau_mask & tau_selections["HLT_path_matched"]
    #     selection_idx = ak.values_astype(tau_mask, np.int32)
    #     selection_steps["HLT_path_matched"] = np.array(ak.any(selection_idx,axis=1), dtype=np.bool_)
        # matches_leg0_e = trigger_object_matching(events.Electron[selected_electron_idxs[0]], events.TrigObj[leg_masks[0]])
        # tau_selections.update({"HLT_path_matched" : matches_leg0_e})
        # tau_mask = tau_mask & ak.any(tau_selections["HLT_path_matched"],axis=1)
        # selection_idx = ak.values_astype(tau_mask, np.int32)
        # selection_steps["HLT_path_matched"] = np.array(selection_idx, dtype=np.bool_)
        #  matching the HLT paths
        # matches_leg0_e = trigger_object_matching(events.Electron[selected_electron_idxs[0]], events.TrigObj[leg_masks[0]])
        # matches_leg1_tau = trigger_object_matching(events.Tau[selected_tau_idxs[0]], events.TrigObj[leg_masks[1]])
        # tau_selections.update({"HLT_path_matched" : ak.any(matches_leg1_tau,axis=1) & ak.any(matches_leg0_e,axis=1) })
        # tau_mask = tau_mask & tau_selections["HLT_path_matched"]
        # selection_idx = ak.values_astype(tau_mask, np.int32)
        # selection_steps["HLT_path_matched"] = np.array(ak.any(selection_idx,axis=1), dtype=np.bool_)
            # else:
    #     if is_single_tau:
    #         assert trigger.n_legs == len(leg_masks) == 1
    #         assert abs(trigger.legs[0].pdg_id) == 15
    #         Lepton = events.Tau[mask_event_selected_couple]
    #         Trigger_Objects = events.TrigObj[leg_masks[0]]
    #         result_pairs_tau, tau_triggered = HLT_path_matching(events,Lepton,Trigger_Objects)
    #         mask_event_e_tau = tau_triggered
    #         (e_tau_matched, _) = ak.broadcast_arrays(ak.firsts(mask_event_e_tau)[:,np.newaxis],tau_mask)
    #         tau_selections.update({"HLT_path_matched" : ak.fill_none(e_tau_matched,False)})
    #         tau_mask = tau_mask & tau_selections["HLT_path_matched"]
    #         selection_idx = ak.values_astype(tau_mask, np.int32)
    #         selection_steps["HLT_path_matched"] = np.array(ak.any(selection_idx,axis=1), dtype=np.bool_)
    #         # mT cut
    #         mT_dict, mask_mT = mT(events, pairs_by_idx, result_pairs_tau)
    #         (mT_cut, _) = ak.broadcast_arrays(ak.firsts(mask_mT)[:,np.newaxis],tau_mask) #after firsts ak.fillNone
    #         tau_selections.update({"mT_less_70" : ak.fill_none(mT_cut,False)})
    #         tau_mask = tau_mask & tau_selections["mT_less_70"]
    #         selection_idx = ak.values_astype(tau_mask, np.int32)
    #         selection_steps["mT_less_70"] = np.array(ak.any(selection_idx,axis=1), dtype=np.bool_)   
            # matching the HLT paths
        # Lepton = events.Electron[mask_event_selected_couple]
        # # Trigger_Objects = events.TrigObj[leg_masks[0]]
        # result_pairs_e, electron_triggered = HLT_path_matching(events,Lepton,Trigger_Objects)
        # mask_event_e_tau = electron_triggered
        # (e_tau_matched, _) = ak.broadcast_arrays(ak.firsts(mask_event_e_tau)[:,np.newaxis],tau_mask)
        # tau_selections.update({"HLT_path_matched" : ak.fill_none(e_tau_matched,False)})
        # tau_mask = tau_mask & tau_selections["HLT_path_matched"]
        # selection_idx = ak.values_astype(tau_mask, np.int32)
        # selection_steps["HLT_path_matched"] = np.array(ak.any(selection_idx,axis=1), dtype=np.bool_)
        # mT cut
        # mT_dict, mask_mT = mT(events, pairs_by_idx, result_pairs_e)
        # (mT_cut, _) = ak.broadcast_arrays(ak.firsts(mask_mT)[:,np.newaxis],tau_mask)
        # tau_selections.update({"mT_less_70" : ak.fill_none(mT_cut,False)})
        # tau_mask = tau_mask & tau_selections["mT_less_70"]
        # selection_idx = ak.values_astype(tau_mask, np.int32)
        # selection_steps["mT_less_70"] = np.array(ak.any(selection_idx,axis=1), dtype=np.bool_)  
            # # matching the HLT paths
        # Lepton = events.Electron[mask_event_selected_couple]
        # Trigger_Objects = events.TrigObj[leg_masks[0]]
        # result_pairs_e, e_triggered = HLT_path_matching(events,Lepton,Trigger_Objects)
        # Lepton = events.Tau[mask_event_selected_couple]
        # Trigger_Objects = events.TrigObj[leg_masks[1]]
        # result_pairs_tau, tau_triggered = HLT_path_matching_tau_cross(events,Lepton,Trigger_Objects)
        # mask_event_e_tau = e_triggered & tau_triggered
        # (e_tau_matched, _) = ak.broadcast_arrays(ak.firsts(mask_event_e_tau)[:,np.newaxis],tau_mask)
        # tau_selections.update({"HLT_path_matched" :ak.fill_none(e_tau_matched,False)})
        # tau_mask = tau_mask & tau_selections["HLT_path_matched"]
        # selection_idx = ak.values_astype(tau_mask, np.int32)
        # selection_steps["HLT_path_matched"] = np.array(ak.any(selection_idx,axis=1), dtype=np.bool_)
        # common_keys = set(result_pairs_e.keys()) & set(result_pairs_tau.keys())
        # pairs_by_idx_cross_trigger = pairs_by_idx
        # for key in list(pairs_by_idx_cross_trigger.keys()):
        #     if key not in common_keys:
        #         del pairs_by_idx_cross_trigger[key]
            # def create_dictionary(up_to_number):
    #     my_dict = {}
    #     for i in range(0, up_to_number):
    #         my_dict[i] = f"value_{i}"  # You can set any value you want here
    #     return my_dict
    # pairs_by_idx_1 = create_dictionary(len(events.Electron))
#################################################################################
###Function that does the HLT matching per object and return a mask per events###
#################################################################################

# def HLT_path_matching(events,Lepton, Trigger_Objects):  
#     trigger_couples = ak.cartesian([Lepton, Trigger_Objects])
#     number_of_couples_per_event_trigger = ak.num(trigger_couples)
#     events_at_least_1_couple_trigger = ak.where(number_of_couples_per_event_trigger >= 1)
#     number_of_electrons_per_event_trigger = ak.num(Lepton[events_at_least_1_couple_trigger])
#     number_of_trig_obj_per_event_trigger = ak.num(Trigger_Objects[events_at_least_1_couple_trigger])
#     indices_for_couples_trigger = list(range(len(events_at_least_1_couple_trigger[0])))
#     good_indices_trigger = []
#     pairs_of_indices_trigger = []  # List to store pairs of indices

#     for idx, k in zip(events_at_least_1_couple_trigger[0], indices_for_couples_trigger):
#         i = 0
#         while i < number_of_electrons_per_event_trigger[k]:
#             j = 0
#             while j < number_of_trig_obj_per_event_trigger[k]:
#                 # pt_match = Trigger_Objects[idx].pt[j] - 1 < Lepton[idx].pt[i] and Trigger_Objects[idx].pt[j] + 1 > Lepton[idx].pt[i]
#                 Delta_R = np.sqrt((Lepton[idx].eta[i] - Trigger_Objects[idx].eta[j])**2 + (Lepton[idx].phi[i] - Trigger_Objects[idx].phi[j])**2)
#                 if Delta_R < 0.5 :
#                     good_indices_trigger.append(idx)
#                     pairs_of_indices_trigger.append((i, j))  # Append the pair (i, j) to the list
#                 j += 1
#             i += 1
#     # Create a dictionary to store pairs for each idx
#     pairs_by_idx_trigger = {}

#     # Populate the dictionary
#     for idx, (i, j) in zip(good_indices_trigger, pairs_of_indices_trigger):
#         current_pair = (i, j)

#         if idx in pairs_by_idx_trigger:
#             pairs_by_idx_trigger[idx].append(current_pair)
#         else:
#             pairs_by_idx_trigger[idx] = [current_pair]

#     index_events = list(pairs_by_idx_trigger.keys())
#     length = len(events.event)  # Replace with your desired length
#     my_list = [[False]] * length

#     for idx in index_events:
#         my_list[idx] = [True]

#     mask_event_HLT_matched = ak.Array(my_list)

#     return pairs_by_idx_trigger, mask_event_HLT_matched

# def HLT_path_matching_tau_cross(events,Lepton, Trigger_Objects):  
#     trigger_couples = ak.cartesian([Lepton, Trigger_Objects])
#     number_of_couples_per_event_trigger = ak.num(trigger_couples)
#     events_at_least_1_couple_trigger = ak.where(number_of_couples_per_event_trigger >= 1)
#     number_of_electrons_per_event_trigger = ak.num(Lepton[events_at_least_1_couple_trigger])
#     number_of_trig_obj_per_event_trigger = ak.num(Trigger_Objects[events_at_least_1_couple_trigger])
#     indices_for_couples_trigger = list(range(len(events_at_least_1_couple_trigger[0])))
#     good_indices_trigger = []
#     pairs_of_indices_trigger = []  # List to store pairs of indices

#     for idx, k in zip(events_at_least_1_couple_trigger[0], indices_for_couples_trigger):
#         i = 0
#         while i < number_of_electrons_per_event_trigger[k]:
#             j = 0
#             while j < number_of_trig_obj_per_event_trigger[k]:
#                 # pt_match = Trigger_Objects[idx].pt[j] - 1 < Lepton[idx].pt[i] and Trigger_Objects[idx].pt[j] + 1 > Lepton[idx].pt[i]
#                 Delta_R = np.sqrt((Lepton[idx].eta[i] - Trigger_Objects[idx].eta[j])**2 + (Lepton[idx].phi[i] - Trigger_Objects[idx].phi[j])**2)
#                 if Delta_R < 0.5 and abs(Lepton[idx].eta[i]) < 2.1:
#                     good_indices_trigger.append(idx)
#                     pairs_of_indices_trigger.append((i, j))  # Append the pair (i, j) to the list
#                 j += 1
#             i += 1
#     # Create a dictionary to store pairs for each idx
#     pairs_by_idx_trigger = {}

#     # Populate the dictionary
#     for idx, (i, j) in zip(good_indices_trigger, pairs_of_indices_trigger):
#         current_pair = (i, j)

#         if idx in pairs_by_idx_trigger:
#             pairs_by_idx_trigger[idx].append(current_pair)
#         else:
#             pairs_by_idx_trigger[idx] = [current_pair]

#     index_events = list(pairs_by_idx_trigger.keys())
#     length = len(events.event)  # Replace with your desired length
#     my_list = [[False]] * length

#     for idx in index_events:
#         my_list[idx] = [True]

#     mask_event_HLT_matched = ak.Array(my_list)

#     return pairs_by_idx_trigger, mask_event_HLT_matched