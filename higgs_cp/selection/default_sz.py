from operator import and_
from functools import reduce

from columnflow.production.util import attach_coffea_behavior
from columnflow.production.categories import category_ids
from columnflow.production.processes import process_ids
#from columnflow.production.cms.mc_weight import mc_weight #Updated the code to use sign of the genWeight
# from columnflow.production.cms.pileup import pu_weight
# from columnflow.production.cms.scale import murmuf_weights
from columnflow.selection.cms.json_filter import json_filter
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.columnar_util import remove_ak_column, optional_column as optional
from higgs_cp.selection.trigger import trigger_selection
from higgs_cp.selection.lepton  import study_muon_selection, study_tau_selection, mutau_selection, extra_lepton_veto, dilepton_veto
from higgs_cp.selection.trigger  import trigger_matching
from higgs_cp.selection.jet_veto import jet_veto
from higgs_cp.production.mutau_vars import mT 
from higgs_cp.production.weights import pu_weight,get_mc_weight,muon_weight

from columnflow.util import maybe_import, dev_sandbox
from higgs_cp.production.example import cutflow_features
from collections import defaultdict, OrderedDict

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")

@selector(uses={"process_id", optional("mc_weight")})
def custom_increment_stats(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: dict,
    **kwargs,
) -> ak.Array:
    """
    Unexposed selector that does not actually select objects but instead increments selection
    *stats* in-place based on all input *events* and the final selection *mask*.
    """
    # get event masks
    event_mask = results.event

    # get a list of unique process ids present in the chunk
    unique_process_ids = np.unique(events.process_id)
    
    # increment plain counts
    n_evt_per_file = self.dataset_inst.n_events/self.dataset_inst.n_files
    #from IPython import embed
    #embed()
    stats["num_events"] = n_evt_per_file
    stats["num_events_selected"] += ak.sum(event_mask, axis=0)
    if self.dataset_inst.is_mc:
        stats[f"sum_mc_weight"] = n_evt_per_file
        stats.setdefault(f"sum_mc_weight_per_process", defaultdict(float))
        for p in unique_process_ids:
            stats[f"sum_mc_weight_per_process"][int(p)] = n_evt_per_file
        
    # create a map of entry names to (weight, mask) pairs that will be written to stats
    weight_map = OrderedDict()
    if self.dataset_inst.is_mc:
        # mc weight for selected events
        weight_map["mc_weight_selected"] = (events.mc_weight, event_mask)

    # get and store the sum of weights in the stats dictionary
    for name, (weights, mask) in weight_map.items():
        joinable_mask = True if mask is Ellipsis else mask

        # sum of different weights in weight_map for all processes
        stats[f"sum_{name}"] += ak.sum(weights[mask])
        # sums per process id
        stats.setdefault(f"sum_{name}_per_process", defaultdict(float))
        for p in unique_process_ids:
            stats[f"sum_{name}_per_process"][int(p)] += ak.sum(
                weights[(events.process_id == p) & joinable_mask],
            )

    return events, results



@selector(
    uses={
        "event",
        attach_coffea_behavior, 
        json_filter,
        get_mc_weight,
        trigger_selection,
        trigger_matching,
        jet_veto,
        study_muon_selection,
        study_tau_selection,
        mutau_selection,
        extra_lepton_veto,
        dilepton_veto,
        process_ids,
        category_ids,
        custom_increment_stats,
        mT
    },
    produces={
        attach_coffea_behavior, 
        json_filter,
        trigger_selection,
        trigger_matching,
        jet_veto,
        get_mc_weight,
        study_muon_selection,
        study_tau_selection,
        mutau_selection,
        extra_lepton_veto,
        dilepton_veto,
        process_ids,
        category_ids,
        custom_increment_stats,
        mT
    },
    sandbox=dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar_dev.sh"),
    exposed=True,
)
def default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    
    # ensure coffea behaviors are loaded
    events = self[attach_coffea_behavior](events, **kwargs)
    events = self[mT](events, **kwargs)
    
    
    if self.dataset_inst.is_mc:
        # add corrected mc weights
        events = self[get_mc_weight](events, **kwargs)    
        
    results = SelectionResult()
    # filter bad data events according to golden lumi mask
    if self.dataset_inst.is_data:
        events, json_filter_results = self[json_filter](events, **kwargs)
        results += json_filter_results
    events, trigger_results = self[trigger_selection](events, call_force=True, **kwargs)
    results += trigger_results
    
    events, jet_veto_results = self[jet_veto](events, call_force=True, **kwargs)
    results += jet_veto_results
    
    events, muon_results, muon_mask = self[study_muon_selection](events,
                                                                 call_force=True,
                                                                 **kwargs)
    results += muon_results
    events, tau_results, tau_mask = self[study_tau_selection](events,
                                                              call_force=True,
                                                              **kwargs)
    results += tau_results
    
    events, mutau_results, pair_mu_idx, pair_tau_idx = self[mutau_selection](events,
                                                  muon_mask,
                                                  tau_mask,
                                                  call_force=True,
                                                  **kwargs)
    results += mutau_results

    events, dilepton_veto_results = self[dilepton_veto](events,
                                                            pair_mu_idx,
                                                            pair_tau_idx,
                                                            call_force=True,
                                                            **kwargs)
    results += dilepton_veto_results
    
    events, extralep_veto_results, pair_mu_idx, pair_tau_idx = self[extra_lepton_veto](events,
                                                            pair_mu_idx,
                                                            pair_tau_idx,
                                                            call_force=True,
                                                            **kwargs)
    results += extralep_veto_results
    
    print(f"Sum evt: before trig mathcing: {ak.sum(ak.num(pair_mu_idx,axis=1))}")
    events, trigger_mathcing_results = self[trigger_matching](events,
                                                            pair_mu_idx,
                                                            pair_tau_idx,
                                                            trigger_results,
                                                            call_force=True,
                                                            **kwargs)
    #from IPython import embed
    #embed()
    results += trigger_mathcing_results
    print(f"Sum evt: after trig mathcing: {ak.sum(trigger_mathcing_results.steps['trigger_matching'])}")
    # write out process IDs
    events = self[process_ids](events, **kwargs)
    events = self[category_ids](events, results=results, **kwargs)
   
   
    event_sel = reduce(and_, results.steps.values())
    results.event = event_sel
    
    #events = remove_ak_column(events, "Muon.mT")
    # some cutflow features This function interfere with get_mc_weights
    #events = self[cutflow_features](events, results.objects, **kwargs)

    # weight_map = {
    #     "num_events": 
    #     "num_events_selected": results.event
    # }
    # if self.dataset_inst.is_mc:
    #     weight_map = {
    #         **weight_map,
    #         "sum_mc_weight" : (events.mc_weight, Ellipsis),
    #         "sum_mc_weight_selected" : (events.mc_weight, results.event),
    #     }
        # pu weights with variations
        #for name in sorted(self[pu_weight].produces):
        #    weight_map[f"sum_mc_weight_{name}"] = (events.mc_weight * events[name], Ellipsis)
        # # pdf and murmuf weights with variations
        # for v in ["", "_up", "_down"]:
        #     weight_map[f"sum_pdf_weight{v}"] = events[f"pdf_weight{v}"]
        #     weight_map[f"sum_pdf_weight{v}_selected"] = (events[f"pdf_weight{v}"], event_sel)
        #     weight_map[f"sum_murmuf_weight{v}"] = events[f"murmuf_weight{v}"]
        #     weight_map[f"sum_murmuf_weight{v}_selected"] = (events[f"murmuf_weight{v}"], event_sel)
        # # btag weights
        # for name in sorted(self[btag_weights].produces):
        #     if not name.startswith("btag_weight"):
        #         continue
        #     weight_map[f"sum_{name}"] = events[name]
        #     weight_map[f"sum_{name}_selected"] = (events[name], event_sel)
        #     weight_map[f"sum_{name}_selected_nobjet"] = (events[name], event_sel_nob)
        #     weight_map[f"sum_mc_weight_{name}_selected_nobjet"] = (events.mc_weight * events[name], event_sel_nob)
            
    # group_map = {}
    
    # group_map = {
    #     **group_map,
    #         # per process
    #         "process": {
    #             "values": events.process_id,
    #             "mask_fn": (lambda v: events.process_id == v),
    #         },
    # }
    # group_combinations.append(("process",))
    events, results = self[custom_increment_stats]( 
                                                   events,
                                                   results,
                                                   stats,
    )
    
    #     results,     
    # events, results = self[increment_stats](
    #     events,
    #     results,
    #     stats,
    #     weight_map=weight_map,
    #     group_map=group_map,
    #     **kwargs,
    # )
    
    return events, results