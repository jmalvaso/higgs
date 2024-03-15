from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.production.util import attach_coffea_behavior
from columnflow.production.categories import category_ids
from columnflow.production.processes import process_ids
from columnflow.selection.cms.json_filter import json_filter
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from higgs_cp.selection.trigger_dev import trigger_selection

from higgs_cp.selection.lepton_trigger_matching import lepton_selection

from columnflow.util import maybe_import, dev_sandbox
from higgs_cp.production.example import cutflow_features
from columnflow.columnar_util import remove_ak_column, optional_column as optional
from collections import defaultdict, OrderedDict
from higgs_cp.production.weights import pu_weight,muon_weight, get_mc_weight
from higgs_cp.production.mutau_vars import mT 

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
        lepton_selection,
        cutflow_features,
        process_ids,
        category_ids,
        custom_increment_stats,
        mT,
    },
    produces={
        attach_coffea_behavior,
        json_filter,
        get_mc_weight,
        trigger_selection,
        lepton_selection,
        cutflow_features,
        process_ids,
        category_ids,
        custom_increment_stats,
        mT,
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
    if self.dataset_inst.is_mc:
        # add corrected mc weights
        events = self[get_mc_weight](events, **kwargs)    
    # prepare the selection results that are updated at every step
    results = SelectionResult()
    
   #trigger selection
    events, trigger_results = self[trigger_selection](events, call_force=True, **kwargs)
    events, trigger_results = self[trigger_selection](events, **kwargs)  
    results += trigger_results
   #lepton selection 
    events, lepton_results = self[lepton_selection](events, trigger_results, **kwargs)
    results += lepton_results
    
   # write out process IDs
    events = self[process_ids](events, **kwargs)
    events = self[category_ids](events, results=results, **kwargs)
    event_sel = reduce(and_, results.steps.values())
    results.event = event_sel
    
    # # keep track of event counts, sum of weights
    # # in several categories
    # weight_map = {
    #     "num_events": Ellipsis,
    #     "num_events_selected": event_sel,
    # }
    # group_map = {}
    # group_combinations = []
    
    # group_map = {
    #     **group_map,
    #         # per process
    #         "process": {
    #             "values": events.process_id,
    #             "mask_fn": (lambda v: events.process_id == v),
    #         },
    # }
    # group_combinations.append(("process",))
            
    # events, results = self[increment_stats](
    #     events,
    #     results,
    #     stats,
    #     weight_map=weight_map,
    #     group_map=group_map,
    #     group_combinations=group_combinations,
    #     **kwargs,
    # )
    events, results = self[custom_increment_stats]( 
                                                   events,
                                                   results,
                                                   stats,
    )  
    
    return events, results
