import functools
from columnflow.production import Producer, producer
from columnflow.util import maybe_import, safe_div, InsertableDict
from columnflow.columnar_util import set_ak_column, has_ak_column,EMPTY_FLOAT, Route, optional_column as optional
from columnflow.production.util import attach_coffea_behavior

#from IPython import embed
ak     = maybe_import("awkward")
np     = maybe_import("numpy")
coffea = maybe_import("coffea")
# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

@producer(
    uses={
        "Pileup.nPU"
    },
    produces={
        "pu_weight"
    },
    mc_only=True,
)
def pu_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    nPU = events.Pileup.nPU
    pu_weight = ak.where (self.mc_weight(nPU) != 0,
                          self.data_weight(nPU)/self.mc_weight(nPU) * self.mc2data_norm,
                          0)
    #from IPython import embed
    #embed()
    events = set_ak_column_f32(events, "pu_weight", pu_weight)
    return events

@pu_weight.setup
def pu_weight_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    """
    Loads the pileup weights added through the requirements and saves them in the
    py:attr:`pu_weight` attribute for simpler access in the actual callable.
    """
    from coffea.lookup_tools import extractor
    ext = extractor()
    data_full_fname = self.config_inst.x.external_files.pileup.data
    data_name = data_full_fname.split('/')[-1].split('.')[0]
    mc_full_fname = self.config_inst.x.external_files.pileup.mc
    mc_name = mc_full_fname.split('/')[-1].split('.')[0]
    ext.add_weight_sets([f'{data_name} pileup {data_full_fname}', f'{mc_name} pileup {mc_full_fname}' ])
    ext.finalize()
    
    self.evaluator = ext.make_evaluator()
    
    mc_integral = 0.
    data_integral = 0.
    for npu in range(0,1000):
        mc_integral += self.evaluator[mc_name](npu)
        data_integral += self.evaluator[data_name](npu)
    
    self.mc_weight = self.evaluator[mc_name]
    self.data_weight = self.evaluator[data_name] 
    self.mc2data_norm = safe_div(mc_integral,data_integral)


@producer(
    uses={"genWeight", optional("LHEWeight.originalXWGTUP")},
    produces={"mc_weight"},
    # only run on mc
    mc_only=True,
)
def get_mc_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Reads the genWeight and LHEWeight columns and makes a decision about which one to save. This
    should have been configured centrally [1] and stored in genWeight, but there are some samples
    where this failed.

    Strategy:

      1. Use LHEWeight.originalXWGTUP when it exists and genWeight is always 1.
      2. In all other cases, use genWeight.

    [1] https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD?rev=99#Weigths
    """
    # determine the mc_weight
    mc_weight = np.sign(events.genWeight)
    if has_ak_column(events, "LHEWeight.originalXWGTUP") and ak.all(events.genWeight == 1.0):
        mc_weight = np.sign(events.LHEWeight.originalXWGTUP)

    # store the column
    events = set_ak_column(events, "mc_weight", mc_weight, value_type=np.float32)

    return events

@producer(
    uses={
        "Muon.pt", "Muon.eta"
    },
    produces={
        "muon_weight"
    },
    mc_only=True,
)
def muon_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    
    muon_weight = self.muon_sf(self, events.Muon.pt, events.Muon.eta)
    events = set_ak_column(events, "muon_weight", muon_weight, value_type=np.float32)

    return events

@muon_weight.setup
def muon_weight_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    from coffea.lookup_tools import extractor
    ext = extractor()
    full_fname = self.config_inst.x.external_files.muon_correction
    ext.add_weight_sets([f'sf_trig ScaleFactor_trg {full_fname}',
                         f'sf_id ScaleFactor_id {full_fname}',
                         f'sf_iso ScaleFactor_iso {full_fname}'])
    ext.finalize()
    self.evaluator = ext.make_evaluator()
    self.muon_sf = lambda self, pt, eta: self.evaluator['sf_trig'](pt,eta) * self.evaluator['sf_id'](pt,eta) * self.evaluator['sf_iso'](pt,eta)

@producer(
    uses={
        "Electron.pt", "Electron.eta"
    },
    produces={
        "electron_weight"
    },
    mc_only=True,
)
def electron_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    
    electron_weight = self.electron_sf(self, events.Electron.pt, events.Electron.eta)
    events = set_ak_column(events, "electron_weight", electron_weight, value_type=np.float32)

    return events

@electron_weight.setup
def electron_weight_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    from coffea.lookup_tools import extractor
    ext = extractor()
    full_fname = self.config_inst.x.external_files.electron_correction
    ext.add_weight_sets([f'sf_trig ScaleFactor_trg {full_fname}',
                         f'sf_id ScaleFactor_id {full_fname}',
                         f'sf_iso ScaleFactor_iso {full_fname}'])
    ext.finalize()
    self.evaluator = ext.make_evaluator()
    self.electron_sf = lambda self, pt, eta: self.evaluator['sf_trig'](pt,eta) * self.evaluator['sf_id'](pt,eta) * self.evaluator['sf_iso'](pt,eta)