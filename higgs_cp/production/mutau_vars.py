import functools
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT, Route
from columnflow.production.util import attach_coffea_behavior
#from IPython import embed
ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses = 
    {
        f"Muon.{var}" for var in ["pt", "eta","phi", "mass","charge"]
    } | {
        f"Tau.{var}" for var in ["pt","eta","phi", "mass", "dxy", "dz", "charge"] 
    } | {
        f"Electron.{var}" for var in ["pt","eta","phi", "mass", "dxy", "dz", "charge"] 
    } | {attach_coffea_behavior},
    produces={
        "etau_mass"
    },
)
def dilepton_mass(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    print("Producing dilepton mass...")
    events = self[attach_coffea_behavior](events, **kwargs)
    electron = ak.firsts(events.Electron, axis=1)
    tau = ak.firsts(events.Tau, axis=1)
    etau_obj = electron + tau
    etau_mass = ak.where(etau_obj.mass2 >=0, etau_obj.mass, EMPTY_FLOAT)
    events = set_ak_column_f32(events,"etau_mass",etau_mass)
    
    return events

@producer(
    uses = 
    {
        f"Electron.{var}" for var in ["pt","phi"]
    } | {
        f"PuppiMET.{var}" for var in ["pt","phi"] 
    } | {attach_coffea_behavior},
    produces={
        "Electron.mT"
    },
)
def mT(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    cos_dphi = np.cos(events.Electron.delta_phi(events.PuppiMET))
    mT_values = np.sqrt(2 * events.Electron.pt * events.PuppiMET.pt * (1 - cos_dphi))
    mT_values = ak.fill_none(mT_values, EMPTY_FLOAT)
    events = set_ak_column_f32(events, Route("Electron.mT"), mT_values)
    return events
    

    
   