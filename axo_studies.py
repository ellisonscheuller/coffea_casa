###################################################################################################
#   axo_studies.py                                                                                #
#   Description: process axo-triggered events and save relevant observables in histograms         #
#   Authors: Noah Zipper, Jannicke Pearkes, Ada Collins, Elliott Kauffman, Natalie Bruhwiler,     #
#            Sabrina Giorgetti                                                                    #
###################################################################################################

###################################################################################################
# IMPORTS

# library imports
import awkward as ak
from collections import defaultdict
import dask
from dask.distributed import Client
import dask_awkward as dak
import dill
import hist
import hist.dask as hda
import json
import numpy as np
import time
import vector
vector.register_awkward()

# coffea imports
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import coffea.processor as processor
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)

# local imports
import plotting_utils

###################################################################################################
# DEFINE SCHEMA
class ScoutingNanoAODSchema(NanoAODSchema):
    """ScoutingNano schema builder

    ScoutingNano is a NanoAOD format that includes Scouting objects
    """

    mixins = {
        **NanoAODSchema.mixins,
        "ScoutingPFJet": "Jet",
        "ScoutingFatJet": "Jet",
        "ScoutingMuonNoVtxDisplacedVertex": "Vertex",
        "ScoutingMuonVtxDisplacedVertex": "Vertex",
        "ScoutingElectron": "Electron",
        "ScoutingPhoton": "Photon", 
        "ScoutingMuonNoVtx": "Muon",
        "ScoutingMuonVtx": "Muon"

    }
    all_cross_references = {
        **NanoAODSchema.all_cross_references
    }
  
###################################################################################################
# HELPER FUNCTIONS FOR PROCESSOR
def find_diObjs(events_obj_coll, isL1, isScouting):
    
    objs_dict = {
        "pt": events_obj_coll.pt,
        "eta": events_obj_coll.eta,
        "phi": events_obj_coll.phi,
    }
    
    # set up four-vectors based on what kind of object we are dealing with
    if isL1:
        objs = ak.zip(
            {
                "pt": events_obj_coll.pt,
                "eta": events_obj_coll.eta,
                "phi": events_obj_coll.phi,
                "mass": ak.zeros_like(events_obj_coll.pt),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )
    elif isScouting:
        try:
            objs = ak.zip(
                {
                    "pt": events_obj_coll.pt,
                    "eta": events_obj_coll.eta,
                    "phi": events_obj_coll.phi,
                    "mass": events_obj_coll.mass,
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior,
            )
        except:
            objs = ak.zip(
                {
                    "pt": events_obj_coll.pt,
                    "eta": events_obj_coll.eta,
                    "phi": events_obj_coll.phi,
                    "mass": events_obj_coll.m,
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior,
            )
    else:
        objs = ak.zip({ 
            k: getattr(events_obj_coll, k) for k in ["x", "y", "z", "t"] }, 
            with_name="LorentzVector", 
            behavior=events_obj_coll.behavior, 
        )
        
    # get combinations
    diObjs = dak.combinations(objs, 2, fields=["obj1", "obj2"])
    diObj = ak.zip(
        {
            "p4": diObjs.obj1+diObjs.obj2,
        },
    )
    
    # get other characteristics
    diObj["obj1_pt"] = diObjs.obj1.pt
    diObj["obj2_pt"] = diObjs.obj2.pt
    diObj["obj1_eta"] = diObjs.obj1.eta
    diObj["obj2_eta"] = diObjs.obj2.eta
    diObj["obj1_phi"] = diObjs.obj1.phi
    diObj["obj2_phi"] = diObjs.obj2.phi
    diObj["pt"] = (diObjs.obj1+diObjs.obj2).pt
    diObj["eta"] = (diObjs.obj1+diObjs.obj2).eta
    diObj["phi"] = (diObjs.obj1+diObjs.obj2).phi
    diObj["mass"] = (diObjs.obj1+diObjs.obj2).mass
        
    return diObj

###################################################################################################
# DEFINE COFFEA PROCESSOR


###################################################################################################
# DEFINE MAIN FUNCTION
def main():
    

###################################################################################################
# RUN SCRIPT
if __name__=="__main__":
    main()