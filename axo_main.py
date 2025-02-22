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
import datetime
import dill
import hist
import hist.dask as hda
import json
import numpy as np
import time
import vector
vector.register_awkward()
import yaml

# coffea imports
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import coffea.processor as processor
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)

from ScoutingNanoAODSchema import ScoutingNanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

hist_selection = {
    "1d_scalar": [
        # "anomaly_score",                       # axol1tl anomaly score
        # "l1ht",                               # ht of l1 objects
        # "l1met",                              # MET of l1 objects
        # "total_l1mult",                       # total l1 object multiplicity
        # "total_l1pt",                         # total l1 pt
        # "scoutinght",                         # ht of scouting objects
        # "scoutingmet",                        # MET of scouting objects
        # "total_scoutingmult",                 # total scouting object multiplicity
        # "total_scoutingpt",                   # total scouting pt
    ],
    "2d_scalar": [
        # "anomaly_score_l1ht",               
        # "anomaly_score_l1met",             
        # "anomaly_score_total_l1mult",         
        # "anomaly_score_total_l1pt",
        # "anomaly_score_scoutinght",
        # "anomaly_score_scoutingmet",
        # "anomaly_score_total_scoutingmult",   
        # "anomaly_score_total_scoutingpt",
    ],
    "1d_object": [
        "n"                                   # object multiplicity
        # "pt",                                  # object pt
        # "pt0",                                 # leading object pt
        # "pt1",                                 # subleading object pt
        # "eta",                                 # object eta
        # "phi",                                 # object phi
    ],
    "2d_object": [
        # "anomaly_score_n",
        # "anomaly_score_pt",
        # "anomaly_score_eta",
        # "anomaly_score_phi",
    ],
    "1d_diobject": [ 
        # "m_log",                               # log axis for diobject invariant mass
        # "m_low",                               # low range axis for diobject invariant mass
        # "m_mid",                               # mid range axis for diobject invariant mass
        # "m",                                   # full range axis for diobject invariant mass
    ],
    "2d_diobject": [
        # "anomaly_score_m_log",
        # "anomaly_score_m_low",
        # "anomaly_score_m_mid",
        # "anomaly_score_m",
    ],
    "dimuon": [
        # "m_log",                               # log axis for dimuon invariant mass
        # "m_low",                               # low range axis for dimuon invariant mass
        # "m_mid",                               # mid range axis for dimuon invariant mass
        # "m",                                   # full range axis for dimuon invariant mass
    ]
}


# # # which hists to save (comment out unwanted)
# hist_selection = {
#     "1d_scalar": {
#         "anomaly_score": self.score_axis,                # axol1tl anomaly score
#         "ht":self.ht_axis,                               # ht of objects
#         "met":self.met_axis,                             # MET of objects
#         "total_mult":self.mult_axis,                     # total object multiplicity
#         "total_pt":self.pt_axis,                         # total pt
#     },
#     "1d_per_object_type": {
#         "n":self.mult_axis,                               # object multiplicity
#         "pt":self.pt_axis,                                 # object pt
#         "eta":self.eta_axis,                              # object eta
#         "phi":self.phi_axis,                              # object phi
#     },
#     "1d_per_object": {
#         "pt":self.pt_axis,                                  # object pt
#         "eta":self.eta_axis,                                 # object eta
#         "phi":self.phi_axis,                                 # object phi
#     },
#     "2d_scalar": {
#         "anomaly_score_ht":(self.score_axis, self.ht_axis),               
#         "anomaly_score_met":(self.score_axis, self.met_axis),             
#         "anomaly_score_total_mult":(self.score_axis, self.mult_axis),         
#         "anomaly_score_total_pt":(self.score_axis, self.pt_axis)
#     },
#     "2d_per_object_type": {
#         "anomaly_score_n":(self.score_axis,self.mult_axis),
#         "anomaly_score_pt":(self.score_axis,self.pt_axis),
#         "anomaly_score_eta":(self.score_axis,self.eta_axis),
#         "anomaly_score_phi":(self.score_axis,self.phi_axis),
#     },
#     "1d_diobject": { 
#         "m_log",                               # log axis for diobject invariant mass
#         "m_low",                               # low range axis for diobject invariant mass
#         "m_mid",                               # mid range axis for diobject invariant mass
#         "m",                                   # full range axis for diobject invariant mass
#     },
#     "2d_diobject": {
#         "anomaly_score_m_log",
#         "anomaly_score_m_low",
#         "anomaly_score_m_mid",
#         "anomaly_score_m",
#     },
#     "dimuon": {
#         "m_log",                               # log axis for dimuon invariant mass
#         "m_low",                               # low range axis for dimuon invariant mass
#         "m_mid",                               # mid range axis for dimuon invariant mass
#         "m",                                   # full range axis for dimuon invariant mass
#     }
# }

# ###################################################################################################
# # HELPER FUNCTIONS FOR PROCESSOR
def find_diObjs(events_obj_coll, isL1, isScouting):
    
    objs_dict = {
        "pt": events_obj_coll.pt,
        "eta": events_obj_coll.eta,
        "phi": events_obj_coll.phi,
    }
    
    # set up four-vectors based on what kind of object we are dealing with
    if isL1:
        objs = dak.zip(
            {
                "pt": events_obj_coll.pt,
                "eta": events_obj_coll.eta,
                "phi": events_obj_coll.phi,
                "mass": dak.zeros_like(events_obj_coll.pt),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )
    elif isScouting:
        try:
            objs = dak.zip(
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
            objs = dak.zip(
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
        objs = dak.zip({ 
            k: getattr(events_obj_coll, k) for k in ["x", "y", "z", "t"] }, 
            with_name="LorentzVector", 
            behavior=events_obj_coll.behavior, 
        )
        
    # get combinations
    diObjs = dak.combinations(objs, 2, fields=["obj1", "obj2"])
    diObj = dak.zip(
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

def create_hist_1d(
    hist_dict, dataset_axis, trigger_axis, observable_axis, hist_name, object_axis=None 
):
    """Creates a 1D histogram and adds it to the provided histogram dictionary."""

    if object_axis==None:
        h = hda.hist.Hist(dataset_axis, trigger_axis, observable_axis, storage="weight", label="nEvents")
    else:
        h = hda.hist.Hist(dataset_axis, trigger_axis, object_axis, observable_axis, storage="weight", label="nEvents")
        
    hist_dict[f'{hist_name}'] = h
    
    return hist_dict

def fill_hist_1d(
    hist_dict, hist_name, dataset, observable, trigger_path, observable_name, object_name=None
):
    """Fills a 1D histogram and adds it to the provided histogram dictionary."""
    
    kwargs = {
        observable_name: observable,
        "dataset": dataset,
        "trigger": trigger_path
    }
    
    if object_name!=None:
        kwargs["object"] = object_name
    
    hist_dict[f'{hist_name}'].fill(**kwargs)
    
    return hist_dict

def create_hist_2d(
    hist_dict, dataset_axis, trigger_axis, observable1_axis, observable2_axis, hist_name, object_axis = None 
):
    """Creates a 2D histogram and adds it to the provided histogram dictionary."""
    if object_axis==None:
        h = hda.hist.Hist(dataset_axis, trigger_axis, observable1_axis, observable2_axis, storage="weight", label="nEvents")
    else:
        h = hda.hist.Hist(dataset_axis, trigger_axis, object_axis, observable1_axis, observable2_axis, storage="weight", label="nEvents")
        
    hist_dict[f'{hist_name}'] = h
    
    return hist_dict

def fill_hist_2d(
    hist_dict, hist_name, dataset, observable1, observable2, trigger_path, observable1_name, observable2_name, object_name = None
):
    """Fills a 2D histogram and adds it to the provided histogram dictionary."""  
    kwargs = {
        observable1_name: observable1,
        observable2_name: observable2,
        "dataset": dataset,
        "trigger": trigger_path
    }
    
    if object_name!=None:
        kwargs["object"] = object_name
    
    hist_dict[f'{hist_name}'].fill(**kwargs)
    
    return hist_dict

def load_config(config_path="config.yaml"):
    """Loads YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_dataset(json_filename, dataset_name, n_files):
    """Loads JSON dataset and filters files based on n_files limit."""
    with open(json_filename, 'r') as f:
        dataset = json.load(f)
    
    if n_files == -1:
        return {dataset_name: {'files': dataset[dataset_name]['files']}}

    # Use dictionary slicing for efficiency
    return {dataset_name: {'files': dict(list(dataset[dataset_name]['files'].items())[:n_files])}}

def preprocess_dataset(dataset, config):
    """Handles preprocessing of dataset."""
    tstart = time.time()
    
    if config["do_preprocessing"]:
        dataset_runnable, dataset_updated = preprocess(
            dataset,
            align_clusters=False,
            step_size=config["coffea_step_size"],
            files_per_batch=config["coffea_files_per_batch"],
            skip_bad_files=True,
            save_form=False,
        )
        with open('preprocessed.json', 'w') as f:
            json.dump((dataset_runnable, dataset_updated), f)
        print("Saved preprocessed filelist as preprocessed.json")
    else:
        with open('preprocessed.json', 'r') as f:
            print("Loading preprocessed filelist from preprocessed.json")
            dataset_runnable, dataset_updated = json.load(f)

    print(f'{time.time() - tstart:.1f}s to pre-process')
    return dataset_runnable

def process_histograms(dataset_runnable, config):
    """Runs the Dask-based histogram processing."""
    tstart = time.time()

    to_compute = apply_to_fileset(
        MakeAXOHists(
            trigger_paths=config["triggers"],
            objects=config["objects"],
            hists_to_process=hist_selection,
            branches_to_save=config["branch_selection"],
            has_scores=config["has_scores"], 
            axo_version=config["axo_version"],
            config=config
        ),
        max_chunks(dataset_runnable, config["coffea_max_chunks"]),
        schemaclass=ScoutingNanoAODSchema,
        uproot_options={"allow_read_errors_with_report": (OSError, TypeError, KeyError)}
    )

    if config["visualize_task_graph"]:
        dask.optimize(to_compute)
        dask.visualize(to_compute, filename=f"dask_coffea_graph_{datetime.date.today().strftime('%Y%m%d')}", format="pdf")

    (hist_result,) = dask.compute(to_compute)
    print(f'{time.time() - tstart:.1f}s to process')

    return hist_result[0]

def save_histogram(hist_result, dataset_name):
    """Saves the histogram to a pickle file."""
    filename = f'hist_result_{dataset_name}_{datetime.date.today().strftime("%Y%m%d")}.pkl'
    with open(filename, 'wb') as f:
        dill.dump(hist_result, f)
    print(f"Histogram saved as {filename}")

def get_per_event_hist_values(reconstruction_level, histogram, events_trig):
    """Retrieve histogram values based on reconstruction level and histogram type. Uses a dictionary lookup with lambda functions to avoid unnecessary computations."""
    level_map = {
        "l1": {
            "ht": lambda: dak.flatten(events_trig.L1EtSum.pt[(events_trig.L1EtSum.etSumType == 1) & (events_trig.L1EtSum.bx == 0)]),
            "met": lambda: dak.flatten(events_trig.L1EtSum.pt[(events_trig.L1EtSum.etSumType == 2) & (events_trig.L1EtSum.bx == 0)]),
            "mult": lambda: (
                dak.num(events_trig.L1Jet.bx[events_trig.L1Jet.bx == 0]) +
                dak.num(events_trig.L1Mu.bx[events_trig.L1Mu.bx == 0]) +
                dak.num(events_trig.L1EG.bx[events_trig.L1EG.bx == 0])
            ),
            "pt": lambda: (
                dak.sum(events_trig.L1Jet.pt[events_trig.L1Jet.bx == 0], axis=1) +
                dak.sum(events_trig.L1Mu.pt[events_trig.L1Mu.bx == 0], axis=1) +
                dak.sum(events_trig.L1EG.pt[events_trig.L1EG.bx == 0], axis=1)
            ),
        },
        "scouting": {
            "ht": lambda: dak.sum(events_trig.ScoutingPFJet.pt, axis=1),
            "met": lambda: events_trig.ScoutingMET.pt,
            "mult": lambda: (
                dak.num(events_trig.ScoutingPFJet) +
                dak.num(events_trig.ScoutingElectron) +
                dak.num(events_trig.ScoutingPhoton) +
                dak.num(events_trig.ScoutingMuonNoVtx)
            ),
            "pt": lambda: (
                dak.sum(events_trig.ScoutingPFJet.pt, axis=1) +
                dak.sum(events_trig.ScoutingElectron.pt, axis=1) +
                dak.sum(events_trig.ScoutingPhoton.pt, axis=1) +
                dak.sum(events_trig.ScoutingMuonNoVtx.pt, axis=1)
            ),
        },
        "full_reco": {
            "ht": lambda: dak.sum(events_trig.Jet.pt, axis=1),
            "met": lambda: events_trig.MET.pt,
            "mult": lambda: (
                dak.num(events_trig.Jet) +
                dak.num(events_trig.Electron) +
                dak.num(events_trig.Photon) +
                dak.num(events_trig.Muon)
            ),
            "pt": lambda: (
                dak.sum(events_trig.Jet.pt, axis=1) +
                dak.sum(events_trig.Electron.pt, axis=1) +
                dak.sum(events_trig.Photon.pt, axis=1) +
                dak.sum(events_trig.Muon.pt, axis=1)
            ),
        },
    }
    
    return level_map.get(reconstruction_level, {}).get(histogram, None)()

def get_per_object_type_hist_values(objects, histogram):
    """Retrieve histogram values based on reconstruction level and histogram type. Uses a dictionary lookup with lambda functions to avoid unnecessary computations."""
    level_map = {
        "ht": lambda: dak.flatten(objects.pt),
        "mult": lambda: dak.num(objects),
        "pt": lambda:  dak.flatten(objects.pt),
        "eta": lambda:  dak.flatten(objects.eta),
        "phi": lambda: dak.flatten(objects.phi)
    }
    return level_map.get(histogram, {})

def get_per_object_hist_values(objects, i, histogram):
    """Retrieve histogram values based on reconstruction level and histogram type. Uses a dictionary lookup with lambda functions to avoid unnecessary computations."""
    level_map = {
        "pt": lambda:  dak.flatten(objects.pt[:,i:i+1]),
        "eta": lambda:  dak.flatten(objects.eta[:,i:i+1]),
        "phi": lambda: dak.flatten(objects.phi[:,i:i+1]),
    }
    return level_map.get(histogram, {})


def clean_objects(objects, cuts):
    level_map = {
        "pt": lambda obj, cut:  dak.flatten(objects.pt[:,i:i+1]),
        "eta": lambda obj, cut:  dak.flatten(objects.eta[:,i:i+1]),
        "phi": lambda obj, cut: dak.flatten(objects.phi[:,i:i+1]),
    }
    
    for cut in cuts:
        mask = None
        if cut == "pt":
            mask = (getattr(br,"pt") > cut)
        elif cut == "pt_leq":
            mask = (getattr(br,"pt") <= cut)
        elif cut == "eta":
            mask = (dak.abs(getattr(br,"eta")) <= cut)
        if mask:
            objects = objects[mask]

    return objects
    

# ###################################################################################################
# # DEFINE COFFEA PROCESSOR
class MakeAXOHists (processor.ProcessorABC):
    def __init__(
        self, 
        trigger_paths=[],
        objects=[],
        hists_to_process={
            "1d_scalar": [],
            "2d_scalar": [],
            "1d_object": [],
            "2d_object": [],
            "1d_diobject": [],
            "2d_diobject": [],
            "dimuon": [],
        },
        branches_to_save={
            "dimuon": [],
        },
        has_scores=True,
        axo_version="v4",
        thresholds=None, 
        object_dict=None,
        config=None
    ):

        the_object_dict =  {'ScoutingPFJet' :      {'cut' : [('pt', 30.)], 'label' : 'j'},
                            'ScoutingElectron' : {'cut' : [('pt', 10)], 'label' : 'e'},
                            'ScoutingMuonNoVtx' :     {'cut' : [('pt', 3)], 'label' : '\mu'},
                            'ScoutingPhoton' :     {'cut' : [('pt', 10)], 'label' : '\gamma'},
                            'L1Jet' :    {'cut' : [('pt', 0.1)], 'label' : 'L1j'},
                            'L1EG' :     {'cut' : [('pt', 0.1)], 'label' : 'L1e'},
                            'L1Mu' :     {'cut' : [('pt', 0.1)], 'label' : 'L1\mu'}
                           }
    
        self.run_dict = {
            'thresholds' : thresholds if thresholds is not None else {
                'AXOVTight_EMU'  : {'name'  : 'AXO VTight', 'score' : 25000/16},
                'AXOTight_EMU'   : {'name'  : 'AXO Tight', 'score' : 20486/16},
                'AXONominal_EMU' : {'name'  : 'AXO Nominal', 'score' : 18580/16},
                'AXOLoose_EMU'   : {'name'  : 'AXO Loose', 'score' : 17596/16},
                'AXOVLoose_EMU'  : {'name'  : 'AXO VLoose', 'score' : 15717/16},
            },
            'objects' : object_dict if object_dict is not None else the_object_dict
        }

        self.sorted_keys = sorted(
            self.run_dict['thresholds'],key=lambda i: self.run_dict['thresholds'][i]['score']
        )
        self.trigger_paths = trigger_paths
        self.objects = objects
        self.has_scores = has_scores
        self.hists_to_process = hists_to_process
        self.branches_to_save = branches_to_save
        self.axo_version = axo_version
        self.config = config
        
        # Define axes for histograms
        # String based axes
        self.dataset_axis = hist.axis.StrCategory(
            [], growth=True, name="dataset", label="Primary dataset"
        )
        self.trigger_axis = hist.axis.StrCategory(
            [], growth=True, name="trigger", label="Trigger"
        )
        self.object_axis = hist.axis.StrCategory(
            [], growth=True, name="object", label="Object"
        )
        # Regular axes
        self.score_axis = hist.axis.Regular(
            100, 0, 4000, name="anomaly_score", label='Anomaly Score'
        )
        self.mult_axis = hist.axis.Regular(
            200, 0, 201, name="mult", label=r'$N_{obj}$'
        )
        self.pt_axis = hist.axis.Regular(
            500, 0, 5000, name="pt", label=r"$p_{T}$ [GeV]"
        )
        self.eta_axis = hist.axis.Regular(
            150, -5, 5, name="eta", label=r"$\eta$"
        )
        self.phi_axis = hist.axis.Regular(
            30, -4, 4, name="phi", label=r"$\phi$"
        )
        self.met_axis = hist.axis.Regular(
            250, 0, 2500, name="met", label=r"$p^{miss}_{T} [GeV]$"
        )
        self.ht_axis = hist.axis.Regular(
            100, 0, 2000, name="ht", label=r"$H_{T}$ [GeV]"
        )
        self.minv_axis = hist.axis.Regular(
            1000, 0, 3000, name="minv", label=r"$m_{obj_{1},obj_{2}}$ [GeV]"
        )
        self.minv_axis_log = hist.axis.Regular(
            1000, 0.01, 3000, name="minv_log", label=r"$m_{obj_{1},obj_{2}}$ [GeV]", 
            transform=hist.axis.transform.log
        )
        self.minv_axis_low = hist.axis.Regular(
            500, 0, 5, name="minv_low", label=r"$m_{obj_{1},obj_{2}}$ [GeV]"
        )
        self.minv_axis_mid = hist.axis.Regular(
            500, 50, 150, name="minv_mid", label=r"$m_{obj_{1},obj_{2}}$ [GeV]"
        )
        
    def process(self, events):
        dataset = events.metadata['dataset']
        cutflow = defaultdict(int)
        cutflow['start'] = dak.num(events.event, axis=0)
        hist_dict = {}
               
        # Check that the objects you want to run on match the available fields in the data
        for object_type in self.objects:
            for my_object in object_type:
                assert(my_object in events.fields,f"Error: {my_object} not in available fields: {events.fields}")
                
        # Check that the triggers you have requested are available in the data
        for trigger in self.trigger_paths:
            if trigger[0:3] == "DST":
                print(events.DST.fields)
                assert(my_object[3:] in events.DST.fields,f"Error: {trigger} not in available DST paths: {events.DST.fields}")
            if trigger[0:3] == "HLT":
                assert(my_object[3:] in events.HLT.fields,f"Error: {trigger} not in available HLT paths: {events.HLT.fields}")
            if trigger[0:2] == "L1":
                assert(my_object[2:] in events.L1.fields,f"Error: {trigger} not in available L1 paths: {events.L1.fields}")

        # Axis mapping 
        axis_map = {
            'anomaly_score': self.score_axis,
            'ht': self.ht_axis,
            'met': self.met_axis,
            'mult': self.mult_axis,
            'pt': self.pt_axis,
            'object': self.object_axis,
            'eta': self.eta_axis,
            'phi': self.phi_axis,
            'minv_log': self.minv_axis_log,
            'minv_low': self.minv_axis_low,
            'minv_mid': self.minv_axis_mid,
            'm': self.minv_axis
        }

        # Create histograms that will be filled for each trigger
        for histogram_group, histograms in self.config["histograms_1d"].items():  # Process each histogram according to its group
            for histogram in histograms: # Variables to plot
                if histogram == "anomaly_score": # Score doesn't depend on reco level
                    hist_name = f"{histogram}"  
                    hist_dict = create_hist_1d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[histogram], hist_name=hist_name) 
                    continue
                for reconstruction_level in self.config["objects"]: # Reco level
                    if histogram_group == "per_event" and histogram != "anomaly_score": # Per event ---
                        hist_name = f"{reconstruction_level}_{histogram}"
                        hist_dict = create_hist_1d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[histogram], hist_name=hist_name) 
                    elif histogram_group == "per_object_type": # Per object type ---
                        for obj in self.config["objects"][reconstruction_level]:
                            hist_name = f"{obj}_{histogram}"
                            hist_dict = create_hist_1d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[histogram], hist_name=hist_name)
                    elif histogram_group == "per_object": # Per pt ordered object ---
                        for obj in self.config["objects"][reconstruction_level]:
                            for i in range(self.config["objects_max_i"][obj]):
                                hist_name = f"{obj}_{i}_{histogram}"
                                hist_dict = create_hist_1d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[histogram], hist_name=hist_name)
                    elif histogram_group == "per_diobject_pair":  # Di-object pairs ---
                        for obj_1, obj_2 in self.config["diobject_pairings"]:
                            hist_name = f"{obj_1}_{obj_2}_{histogram}"
                            hist_dict = create_hist_1d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[histogram], hist_name=hist_name)

        print("hist_dict initialized:",hist_dict)
       
        # Trigger requirement
        for trigger_path in self.trigger_paths: # loop over trigger paths
            events_trig = None
                
            # select events for current trigger
            if trigger_path == "all":
                print("all")
                events_trig = events
            else:
                print(trigger_path)
                trig_br = getattr(events,trigger_path.split('_')[0])
                trig_path = '_'.join(trigger_path.split('_')[1:])
                events_trig = events[getattr(trig_br,trig_path)] # select events passing trigger  

                
            # save cutflow information
            cutflow[trigger_path] = dak.num(events_trig.event, axis=0)

            for histogram_group, histograms in self.config["histograms_1d"].items():
                print("Histogram group: ", histogram_group)
                for histogram in histograms:
                    print("Histogram type: ",histogram)
                    if histogram == "anomaly_score": # score hists with scalars
                        if self.has_scores:
                            if self.axo_version == "v4":
                                hist_values = events_trig.axol1tl.score_v4
                            elif self.axo_version == "v3":
                                hist_values = events_trig.axol1tl.score_v3
                        #print("hist_values",hist_values.compute())
                        fill_hist_1d(hist_dict, histogram, dataset, hist_values, trigger_path, histogram)
                    elif histogram_group == "per_event" and histogram != "anomaly_score":
                        for reconstruction_level in self.config["objects"]:
                            print("Reconstruction level: ",reconstruction_level)
                            hist_values = get_per_event_hist_values(reconstruction_level, histogram, events_trig)

                            # if reconstruction_level == "l1":
                            #     if histogram == "ht":
                            #         hist_values = dak.flatten(events_trig.L1EtSum.pt[(events_trig.L1EtSum.etSumType==1) & (events_trig.L1EtSum.bx==0)])
                            #     if histogram == "met":
                            #         hist_values = dak.flatten(events_trig.L1EtSum.pt[(events_trig.L1EtSum.etSumType==2) & (events_trig.L1EtSum.bx==0)])
                            #     if histogram == "mult":
                            #         hist_values = (dak.num(events_trig.L1Jet.bx[events_trig.L1Jet.bx == 0]) 
                            #          + dak.num(events_trig.L1Mu.bx[events_trig.L1Mu.bx == 0]) 
                            #          + dak.num(events_trig.L1EG.bx[events_trig.L1EG.bx ==0]))
                            #     if histogram == "pt":
                            #         hist_values = (dak.sum(events_trig.L1Jet.pt[events_trig.L1Jet.bx == 0],axis=1) 
                            #            + dak.sum(events_trig.L1Mu.pt[events_trig.L1Mu.bx == 0],axis=1) 
                            #            + dak.sum(events_trig.L1EG.pt[events_trig.L1EG.bx ==0],axis=1))
                            # if reconstruction_level == "scouting":
                            #     if histogram == "ht":
                            #         hist_values = dak.sum(events_trig.ScoutingPFJet.pt,axis=1)
                            #     if histogram == "met":
                            #         hist_values = events_trig.ScoutingMET.pt
                            #     if histogram == "mult":
                            #         hist_values = (dak.num(events_trig.ScoutingPFJet) 
                            #                        + dak.num(events_trig.ScoutingElectron) 
                            #                        + dak.num(events_trig.ScoutingPhoton) 
                            #                        + dak.num(events_trig.ScoutingMuonNoVtx))
                            #     if histogram == "pt":
                            #         hist_values = (dak.sum(events_trig.ScoutingPFJet.pt,axis=1) 
                            #                        + dak.sum(events_trig.ScoutingElectron.pt,axis=1)
                            #                        + dak.sum(events_trig.ScoutingPhoton.pt,axis=1)
                            #                        + dak.sum(events_trig.ScoutingMuonNoVtx.pt,axis=1))
                            # if reconstruction_level == "full_reco":
                            #     if histogram == "ht":
                            #         hist_values = dak.sum(events_trig.Jet.pt,axis=1)
                            #     if histogram == "met":
                            #         hist_values = events_trig.MET.pt
                            #     if histogram == "mult":
                            #         hist_values = (dak.num(events_trig.Jet) 
                            #                        + dak.num(events_trig.Electron) 
                            #                        + dak.num(events_trig.Photon) 
                            #                        + dak.num(events_trig.Muon))
                            #     if histogram == "pt":
                            #         hist_values = (dak.sum(events_trig.Jet.pt,axis=1) 
                            #                        + dak.sum(events_trig.Electron.pt,axis=1) 
                            #                        + dak.sum(events_trig.Photon.pt,axis=1) 
                            #                        + dak.sum(events_trig.Muon.pt,axis=1))
                            fill_hist_1d(hist_dict, reconstruction_level+"_"+histogram, dataset, hist_values, trigger_path, histogram)
                    if histogram_group != "per_event":
                        # Perform object level cleaning 
                        for reconstruction_level, object_types in self.config["objects"].items():
                            for object_type in object_types:
                                print(object_type)
                                objects = getattr(events_trig, object_type)
                                
                                if reconstruction_level == "l1":
                                    objects = objects[objects.bx==0] # filter for bunch crossing == 0 

                                # Apply cuts defined in config
                                #objects = clean_objects(objects, self.config["object_cleaning"])


                                # for cut in self.config["object_cleaning"]:
                                #     mask = None
                                #     if cut == "pt":
                                #         mask = (getattr(br,"pt") > cut)
                                #     elif cut == "pt_leq":
                                #         mask = (getattr(br,"pt") <= cut)
                                #     elif cut == "eta":
                                #         mask = (dak.abs(getattr(br,"eta")) <= cut)
                                #     if mask:
                                #         objects = objects[mask]
                                    
                                if histogram_group == "per_object_type":  
                                    hist_values = get_per_object_type_hist_values(objects, histogram)
                                    # if histogram == "ht":
                                    #     hist_values = dak.flatten(objects.pt)
                                    # elif histogram == "mult":
                                    #     hist_values = dak.num(objects)
                                    # elif histogram == "pt":
                                    #     hist_values =  dak.flatten(objects.pt)
                                    # elif histogram == "eta":
                                    #     hist_values =  dak.flatten(objects.eta)
                                    # elif histogram == "phi":
                                    #     hist_values =  dak.flatten(objects.phi)
                                        
                                    fill_hist_1d(
                                            hist_dict, 
                                            object_type+"_"+histogram, 
                                            dataset, 
                                            hist_values, 
                                            trigger_path, 
                                            histogram
                                        )
        
                                # if histogram_group == "per_object": 
                                #     for i in range(self.config["objects_max_i"][object_type]):
                                #         hist_values = get_per_object_hist_values(objects, i, histogram)
                                #         fill_hist_1d(
                                #                 hist_dict, 
                                #                 object_type+"_"+str(i)+"_"+histogram, 
                                #                 dataset, 
                                #                 hist_values, 
                                #                 trigger_path, 
                                #                 histogram
                                #             )
                        

                            


                    
        # # dimuon analysis
        # if len(self.hists_to_process["dimuon"])>0:
        #     # At least two opposite sign muons
        #     events = events[(dak.num(events.ScoutingMuonNoVtx,axis=1)>=2) & (dak.sum(events.ScoutingMuonNoVtx.charge[:,0:2],axis=1)==0)]
        #     obj = "ScoutingMuonNoVtx"
        #     obj_dict = self.run_dict['objects'][obj]
            
        #     # save branches if enabled
        #     branch_save_dict = {}
            
        #     # create histograms to fill for each trigger
        #     if ("m_log" in self.hists_to_process["dimuon"]):
        #         hist_dict = create_hist_1d(
        #             hist_dict, 
        #             self.dataset_axis, 
        #             self.trigger_axis, 
        #             self.minv_axis_log, 
        #             'dimuon_m_log'
        #         )
        #     if ("m_low" in self.hists_to_process["dimuon"]):
        #         hist_dict = create_hist_1d(
        #             hist_dict, 
        #             self.dataset_axis, 
        #             self.trigger_axis, 
        #             self.minv_axis_low, 
        #             'dimuon_m_low'
        #         )
        #     if ("m_mid" in self.hists_to_process["dimuon"]):
        #         hist_dict = create_hist_1d(
        #             hist_dict, 
        #             self.dataset_axis, 
        #             self.trigger_axis, 
        #             self.minv_axis_mid, 
        #             'dimuon_m_mid'
        #         )
        #     if ("m" in self.hists_to_process["dimuon"]):
        #         hist_dict = create_hist_1d(
        #             hist_dict, 
        #             self.dataset_axis, 
        #             self.trigger_axis, 
        #             self.minv_axis, 
        #             'dimuon_m'
        #         )

        #     for trigger_path in self.trigger_paths: # loop over trigger paths
        #         events_trig = None

        #         if trigger_path == "all":
        #             events_trig = events
        #         else:
        #             trig_br = getattr(events,trigger_path.split('_')[0])
        #             trig_path = '_'.join(trigger_path.split('_')[1:])
        #             events_trig = events[getattr(trig_br,trig_path)] # select events passing this trigger
        #         cutflow["dimuon"+trigger_path] = dak.num(events_trig.event, axis=0)

        #         cut_list = obj_dict['cut']
        #         label = obj_dict['label']
        #         isL1Obj = 'L1' in obj
        #         isScoutingObj = 'Scouting' in obj
        #         br = getattr(events_trig, obj)

        #         # Apply list of cuts to relevant branches
        #         for var, cut in cut_list:
        #             mask = (getattr(br,var) > cut)
        #             br = br[mask]        

        #         # Build di-object candidate
        #         objs = br[dak.argsort(br.pt, axis=1)]
        #         diObj = find_diObjs(objs[:,0:2], isL1Obj,isScoutingObj)

        #         if ("m_log" in self.hists_to_process["dimuon"]):
        #             hist_dict = fill_hist_1d(
        #                 hist_dict, 
        #                 'dimuon_m_log', 
        #                 dataset, 
        #                 dak.flatten(diObj.mass), 
        #                 trigger_path, 
        #                 "minv_log"
        #             )
        #         if ("m_low" in self.hists_to_process["dimuon"]):
        #             hist_dict = fill_hist_1d(
        #                 hist_dict, 
        #                 'dimuon_m_low', 
        #                 dataset, 
        #                 dak.flatten(diObj.mass), 
        #                 trigger_path, 
        #                 "minv_low"
        #             )
        #         if ("m_mid" in self.hists_to_process["dimuon"]):
        #             hist_dict = fill_hist_1d(
        #                 hist_dict, 
        #                 'dimuon_m_mid', 
        #                 dataset, 
        #                 dak.flatten(diObj.mass), 
        #                 trigger_path, 
        #                 "minv_mid"
        #             )
        #         if ("m" in self.hists_to_process["dimuon"]):
        #             hist_dict = fill_hist_1d(
        #                 hist_dict, 
        #                 'dimuon_m', 
        #                 dataset, 
        #                 dak.flatten(diObj.mass), 
        #                 trigger_path, 
        #                 "minv"
        #             )
                    
        #         # save branches if enabled
        #         if "dimuon_mass" in self.branches_to_save["dimuon"]: 
        #             branch_save_dict[f"dimuon_mass_{trigger_path}"] = dak.flatten(diObj.mass)
        #         if "dimuon_pt" in self.branches_to_save["dimuon"]: 
        #             branch_save_dict[f"dimuon_pt_{trigger_path}"] = dak.flatten(diObj.pt)
        #         if "dimuon_eta" in self.branches_to_save["dimuon"]: 
        #             branch_save_dict[f"dimuon_eta_{trigger_path}"] = dak.flatten(diObj.eta)
        #         if "dimuon_phi" in self.branches_to_save["dimuon"]: 
        #             branch_save_dict[f"dimuon_phi_{trigger_path}"] = dak.flatten(diObj.phi)
        #         if "dimuon_pt_1" in self.branches_to_save["dimuon"]: 
        #             branch_save_dict[f"dimuon_pt_1_{trigger_path}"] = dak.flatten(diObj.obj1_pt)
        #         if "dimuon_pt_2" in self.branches_to_save["dimuon"]: 
        #             branch_save_dict[f"dimuon_pt_2_{trigger_path}"] = dak.flatten(diObj.obj2_pt)
        #         if "dimuon_eta_2" in self.branches_to_save["dimuon"]: 
        #             branch_save_dict[f"dimuon_eta_1_{trigger_path}"] = dak.flatten(diObj.obj1_eta)
        #         if "dimuon_eta_2" in self.branches_to_save["dimuon"]: 
        #             branch_save_dict[f"dimuon_eta_2_{trigger_path}"] = dak.flatten(diObj.obj2_eta)
        #         if "dimuon_phi_1" in self.branches_to_save["dimuon"]: 
        #             branch_save_dict[f"dimuon_phi_1_{trigger_path}"] = dak.flatten(diObj.obj1_phi)
        #         if "dimuon_phi_2" in self.branches_to_save["dimuon"]: 
        #             branch_save_dict[f"dimuon_phi_2_{trigger_path}"] = dak.flatten(diObj.obj2_phi)
                   
        #     if len(self.branches_to_save["dimuon"])>0:
        #         dak_zip = dak.zip(branch_save_dict)
        #         dak_zip.persist().to_parquet("branches")
            
        return_dict = {}
        
        return_dict['cutflow'] = [{i:cutflow[i].compute()} for i in cutflow]#.compute()
        return_dict['hists'] = hist_dict
        return_dict['trigger'] = self.trigger_paths if len(self.trigger_paths)>0 else None
                
        return return_dict

    def postprocess(self, accumulator):
        return accumulator


###################################################################################################
# DEFINE MAIN FUNCTION

def main():
    """Main script execution."""
    config = load_config()  

    dataset_skimmed = load_dataset(config["json_filename"], config["dataset_name"], config["n_files"])
    
    print(f"Processing {len(dataset_skimmed[config['dataset_name']]['files'])} files")
    dataset_runnable = preprocess_dataset(dataset_skimmed, config)
    hist_result = process_histograms(dataset_runnable, config)
    print(hist_result)

    if config["save_hists"]:
        save_histogram(hist_result, config["dataset_name"])

    print("Finished")
  
    
###################################################################################################
# RUN SCRIPT
if __name__=="__main__":
    main()