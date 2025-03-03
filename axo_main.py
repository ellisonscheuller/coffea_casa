###################################################################################################
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
import hist
import hist.dask as hda
from itertools import chain
import json
import numpy as np
import time
import vector
vector.register_awkward()
import yaml
import uproot

# coffea imports
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import coffea.processor as processor
from coffea.util import save
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)

from ScoutingNanoAODSchema import ScoutingNanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

import fsspec
fsspec.config.conf['xrootd'] = {'timeout': 600}


####################################################################################################
# HELPER FUNCTIONS FOR PROCESSOR

def create_four_vectors(objects, reconstruction_level):
    if reconstruction_level == "l1":
        return dak.zip(
            {
                "pt": objects.pt,
                "eta": objects.eta,
                "phi": objects.phi,
                "mass": dak.zeros_like(objects.pt),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )
    elif reconstruction_level == "scouting":
        try:
            return dak.zip(
                {
                    "pt": objects.pt,
                    "eta": objects.eta,
                    "phi": objects.phi,
                    "mass": objects.mass,
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior,
            )
        except AttributeError:
            return dak.zip(
                {
                    "pt": objects.pt,
                    "eta": objects.eta,
                    "phi": objects.phi,
                    "mass": objects.m,
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior,
            )
    else:
        return dak.zip(
            {k: getattr(objects, k) for k in ["x", "y", "z", "t"]},
            with_name="LorentzVector",
            behavior=objects.behavior,
        )


def find_diobjects(obj_coll1, obj_coll2, reconstruction_level):

    objs1 = create_four_vectors(obj_coll1, reconstruction_level)
    objs2 = create_four_vectors(obj_coll2, reconstruction_level)

    # Create all possible pairings between objects from the two collections
    diObjs = dak.cartesian({"obj1": objs1, "obj2": objs2})

    # Remove self-pairings
    if obj_coll1 is obj_coll2:
        same_object_mask = diObjs.obj1.pt != diObjs.obj2.pt
        diObjs = diObjs[same_object_mask]
    
    diObj = dak.zip(
        {
            "p4": diObjs.obj1 + diObjs.obj2,
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
        with open(config["preprocessing_file"], 'r') as f:
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
            #hists_to_process=hist_selection,
            #branches_to_save=config["branch_selection"],
            has_scores=config["has_scores"], 
            axo_version=config["axo_version"],
            config=config
        ),
        max_chunks(dataset_runnable, config["coffea_max_chunks"]),
        schemaclass=ScoutingNanoAODSchema,
        uproot_options={"allow_read_errors_with_report": (OSError, TypeError, KeyError, ValueError, RuntimeError, uproot.exceptions.KeyInFileError)}
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
    save(hist_result, filename)
    print(f"Histogram saved as {filename}")

def get_anomaly_score_hist_values(has_scores,axo_version, events_trig):
    assert has_scores, "Error, dataset does not have axol1tl scores"
    if axo_version == "v4":
        hist_values = events_trig.axol1tl.score_v4
    elif axo_version == "v3":
        hist_values = events_trig.axol1tl.score_v3
    return hist_values

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
        "ht": lambda: dak.sum(objects.pt,axis=1),
        "mult": lambda: dak.num(objects),
        "pt": lambda:  dak.flatten(objects.pt),
        "eta": lambda:  dak.flatten(objects.eta),
        "phi": lambda: dak.flatten(objects.phi),
    }
    return level_map.get(histogram, {})()

def get_per_object_hist_values(objects, i, histogram):
    """Retrieve histogram values based on reconstruction level and histogram type. Uses a dictionary lookup with lambda functions to avoid unnecessary computations."""
    level_map = {
        "pt": lambda:  dak.flatten(objects.pt[:,i:i+1]),
        "eta": lambda:  dak.flatten(objects.eta[:,i:i+1]),
        "phi": lambda: dak.flatten(objects.phi[:,i:i+1]),
    }
    return level_map.get(histogram, {})()


def clean_objects(objects, cuts, reconstruction_level=None):

    if reconstruction_level == "l1":
        objects = objects[objects.bx==0] # Filter for bunch crossing == 0 

    # Find the first valid branch to initialize the mask
    reference_branch = next((br for br in cuts if hasattr(objects, br)), None)
    if reference_branch is None:
        return objects  # No valid branches exist, return unmodified

    # Initialize mask with all values set to True
    mask = ak.ones_like(getattr(objects, reference_branch), dtype=bool)

    for br, cut in cuts.items():
        if cut and hasattr(objects, br):  # Ensure branch exists in objects
            lower_cut = cut[0] if cut[0] is not None else float('-inf')
            upper_cut = cut[1] if cut[1] is not None else float('inf')

            # Apply cuts to the mask
            mask = mask & (getattr(objects, br) > lower_cut) & (getattr(objects, br) < upper_cut)

    return objects[mask]

def get_required_observables(self):

    required_observables = {
        "per_event": set(),
        "per_object_type": set(),
        "per_object": set(),
        "per_diobject_pair": set()
    }

    # get 1d histogram observables
    for category, hist_list in self.config.get("histograms_1d", {}).items():
        required_observables[category].update(hist_list)

    # get 2d histogram observables
    for entry in self.config.get("histograms_2d", []):
        x_cat, x_var = entry["x_category"], entry["x_var"]
        y_cat, y_var = entry["y_category"], entry["y_var"]

        required_observables[x_cat].add(x_var)
        required_observables[y_cat].add(y_var)

    return required_observables

def calculate_observables(self, observables, events):

    observable_dict = {
        "per_event": {},
        "per_object_type": {},
        "per_object": {},
        "per_diobject_pair": {},
    }

    # calculate per-event observables
    if "anomaly_score" in observables["per_event"]:
        observable_dict["per_event"]["anomaly_score"] = get_anomaly_score_hist_values(
            self.has_scores, 
            self.axo_version, 
            events
        )
    for observable in observables["per_event"]:
        if observable!="anomaly_score":
            for reconstruction_level in self.config["objects"]:
                if reconstruction_level not in observable_dict["per_event"].keys():
                    observable_dict["per_event"][reconstruction_level] = {}
                observable_dict["per_event"][reconstruction_level][observable] = get_per_event_hist_values(
                    reconstruction_level, 
                    observable, 
                    events
                )

    # calculate per-object-type and per-object observables
    for reconstruction_level, object_types in self.config["objects"].items(): 
        if reconstruction_level not in observable_dict["per_object_type"].keys():
            observable_dict["per_object_type"][reconstruction_level] = {}

        if reconstruction_level not in observable_dict["per_object"].keys():
            observable_dict["per_object"][reconstruction_level] = {}
        
        for object_type in object_types:
            if object_type not in observable_dict["per_object_type"][reconstruction_level].keys():
                observable_dict["per_object_type"][reconstruction_level][object_type] = {}

            if object_type not in observable_dict["per_object"][reconstruction_level].keys():
                observable_dict["per_object"][reconstruction_level][object_type] = {}
                        
            # Get objects and apply object level cleaning
            objects = getattr(events, object_type)
            objects = clean_objects(objects, self.config["object_cleaning"][object_type], reconstruction_level)
            
            for observable in observables["per_object_type"]:
                observable_dict["per_object_type"][reconstruction_level][object_type][observable] = get_per_object_type_hist_values(
                    objects, 
                    observable
                )

            for observable in observables["per_object"]:
                for i in range(self.config["objects_max_i"][object_type]):
                    observable_dict["per_object"][reconstruction_level][object_type][f"{observable}_{i}"] = get_per_object_hist_values(
                        objects, 
                        i, 
                        observable)

        # calculate per-diobject-pair observables
        for reconstruction_level, pairings in self.config["diobject_pairings"].items():
            if reconstruction_level not in observable_dict["per_diobject_pair"].keys():
                observable_dict["per_diobject_pair"][reconstruction_level] = {}

            for pairing in pairings:
                object_type_1 = pairing[0]
                object_type_2 = pairing[1]
                if f"{object_type_1}_{object_type_2}" not in observable_dict["per_diobject_pair"][reconstruction_level].keys():
                    observable_dict["per_diobject_pair"][reconstruction_level][f"{object_type_1}_{object_type_2}"] = {}
                if object_type_1 == object_type_2: # same object
                    objects = getattr(events, object_type_1)
                    objects = clean_objects(objects, self.config["object_cleaning"][object_type_1])
                    di_objects = find_diobjects(objects[:,0:1], objects[:,1:2], reconstruction_level)
                else:
                    objects_1 = getattr(events, object_type_1)
                    objects_1 = clean_objects(objects_1, self.config["object_cleaning"][object_type_1])
                    objects_2 = getattr(events, object_type_2)
                    objects_2 = clean_objects(objects_2, self.config["object_cleaning"][object_type_2])
                    di_objects = find_diobjects(objects_1[:,0:1],objects_2[:,0:1], reconstruction_level)
                        
                for observable in observables["per_diobject_pair"]:
                    observable_dict["per_diobject_pair"][reconstruction_level][f"{object_type_1}_{object_type_2}"][observable] = dak.flatten(di_objects[observable])
                        
            

    return observable_dict
        


def run_the_megaloop(self,events_trig,hist_dict,branch_save_dict,dataset,trigger_path):

        # get dictionary of observables to compute
        required_observables = get_required_observables(self)
        print("required observables = ", required_observables)

        # compute observables
        observable_calculations = calculate_observables(self, required_observables, events_trig)

        for histogram_group, histograms in self.config["histograms_1d"].items():
            print("Histogram group: ", histogram_group)
            
            if histogram_group == "per_event": # event level histograms
                for histogram in histograms:
                    print("Histogram type: ",histogram)
                    if histogram == "anomaly_score":
                        fill_hist_1d(
                            hist_dict, 
                            histogram, 
                            dataset, 
                            observable_calculations["per_event"]["anomaly_score"], 
                            trigger_path, 
                            histogram
                        )
                    else: 
                        for reconstruction_level in self.config["objects"]:
                            print("Reconstruction level: ",reconstruction_level)
                            fill_hist_1d(
                                hist_dict, 
                                reconstruction_level+"_"+histogram, 
                                dataset, 
                                observable_calculations["per_event"][reconstruction_level][histogram], 
                                trigger_path, 
                                histogram
                            )
                            
            if histogram_group == "per_object_type" or histogram_group == "per_object": # object level histograms 
                for reconstruction_level, object_types in self.config["objects"].items(): 
                    for object_type in object_types:
                        print("Object type:",object_type)
                        for histogram in histograms:
                            print("Histogram type: ",histogram)
                            
                            if histogram_group == "per_object_type":  
                                fill_hist_1d(
                                    hist_dict, 
                                    object_type+"_"+histogram, 
                                    dataset, 
                                    observable_calculations["per_object_type"][reconstruction_level][object_type][histogram], 
                                    trigger_path, 
                                    histogram
                                )
    
                            if histogram_group == "per_object": 
                                for i in range(self.config["objects_max_i"][object_type]):
                                    fill_hist_1d(
                                        hist_dict, 
                                        object_type+"_"+str(i)+"_"+histogram, 
                                        dataset,
                                        observable_calculations["per_object"][reconstruction_level][object_type][f"{histogram}_{i}"],
                                        trigger_path, 
                                        histogram
                                    )
                                    
            elif histogram_group == "per_diobject_pair": # di-object masses etc 
                for reconstruction_level, pairings in self.config["diobject_pairings"].items():
                    for pairing in pairings:
                        print("Pairing:",pairing)
                        object_type_1 = pairing[0]
                        object_type_2 = pairing[1]
                        for histogram in histograms:
                            print("Histogram type: ",histogram)
                            fill_hist_1d(
                                hist_dict,
                                f"{object_type_1}_{object_type_2}_{histogram}", 
                                dataset, 
                                observable_calculations["per_diobject_pair"][reconstruction_level][f"{object_type_1}_{object_type_2}"][histogram], 
                                trigger_path, 
                                histogram
                            )
                            if self.config["save_branches"]:
                                branch_save_dict[f"{object_type_1}_{object_type_2}_{histogram}"] = observable_calculations["per_diobject_pair"][reconstruction_level][f"{object_type_1}_{object_type_2}"][histogram]
        
        
        
        return hist_dict, branch_save_dict
       
def initialize_hist_dict(self,hist_dict):
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
        'mass': self.mass_axis
    }
    for histogram_group, histograms in self.config["histograms_1d"].items():  # Process each histogram according to its group
        for histogram in histograms: # Variables to plot
            if histogram == "anomaly_score": # Score doesn't depend on reco level
                hist_name = f"{histogram}"  
                hist_dict = create_hist_1d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[histogram], hist_name=hist_name) 
                continue
            for reconstruction_level in self.config["objects"]: # Loop over different object reconstruction levels
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
            if histogram_group == "per_diobject_pair":  # Di-object pairs ---
                for reconstruction_level in self.config["diobject_pairings"]:
                    for pairing in self.config["diobject_pairings"][reconstruction_level]:
                        obj_1, obj_2 = pairing[0],pairing[1]
                        hist_name = f"{obj_1}_{obj_2}_{histogram}"
                        hist_dict = create_hist_1d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[histogram], hist_name=hist_name)
    return hist_dict

# ###################################################################################################
# # DEFINE COFFEA PROCESSOR
class MakeAXOHists (processor.ProcessorABC):
    def __init__(
        self, 
        trigger_paths=[],
        objects=[],
        has_scores=True,
        axo_version="v4",
        thresholds=None, 
        object_dict=None,
        config=None
    ):

        self.trigger_paths = trigger_paths
        self.objects = objects
        self.has_scores = has_scores
        self.axo_version = axo_version
        self.config = config
        
        # Define axes for histograms # TODO: maybe move this into a dictionary elsewhere 
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
            600, 0, 600, name="anomaly_score", label='Anomaly Score'
        )
        self.mult_axis = hist.axis.Regular(
            200, 0, 201, name="mult", label=r'$N_{obj}$'
        )
        self.pt_axis = hist.axis.Regular(
            500, 0, 5000, name="pt", label=r"$p_{T}$ [GeV]"
        )
        self.eta_axis = hist.axis.Regular(
            50, -5, 5, name="eta", label=r"$\eta$"
        )
        self.phi_axis = hist.axis.Regular(
            30, -4, 4, name="phi", label=r"$\phi$"
        )
        self.met_axis = hist.axis.Regular(
            250, 0, 2500, name="met", label=r"$p^{miss}_{T} [GeV]$"
        )
        self.ht_axis = hist.axis.Regular(
            200, 0, 4000, name="ht", label=r"$H_{T}$ [GeV]"
        )
        self.mass_axis = hist.axis.Regular(
            300, 0, 3000, name="mass", label=r"$m_{obj_{1},obj_{2}}$ [GeV]"
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
        branch_save_dict = {}
               
        # Check that the objects you want to run on match the available fields in the data
        for object_type in self.config["objects"]:
            for my_object in self.config["objects"][object_type]:
                assert my_object in events.fields, f"Error: {my_object} not in available fields: {events.fields}" 
                
        # Check that the triggers you have requested are available in the data
        print("Trigger paths:",self.trigger_paths)
        for trigger in self.trigger_paths:
            print("Trigger",trigger)
            if trigger[0:3] == "DST":
                assert trigger[4:] in events.DST.fields, f"Error: {trigger[4:]} not in available DST paths: {events.DST.fields}"
            if trigger[0:3] == "HLT":
                assert trigger[4:] in events.HLT.fields, f"Error: {trigger[4:]} not in available HLT paths: {events.HLT.fields}"
            if trigger[0:2] == "L1":
                assert trigger[3:] in events.L1.fields, f"Error: {trigger[3:]} not in available L1 paths: {events.L1.fields}"


        # Initialize histograms that will be filled for each trigger
        hist_dict = initialize_hist_dict(self,hist_dict)
        print("hist_dict initialized:",hist_dict)
       
        # Run the different modules
        if self.config["module"] == "default" or self.config["module"] == "purity": 
            # This module makes 1D histograms for the triggers and objects specified in the configuration file
            assert ("test" in self.config["dataset_name"]) or ("10" in self.config["dataset_name"], 
                   "Error: cannot run default behaviour on entire dataset, stay below 10% e.g. 2024I_10")  #don't unblind!
            
            for trigger_path in self.trigger_paths: # loop over trigger paths
                events_trig = None
                    
                # select events for current trigger
                if trigger_path == "all_available_triggers":
                    print("all_available_triggers")
                    events_trig = events
                else:
                    print(trigger_path)
                    trig_br = getattr(events,trigger_path.split('_')[0])
                    trig_path = '_'.join(trigger_path.split('_')[1:])
                    events_trig = events[getattr(trig_br,trig_path)] # select events passing trigger  
    
                # save cutflow information
                cutflow[trigger_path] = dak.num(events_trig.event, axis=0)

                # run over all objects specified in the configuration file
                hist_dict, branch_save_dict = run_the_megaloop(self, events_trig, hist_dict, branch_save_dict,dataset,trigger_path)
                                            
                if self.config["save_branches"]: 
                    dak_zip = dak.zip(branch_save_dict)
                    dak_zip.persist().to_parquet(self.config["branch_writing_path"] + "axo_branches")

        
        if self.config["module"] == "efficiency":
            # This module that makes uses the orthogonal method to study trigger efficiencies 

            # select out events passing the orthogonal trigger 
            ortho_trig = self.config["orthogonal_trigger"]
            ortho_trig_br = getattr(events,ortho_trig.split('_')[0])
            ortho_trig_path = '_'.join(ortho_trig.split('_')[1:])
            events_ortho = events[getattr(ortho_trig_br,ortho_trig_path)]
            # save cutflow and distributions of the orthogonal trigger
            cutflow[ortho_trig] = dak.num(events_ortho.event, axis=0)
            hist_dict, branch_save_dict = run_the_megaloop(self, events_ortho, hist_dict, branch_save_dict,dataset,ortho_trig)

            new_trigger_paths = []
            for trigger_path in self.trigger_paths: # loop over trigger paths
                    print(trigger_path)
                    events_trig = None
                        
                    # select events for current trigger
                    if trigger_path == "all_available_triggers":
                        print("all_available_triggers")
                        events_trig = events_ortho
                    elif trigger_path == "all_l1_triggers":
                        events_br = getattr(events_ortho, "L1")
                        events_l1_selection = dak.zeros_like(getattr(events_br,"ZeroBias"))
                        fields = [f for f in events_ortho.L1.fields if not ("CICADA" in f or "AXO" in f or "ZeroBias" in f)]
                        #print(fields)
                        for i in fields:
                            events_l1_selection = dak.where(getattr(events_ortho.L1,i)==1, 1, events_l1_selection) # if triggered by a different bit set to 0
                        events_l1_selection_bool = dak.values_astype(events_l1_selection,bool)
                        events_trig = events_ortho[events_l1_selection_bool]

                    else: # select events passing the orthogonal dataset and the trigger of interest
                        trig_br = getattr(events_ortho,trigger_path.split('_')[0])
                        trig_path = '_'.join(trigger_path.split('_')[1:])
                        events_trig = events_ortho[getattr(trig_br,trig_path)] 

                    new_trigger_path = f"{ortho_trig}_{trigger_path}"
                    new_trigger_paths += [new_trigger_path]
        
                    # save cutflow information for the trigger of interest
                    cutflow[new_trigger_path] = dak.num(events_trig.event, axis=0)

                    # run over all objects specified in the configuration file
                    hist_dict, branch_save_dict = run_the_megaloop(self, events_trig, hist_dict, branch_save_dict,dataset,new_trigger_path)
                
            self.trigger_paths += new_trigger_paths 
            if ortho_trig not in self.trigger_paths:
                self.trigger_paths += [ortho_trig]

        if self.config["module"] == "purity":
            # This module looks at the purity of the triggered events with respect to other triggers
            assert ("test" in self.config["dataset_name"]) or ("10" in self.config["dataset_name"], 
                   "Error: cannot run default behaviour on entire dataset, stay below 10% e.g. 2024I_10") #don't unblind!
            #for pure_wrt_to in ["DST"]:#,"HLT","DST"]:
            for pure_wrt_to in ["L1"]:#,"HLT","DST"]:
                #pure_wrt_to = "L1"#, "HLT", "DST"]
                print(events.fields)
                events_br = getattr(events, pure_wrt_to)
                #events_pure_selection = dak.ones_like(getattr(events_br,"PFScouting_ZeroBias"))
                events_pure_selection = dak.ones_like(getattr(events_br,"ZeroBias"))
                
                fields = [f for f in events_br.fields if not ("CICADA" in f or "AXO" in f or "ZeroBias" in f)]
                # ---
                for i in fields:
                    events_pure_selection = dak.where(getattr(events_br,i)==1, 0, events_pure_selection) # if triggered by a different bit set to 0
                events_pure = events[dak.values_astype(events_pure_selection,bool)]
                # #----
                # mask = dak.ones_like(getattr(events_br,"ZeroBias"),dtype=bool)#dak.ones_like(events_pure_selection, dtype=bool)

                # # Iterate over the fields and apply the condition to the mask
                # for i in fields:
                #     mask = mask & (getattr(events_br, i) == 1)  # Collect which trigger bits are set to 1 
                
                # # Apply the final mask
                # events_pure_selection = dak.where(mask, events_pure_selection, 0)
                # events_pure = events[dak.values_astype(events_pure_selection,bool)]

                new_trigger_paths = []
                for trigger_path in self.trigger_paths: # loop over trigger paths
                    events_trig = None
                        
                    # select events for current trigger
                    if trigger_path == "all_available_triggers":
                        print("all_available_triggers")
                        events_trig = events_pure
                    else:
                        print(trigger_path)
                        trig_br = getattr(events_pure,trigger_path.split('_')[0])
                        trig_path = '_'.join(trigger_path.split('_')[1:])
                        events_trig = events_pure[getattr(trig_br,trig_path)] # select events passing trigger 
    
                    new_trigger_path = f"pure_{pure_wrt_to}_{trigger_path}"
                    new_trigger_paths += [new_trigger_path]
    
    
                    # save cutflow information
                    cutflow[new_trigger_path] = dak.num(events_trig.event, axis=0)
    
                    # run over all objects specified in the configuration file
                    hist_dict, branch_save_dict = run_the_megaloop(self, events_trig, hist_dict, branch_save_dict,dataset, new_trigger_path)
                    
                self.trigger_paths += new_trigger_paths 
            
                
                
        return_dict = {}
        return_dict['cutflow'] = [{i:cutflow[i]} for i in cutflow]#[{i:cutflow[i].compute()} for i in cutflow]#.compute()
        return_dict['hists'] = hist_dict
        return_dict['trigger'] = self.trigger_paths if len(self.trigger_paths)>0 else None
                
        return return_dict

    def postprocess(self, accumulator):
        return accumulator


###################################################################################################
# DEFINE MAIN FUNCTION

def main():
    """Main script execution."""
    client = Client("tls://localhost:8786")
    client.upload_file("./ScoutingNanoAODSchema.py");

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
