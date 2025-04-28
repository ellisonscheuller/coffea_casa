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
from collections import defaultdict
import csv
import dask
from dask.distributed import Client
import dask_awkward as dak
import datetime
import hist
from itertools import product
import json
import os
import re
import time
import vector
vector.register_awkward()
import uproot
import zipfile

# coffea imports
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import coffea.processor as processor
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)

from ScoutingNanoAODSchema import ScoutingNanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

from utils import find_diobjects, load_config, load_dataset, get_required_observables, calculate_observables, save_histogram, create_hist_1d, create_hist_2d, fill_hist_1d, fill_hist_2d, clone_axis

import fsspec
fsspec.config.conf['xrootd'] = {'timeout': 600}


####################################################################################################
# HELPER FUNCTIONS FOR PROCESSOR

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

def run_the_megaloop(self,events_trig,hist_dict,branch_save_dict,dataset,trigger_path):

        # get dictionary of observables to compute
        required_observables = get_required_observables(self)
        print("required observables = ", required_observables)

        # compute observables
        observable_calculations = calculate_observables(self, required_observables, events_trig)

        # fill 1d histograms
        for histogram_group, histograms in self.config["histograms_1d"].items() if self.config["histograms_1d"] else []:
            print("Histogram group: ", histogram_group)
                
            if histogram_group == "per_event": # event level histograms
                for histogram in histograms if histograms else []:
                    print("Histogram type: ",histogram)
                    if "_score" in histogram:
                        fill_hist_1d(
                            hist_dict, 
                            histogram, 
                            dataset, 
                            observable_calculations["per_event"][histogram], 
                            trigger_path, 
                            histogram
                        )
                    else: 
                        for reconstruction_level in self.config["objects"] \
                            if self.config["objects"] else []:
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
                for reconstruction_level, object_types in self.config["objects"].items() \
                    if self.config["objects"] else []: 
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

        # fill 2d histograms
        for entry in self.config["histograms_2d"] if self.config["histograms_2d"] else []:
            x_cat, x_var = entry["x_category"], entry["x_var"]
            y_cat, y_var = entry["y_category"], entry["y_var"]
            print("2D Histogram: ", x_cat, "-", x_var, ", ", y_cat, "-", y_var)

            if (x_cat == "per_diobject_pair") and ((y_cat == "per_object_type") | (y_cat == "per_object")):
                raise NotImplementedError("Cannot create 2d histogram of mixed categories per_diobject_pair and per_object or per_object_type")
            if (y_cat == "per_diobject_pair") and ((x_cat == "per_object_type") | (x_cat == "per_object")):
                raise NotImplementedError("Cannot create 2d histogram of mixed categories per_diobject_pair and per_object or per_object_type")

            for reconstruction_level in self.config["objects"] if self.config["objects"] else []:
                if x_cat=="per_event": 
                    if "_score" in x_var:
                        x_obs = observable_calculations["per_event"][x_var]
                    else: 
                        x_obs = observable_calculations["per_event"][reconstruction_level][x_var]

                if y_cat=="per_event": 
                    if "_score" in y_var:
                        y_obs = observable_calculations["per_event"][y_var]
                    else: 
                        y_obs = observable_calculations["per_event"][reconstruction_level][y_var]

                # fill histogram if neither category is per object
                if (x_cat=="per_event") and (y_cat=="per_event"):
                    hist_name = reconstruction_level+"_"+x_var+"_"+reconstruction_level+"_"+y_var
                    fill_hist_2d(hist_dict, hist_name, dataset, x_obs, y_obs, trigger_path, x_var, y_var)
                    continue
            

                x_is_object = x_cat in ["per_object_type", "per_object"]
                y_is_object = y_cat in ["per_object_type", "per_object"]
                        
                if x_is_object or y_is_object:

                    # fill histograms between two different object types
                    if (x_cat=="per_object_type") and (y_cat=="per_object_type"):
                        
                        for object_type_1, object_type_2 in product(self.config["objects"][reconstruction_level], repeat=2):
                            # avoid redundant self-self 2d histogram
                            if (object_type_1==object_type_2) and (x_var==y_var): continue

                            # avoid 2d histograms between two flattened arrays of different dimensinos
                            if (object_type_1!=object_type_2) and ((x_var in ["pt", "eta", "phi"]) and (y_var in ["pt", "eta", "phi"])): continue
                                
                            x_obs = observable_calculations["per_object_type"][reconstruction_level][object_type_1][x_var]
                            y_obs = observable_calculations["per_object_type"][reconstruction_level][object_type_2][y_var]
                            hist_name = object_type_1+"_"+x_var+"_"+object_type_2+"_"+y_var
                            if x_var==y_var:
                                fill_hist_2d(hist_dict, hist_name, dataset, x_obs, y_obs, trigger_path, f"{x_var}_1", f"{y_var}_2")
                            else:
                                fill_hist_2d(hist_dict, hist_name, dataset, x_obs, y_obs, trigger_path, x_var, y_var)
                        continue

                    # other cases (histograms not crossing between object types)
                    for object_type in self.config["objects"][reconstruction_level]:

                        # handle per_object_type cases
                        if x_cat=="per_object_type":
                            x_obs = observable_calculations["per_object_type"][reconstruction_level][object_type][x_var]
                        if y_cat=="per_object_type":
                            y_obs = observable_calculations["per_object_type"][reconstruction_level][object_type][y_var]


                        # fill histogram if we already have all info
                        hist_name = ""
                        if (x_cat=="per_event") and (y_cat=="per_object_type"):
                            if (y_var=="mult") or (y_var=="ht"):
                                x_obs_mod = x_obs
                            else:
                                y_obs_jagged = dak.unflatten(y_obs, observable_calculations["per_object_type"][reconstruction_level][object_type]["mult"])
                                x_obs_broadcast = dak.broadcast_arrays(x_obs, y_obs_jagged)[0]
                                x_obs_mod = dak.flatten(x_obs_broadcast)
                            hist_name = reconstruction_level+"_"+x_var+"_"+object_type+"_"+y_var
                            fill_hist_2d(hist_dict, hist_name, dataset, x_obs_mod, y_obs, trigger_path, x_var, y_var)
                            
                        elif (x_cat=="per_object_type") and (y_cat=="per_event"):
                            if (x_var=="mult") or (x_var=="ht"):
                                y_obs_mod = y_obs
                            else:
                                x_obs_jagged = dak.unflatten(x_obs, observable_calculations["per_object_type"][reconstruction_level][object_type]["mult"])
                                y_obs_broadcast = dak.broadcast_arrays(y_obs, x_obs_jagged)[0]
                                y_obs_mod = dak.flatten(y_obs_broadcast)
                            hist_name = object_type+"_"+x_var+"_"+reconstruction_level+"_"+y_var
                            fill_hist_2d(hist_dict, hist_name, dataset, x_obs, y_obs_mod, trigger_path, x_var, y_var)
                            
                        if hist_name!="":
                            continue
                        

                        # handle per_object cases
                        if (x_cat=="per_object") or (y_cat=="per_object"):
                            for i in range(self.config["objects_max_i"][object_type]):
                                if x_cat=="per_object":
                                    x_obs = observable_calculations["per_object"][reconstruction_level][object_type][f"{x_var}_{i}"]
                                if y_cat=="per_object":
                                    y_obs = observable_calculations["per_object"][reconstruction_level][object_type][f"{y_var}_{i}"]

                                # fill histograms if we have all info
                                hist_name = ""
                                if x_cat=="per_object":
                                    if y_cat=="per_object_type":
                                        raise NotImplementedError("Cannot create 2d histogram of mixed categories per_object_type and per_object")
                                    if y_cat=="per_event":
                                        x_obs_jagged = dak.unflatten(x_obs, observable_calculations["per_object"][reconstruction_level][object_type][f"counts_{i}"])
                                        y_obs_broadcast = dak.broadcast_arrays(y_obs, x_obs_jagged)[0]
                                        y_obs_mod = dak.flatten(y_obs_broadcast)
                                        hist_name = object_type+"_"+str(i)+"_"+x_var+"_"+reconstruction_level+"_"+y_var
                                        fill_hist_2d(hist_dict, hist_name, dataset, x_obs, y_obs_mod, trigger_path, x_var, y_var)
                                        continue
                                    if y_cat=="per_object":
                                        hist_name = object_type+"_"+str(i)+"_"+x_var+"_"+object_type+"_"+str(i)+"_"+y_var
                                        fill_hist_2d(hist_dict, hist_name, dataset, x_obs, y_obs, trigger_path, x_var, y_var)
                                        continue
                                        
                                if y_cat=="per_object":
                                    if x_cat=="per_object_type":
                                        raise NotImplementedError("Cannot create 2d histogram of mixed categories per_object_type and per_object")
                                    y_obs_jagged = dak.unflatten(y_obs, observable_calculations["per_object"][reconstruction_level][object_type][f"counts_{i}"])
                                    x_obs_broadcast = dak.broadcast_arrays(x_obs, y_obs_jagged)[0]
                                    x_obs_mod = dak.flatten(x_obs_broadcast)
                                    hist_name = reconstruction_level+"_"+x_var+"_"+object_type+"_"+str(i)+"_"+y_var
                                    fill_hist_2d(hist_dict, hist_name, dataset, x_obs, y_obs, trigger_path, x_var, y_var)
                                if hist_name!="":
                                    continue

            # handle per_diobject_pair cases
            if (x_cat=="per_diobject_pair") or (y_cat=="per_diobject_pair"):

                if (((x_cat=="per_event") and not ("_score" in x_var)) 
                    or ((y_cat=="per_event") and ("_score" in y_var))):
                    raise NotImplementedError("Cannot create 2d histogram of mixed categories per_diobject_pair and per_event except for anomaly score")
                
                for reconstruction_level, pairings in self.config["diobject_pairings"].items():
                    for pairing in pairings if pairings else []:
                        print("Pairing:",pairing)
                        object_type_1 = pairing[0]
                        object_type_2 = pairing[1]

                        if x_cat=="per_diobject_pair":
                            x_obs = observable_calculations["per_diobject_pair"][reconstruction_level][f"{object_type_1}_{object_type_2}"][x_var]
                        if y_cat=="per_diobject_pair":
                            y_obs = observable_calculations["per_diobject_pair"][reconstruction_level][f"{object_type_1}_{object_type_2}"][y_var]

                        hist_name=""
                        if (x_cat=="per_diobject_pair") and (y_cat=="per_diobject_pair"):
                            hist_name = f"{object_type_1}_{object_type_2}_{x_var}_{object_type_1}_{object_type_2}_{y_var}"
                        if (x_cat=="per_diobject_pair") and ("_score" in y_var):
                            hist_name = f"{object_type_1}_{object_type_2}_{x_var}_{y_var}"
                        if ("_score" in x_var) and (y_cat=="per_diobject_pair"):
                            hist_name = f"{object_type_1}_{object_type_2}_{x_var}_{y_var}"
                        if hist_name!="":
                            fill_hist_2d(hist_dict, hist_name, dataset, x_obs, y_obs, trigger_path, x_var, y_var)
                            
        return hist_dict, branch_save_dict
       
def initialize_hist_dict(self,hist_dict):
    # Axis mapping 
    axis_map = {
        'axo_score': self.axo_score_axis,
        'cicada_score': self.cicada_score_axis,
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
        'mass': self.mass_axis,
        'deltaR': self.deltaR_axis
    }
    for histogram_group, histograms in self.config["histograms_1d"].items() \
        if self.config["histograms_1d"] else []:  # Process each histogram according to its group
        for histogram in histograms if histograms else []: # Variables to plot
            if "_score" in histogram: # Score doesn't depend on reco level
                hist_name = f"{histogram}"  
                hist_dict = create_hist_1d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[histogram], hist_name=hist_name) 
                continue
            for reconstruction_level in self.config["objects"] \
                if self.config["objects"] else []: # Loop over different object reconstruction levels
                if histogram_group == "per_event" and not ("_score" in histogram): # Per event ---
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
                for reconstruction_level in self.config["diobject_pairings"] \
                    if self.config["diobject_pairings"] else []:
                    for pairing in self.config["diobject_pairings"][reconstruction_level] \
                        if self.config["diobject_pairings"][reconstruction_level] else []:
                        obj_1, obj_2 = pairing[0],pairing[1]
                        hist_name = f"{obj_1}_{obj_2}_{histogram}"
                        hist_dict = create_hist_1d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[histogram], hist_name=hist_name)

    for entry in self.config["histograms_2d"] \
        if self.config.get("histograms_2d") else []:
        x_cat, x_var = entry["x_category"], entry["x_var"]
        y_cat, y_var = entry["y_category"], entry["y_var"]

        if (x_cat == "per_diobject_pair") and ((y_cat == "per_object_type") | (y_cat == "per_object")):
            raise NotImplementedError("Cannot create 2d histogram of mixed categories per_diobject_pair and per_object or per_object_type")
        if (y_cat == "per_diobject_pair") and ((x_cat == "per_object_type") | (x_cat == "per_object")):
            raise NotImplementedError("Cannot create 2d histogram of mixed categories per_diobject_pair and per_object or per_object_type")

        for reconstruction_level in self.config["objects"]:
            
            if (x_cat=="per_event") and (y_cat=="per_event"):
                hist_name = reconstruction_level+"_"+x_var+"_"+reconstruction_level+"_"+y_var
                create_hist_2d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[x_var], axis_map[y_var], hist_name)
                continue
            
            x_is_object = x_cat in ["per_object_type", "per_object"]
            y_is_object = y_cat in ["per_object_type", "per_object"]
                        
            if x_is_object or y_is_object:

                if (x_cat=="per_object_type") and (y_cat=="per_object_type"):
                    for object_type_1, object_type_2 in product(self.config["objects"][reconstruction_level], repeat=2):
                        # avoid creating redundant self-self 2d histogram
                        if (object_type_1==object_type_2) and (x_var==y_var): continue

                        # can't create 2d histograms between two flattened arrays of different dimensinos
                        if (object_type_1!=object_type_2) and ((x_var in ["pt", "eta", "phi"]) and (y_var in ["pt", "eta", "phi"])): continue
                            
                        hist_name = object_type_1+"_"+x_var+"_"+object_type_2+"_"+y_var
                        if x_var==y_var:
                            create_hist_2d(hist_dict, self.dataset_axis, self.trigger_axis, clone_axis(axis_map[x_var], f"{x_var}_1"), 
                                           clone_axis(axis_map[y_var], f"{y_var}_2"), hist_name)
                        else:
                            create_hist_2d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[x_var], axis_map[y_var], hist_name)
                
                for object_type in self.config["objects"][reconstruction_level]:

                    hist_name = ""
                    if (x_cat=="per_event") and (y_cat=="per_object_type"):
                        hist_name = reconstruction_level+"_"+x_var+"_"+object_type+"_"+y_var
                    elif (x_cat=="per_object_type") and (y_cat=="per_event"):
                        hist_name = object_type+"_"+x_var+"_"+reconstruction_level+"_"+y_var
                    if hist_name!="":
                        create_hist_2d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[x_var], axis_map[y_var], hist_name)
                        continue
                        

                    if (x_cat=="per_object") or (y_cat=="per_object"):
                        for i in range(self.config["objects_max_i"][object_type]):
                                
                            hist_name = ""
                            if x_cat=="per_object":
                                if y_cat=="per_event":
                                    hist_name = object_type+"_"+str(i)+"_"+x_var+"_"+reconstruction_level+"_"+y_var
                                if y_cat=="per_object_type":
                                    raise NotImplementedError("Cannot create 2d histogram of mixed categories per_object_type and per_object")
                                if y_cat=="per_object":
                                    hist_name = object_type+"_"+str(i)+"_"+x_var+"_"+object_type+"_"+str(i)+"_"+y_var
                            if y_cat=="per_object":
                                if x_cat=="per_event":
                                    hist_name = reconstruction_level+"_"+x_var+"_"+object_type+"_"+str(i)+"_"+y_var
                                if x_cat=="per_object_type":
                                    raise NotImplementedError("Cannot create 2d histogram of mixed categories per_object_type and per_object")
                            if hist_name!="":
                                create_hist_2d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[x_var], axis_map[y_var], hist_name)
                                continue

        if (x_cat=="per_diobject_pair") or (y_cat=="per_diobject_pair"):

            if (((x_cat=="per_event") and not ("_score" in x_var)) 
                or ((y_cat=="per_event") and not ("_score" in y_var))):
                raise NotImplementedError("Cannot create 2d histogram of mixed categories per_diobject_pair and per_event except for anomaly score")
                
            for reconstruction_level, pairings in self.config["diobject_pairings"].items() \
                if self.config["diobject_pairings"] else []:
                for pairing in pairings if pairings else []:
                    object_type_1 = pairing[0]
                    object_type_2 = pairing[1]

                    hist_name=""
                    if (x_cat=="per_diobject_pair") and (y_cat=="per_diobject_pair"):
                        hist_name = f"{object_type_1}_{object_type_2}_{x_var}_{object_type_1}_{object_type_2}_{y_var}"
                    if (x_cat=="per_diobject_pair") and ("_score" in y_var):
                        hist_name = f"{object_type_1}_{object_type_2}_{x_var}_{y_var}"
                    if ("_score" in x_var) and (y_cat=="per_diobject_pair"):
                        hist_name = f"{object_type_1}_{object_type_2}_{x_var}_{y_var}"
                    if hist_name!="":
                        create_hist_2d(hist_dict, self.dataset_axis, self.trigger_axis, axis_map[x_var], axis_map[y_var], hist_name)
                        
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

        if config["use_emulated_score"]:
            # Unzip only once per worker (guard with a flag)
            if not os.path.exists("config"):
                with zipfile.ZipFile("config.zip", "r") as zip_ref:
                    zip_ref.extractall("config")

            # Load the CSV into memory
            self.axo_thresholds = {}
            with open(f"config/axo_thresholds_{axo_version}.csv", newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.axo_thresholds[row["L1 Seed"]] = float(row["Threshold"])

            self.cicada_thresholds = {}
            with open(f"config/cicada_thresholds.csv", newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.cicada_thresholds[row["L1 Seed"]] = float(row["Threshold"])

        
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
        if self.axo_version=="v3":
            self.axo_score_axis = hist.axis.Regular(
                1000, 0, 3000, name="axo_score", label='AXOL1TL Anomaly Score'
            )
        else:
            self.axo_score_axis = hist.axis.Regular(
                600, 0, 600, name="axo_score", label='AXOL1TL Anomaly Score'
            )
        self.cicada_score_axis = hist.axis.Regular(
            256, 0, 256, name="cicada_score", label='CICADA Anomaly Score'
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
            100, 0, 1000, name="met", label=r"$p^{miss}_{T} [GeV]$"
        )
        self.ht_axis = hist.axis.Regular(
            200, 0, 4000, name="ht", label=r"$H_{T}$ [GeV]"
        )
        self.mass_axis = hist.axis.Regular(
            300, 0, 3000, name="mass", label=r"$m_{obj_{1},obj_{2}}$ [GeV]"
        )
        self.deltaR_axis = hist.axis.Regular(
            300, 0, 6.5, name="deltaR", label=r"$\Delta R$ between $obj_1$ and $obj_2$"
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
        for object_type in self.config["objects"] if self.config["objects"] else []:
            for my_object in self.config["objects"][object_type] if self.config["objects"][object_type] else []:
                assert my_object in events.fields, f"Error: {my_object} not in available fields: {events.fields}" 
                
        # Check that the triggers you have requested are available in the data
        print("Trigger paths:",self.trigger_paths)
        for trigger in self.trigger_paths:
            print("Trigger",trigger)
            if (trigger[0:3] == "DST"):
                if (("AXO" in trigger) or ("CICADA" in trigger)):
                    if (not self.config["use_emulated_score"]):
                        assert trigger[4:] in events.DST.fields, f"Error: {trigger[4:]} not in available DST paths: {events.DST.fields}"
                else:
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
                elif ("AXO" in trigger_path) and self.config["use_emulated_score"]:
                    print(trigger_path + " (emulated)")
                    axo_trigger_name = (re.search(r"(AXO\w+)", trigger_path)[0]).replace("_", "")
                    if self.axo_version not in ["v3", "v4"]:
                        raise NotImplementedError(f"axo version {self.axo_version} not implemented")
                    score_attr = f"{'v3' if self.axo_version == 'v3' else 'v4'}_AXOScore" if self.config["is_l1nano"] else f"score_{self.axo_version}"
                    events_trig = events[getattr(events.axol1tl, score_attr) > self.axo_thresholds[axo_trigger_name]]
                elif ("CICADA" in trigger_path) and self.config["use_emulated_score"]:
                    print(trigger_path + " (emulated)")
                    if not self.config["is_l1nano"]:
                        raise NotImplementedError(f"CICADA score is not implemented in Scouting NanoAOD")
                    cicada_trigger_name = (re.search(r"(CICADA\w+)", trigger_path)[0]).replace("_", "")
                    events_trig = events[getattr(events.CICADA2024, "CICADAScore") > self.cicada_thresholds[cicada_trigger_name]]
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
                    dak_zip.persist().to_parquet(self.config["branch_writing_path"] + f"axo_branches_{trigger_path}")

        
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

                if self.config["save_branches"]: 
                    dak_zip = dak.zip(branch_save_dict)
                    dak_zip.persist().to_parquet(self.config["branch_writing_path"] + f"/axo_branches_{new_trigger_path}")
                
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
    client.upload_file("./utils.py");

    config = load_config()
    
    if config["use_emulated_score"]:
        client.upload_file("config.zip");

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
