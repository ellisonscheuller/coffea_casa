from coffea.nanoevents.methods import vector
from coffea.util import save
import dask_awkward as dak
import datetime
import hist.dask as hda
import json
import time
import yaml

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
    mask = dak.ones_like(getattr(objects, reference_branch), dtype=bool)

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