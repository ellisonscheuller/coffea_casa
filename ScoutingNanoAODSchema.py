from coffea.nanoevents.schemas.nanoaod import NanoAODSchema

class ScoutingNanoAODSchema(NanoAODSchema):
    """ScoutingNano schema builder

    ScoutingNano is a NanoAOD format that includes Scouting objects
    """

    mixins = {
        **NanoAODSchema.mixins,

        "ScoutingJet": "Jet",
        "ScoutingFatJet": "Jet",
        "ScoutingMuonNoVtxDisplacedVertex": "Vertex",
        "ScoutingMuonVtxDisplacedVertex": "Vertex",
        "ScoutingPrimaryVertex":"Vertex",
        "ScoutingElectron": "Electron",
        "ScoutingPhoton": "Photon", 
        "ScoutingMuonNoVtx": "Muon",
        "ScoutingMuonVtx": "Muon"

    }
    all_cross_references = {
        **NanoAODSchema.all_cross_references
    }