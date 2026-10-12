# Test fixtures for .semgrep/ml-supply-chain.yml (run: semgrep --test .semgrep)
import pickle

import joblib
import requests
import skops.io as sio
import torch
import yaml

# ruleid: unsafe-model-deserialization
model = pickle.load(open("model.pkl", "rb"))
# ruleid: unsafe-model-deserialization
model = joblib.load("model.joblib")
# ruleid: torch-load-without-weights-only
weights = torch.load("model.pt")
# ok: torch-load-without-weights-only
weights = torch.load("model.pt", weights_only=True)
# ruleid: skops-blanket-trust
model = sio.load("model.skops", trusted=sio.get_untrusted_types(file="model.skops"))
# ok: skops-blanket-trust
model = sio.load("model.skops", trusted=["sklearn.tree._tree.Tree"])
# ruleid: yaml-unsafe-load
cfg = yaml.load(open("params.yaml"), Loader=yaml.Loader)
# ok: yaml-unsafe-load
cfg = yaml.safe_load(open("params.yaml"))
# ruleid: tls-verification-disabled
requests.get("https://example.com/data.csv", verify=False)
