#!/usr/bin/env python
import os
import json

filename = "en_train.json"
path = os.path.abspath(os.path.join("..", "data", "raw",  "ace2005", filename))

with open(path) as f:
    ace_train = json.load(f)



