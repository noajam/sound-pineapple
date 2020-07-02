import json
import pathlib

def read_json(configpath):
    with open(pathlib.Path(configpath)) as jsonFile:
       jsonDict = json.load(jsonFile)
       return jsonDict
   
