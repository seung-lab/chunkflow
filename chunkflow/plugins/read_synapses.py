
import json
import os 


def execute(args: str = None):
    file_name = args
    assert os.path.exists(file_name)
    with open(file_name) as file:
        synapses = json.load(file)

    return [synapses]
