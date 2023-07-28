from ast import literal_eval

import numpy as np

# from chunkflow.chunk import Chunk
# from chunkflow.volume import PrecomputedVolume
# from chunkflow.flow.plugin import str_to_dict 



def simplest_type(s: str):
    try:
        return literal_eval(s)
    except:
        return s

def str_to_dict(string: str):
    keywords = {}
    for item in string.split(';'):
        assert '=' in item
        item = item.split('=')
        keywords[item[0]] = simplest_type(item[1])
    return keywords

