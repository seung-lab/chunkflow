import types
from importlib.machinery import SourceFileLoader


def load_source(fname: str):
    """ Imports a module from source.

    Parameters
    -----------
    fname:
        file path of the python code.
    """
    loader = SourceFileLoader("Model", fname)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod
