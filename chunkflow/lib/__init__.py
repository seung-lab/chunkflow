import importlib, types


def load_source(fname: str, module_name: str = "Model"):
    """ Imports a module from source.

    Parameters
    -----------
    fname:
        file path of the python code.
    module_name:
        name of the module. do we really need this parameter?
    """
    loader = importlib.machinery.SourceFileLoader(module_name, fname)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod
