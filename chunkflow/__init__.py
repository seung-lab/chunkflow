from os import path
script_path = path.dirname(path.abspath( __file__ ))
version_file = path.join(script_path, '../VERSION.txt')

with open(version_file, "r") as f:
    __version__ = f.read().strip()
