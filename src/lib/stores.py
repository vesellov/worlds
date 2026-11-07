"""
Data of the stores in the files of the catalog.

Models and textures of the game are published in a few stores, every store is one version of the original game,
see lib.models.DEFAULT_STORE_ID. The same model can be different in two stores, so every file of the catalog
which describes the models keeps the data of every store apart: {"<store id>": {name: record}}, and every record
there has "store_id" of its store. figures.json, animations.json, armors.json, weapons.json and materials.json
are built by tools/build_registry.py, see "make ei_registry".

Every template of a unit says which store it is of (lib.models.Template.store_id), so the server and the client
take the animations, the parts, the armors and the weapons of the unit from the data of that store.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import json

#------------------------------------------------------------------------------

from lib import models

#------------------------------------------------------------------------------


def read(file_path):
    """
    Data of every store of a file of the catalog: store id -> data.
    """
    with open(file_path, 'rt') as f:
        stores_data = json.loads(f.read())
    return dict((int(store_id), data) for store_id, data in stores_data.items())


def of(stores_data, store_id=None):
    """
    Data of that store, the default store when store_id is None, empty when the catalog has no data of that store.
    """
    if store_id is None:
        store_id = models.DEFAULT_STORE_ID
    return stores_data.get(int(store_id), {})
