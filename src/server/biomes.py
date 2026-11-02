"""
Biomes of the game server.

Every biome is one world: its land, buildings and plants are the files which tools/pack_biome.py packed into
<biomes dir>/<biome_id>.tar.gz. The server reads those files right from the archive, and the archive itself is sent
to the clients over the web socket before they enter the biome.

<biomes dir>/server.json keeps the name of the server and the list of its biomes, the first one is where the clients
come when they connect:

    {"name": "alice", "biomes": [{"id": "main", "name": "Main land", "archive": "main.tar.gz"}]}
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import io
import json
import base64
import hashlib
import tarfile

#------------------------------------------------------------------------------

from lib import models

#------------------------------------------------------------------------------

CONFIG_FILE_NAME = 'server.json'

# files of the biome archive, the client loads them in load_land() of src/client/main.py and the server reads some of them,
# tools/pack_biome.py packs only these files
REQUIRED_FILES = ('heightmap.png', 'encoded.png', 'tiles.json', 'plants.json', 'buildings.json')

#------------------------------------------------------------------------------


class BiomeArchive(object):

    def __init__(self, biome_id, name, archive_path):
        self.id = biome_id
        self.name = name or biome_id
        self.archive_path = archive_path
        with open(archive_path, 'rb') as f:
            self.data = f.read()
        self.sha256 = hashlib.sha256(self.data).hexdigest()
        self.size = len(self.data)
        self.files = {}
        with tarfile.open(fileobj=io.BytesIO(self.data), mode='r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    self.files[os.path.normpath(member.name)] = tar.extractfile(member).read()
        missing = [f for f in REQUIRED_FILES if f not in self.files]
        if missing:
            raise ValueError(f'biome {biome_id} archive {archive_path} has no {missing}')

    def read(self, file_name):
        return self.files[file_name]

    def read_json(self, file_name):
        return json.loads(self.files[file_name].decode('utf-8'))

    def info(self):
        return models.Biome(id=self.id, name=self.name, sha256=self.sha256, size=self.size)

    def chunks(self):
        """
        Pieces of the archive to be sent to the client as "biome_data" messages, base64 encoded.
        """
        for i in range(0, self.size, models.BIOME_CHUNK_SIZE):
            yield base64.b64encode(self.data[i:i + models.BIOME_CHUNK_SIZE]).decode('ascii')

    def chunks_count(self):
        return (self.size + models.BIOME_CHUNK_SIZE - 1) // models.BIOME_CHUNK_SIZE

#------------------------------------------------------------------------------


def load(biomes_dir):
    """
    Returns (server name, [BiomeArchive]) from <biomes_dir>/server.json, biomes are in the order of the list.
    """
    config_path = os.path.join(biomes_dir, CONFIG_FILE_NAME)
    if not os.path.isfile(config_path):
        raise Exception(f'{config_path} not found, create biomes with tools/pack_biome.py')
    with open(config_path, 'rt') as f:
        config = json.loads(f.read())
    archives = []
    for info in config.get('biomes') or []:
        archives.append(BiomeArchive(info['id'], info.get('name'), os.path.join(biomes_dir, info['archive'])))
    if not archives:
        raise Exception(f'no biomes in {config_path}, add them with tools/pack_biome.py')
    return config.get('name') or 'local', archives
