"""
Packs files of a biome (by default the ./assets/ folder) into <biomes dir>/<biome_id>.tar.gz and adds the biome
to <biomes dir>/server.json, the game server loads its biomes from there, see src/server/biomes.py.
Only the files the client loads go into the archive, see load_land() in src/client/main.py and
biomes.REQUIRED_FILES, other files of the folder are skipped.
The first biome of the list is where the clients come when they connect to the server.
Packing the same biome again replaces its archive and keeps its place in the list.

    PYTHONPATH=src venv/bin/python3 tools/pack_biome.py main --server-name alice --name "Main land"
    PYTHONPATH=src venv/bin/python3 tools/pack_biome.py desert --source_dir_path ./desert_assets/
    PYTHONPATH=src venv/bin/python3 tools/pack_biome.py --list
    PYTHONPATH=src venv/bin/python3 tools/pack_biome.py desert --remove
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import re
import sys
import gzip
import json
import tarfile
import argparse

#------------------------------------------------------------------------------

from server import biomes

#------------------------------------------------------------------------------

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#------------------------------------------------------------------------------


def read_config(biomes_dir_path):
    config_path = os.path.join(biomes_dir_path, biomes.CONFIG_FILE_NAME)
    if os.path.isfile(config_path):
        with open(config_path, 'rt') as f:
            return json.loads(f.read())
    return {'name': 'local', 'biomes': []}


def write_config(biomes_dir_path, config):
    config_path = os.path.join(biomes_dir_path, biomes.CONFIG_FILE_NAME)
    with open(config_path + '.part', 'wt') as f:
        f.write(json.dumps(config, indent=2))
    os.replace(config_path + '.part', config_path)


def skipped_files(source_dir_path):
    """
    Files of the source_dir_path folder which are not needed by the client and do not go into the archive.
    """
    skipped = []
    for folder_path, _, files in os.walk(source_dir_path):
        for file_name in files:
            rel_path = os.path.relpath(os.path.join(folder_path, file_name), source_dir_path)
            if rel_path not in biomes.REQUIRED_FILES:
                skipped.append(rel_path)
    return sorted(skipped)


def pack(source_dir_path, archive_path):
    """
    Writes the files of the biome (biomes.REQUIRED_FILES) into the .tar.gz archive, files are at the root of the archive.
    """
    names = sorted(biomes.REQUIRED_FILES)
    part_path = archive_path + '.part'
    # no time in the gzip header: the same files give the same archive and the clients do not download it again
    with open(part_path, 'wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w') as tar:
        for name in names:
            tar.add(os.path.join(source_dir_path, name), arcname=name, recursive=False)
    os.replace(part_path, archive_path)
    return names


def main():
    parser = argparse.ArgumentParser(description='pack files of a biome into an archive for the game server')
    parser.add_argument('biome_id', nargs='?', help='id of the biome: letters, digits, "-" and "_"')
    parser.add_argument('--source', dest='source_dir_path', metavar='SOURCE', default=os.path.join(ROOT_PATH, 'assets'), help='folder with the files of the biome, ./assets/ by default')
    parser.add_argument('--biomes-dir', dest='biomes_dir_path', metavar='BIOMES_DIR', default=os.path.join(ROOT_PATH, 'biomes'), help='where the server keeps its biomes, ./biomes/ by default')
    parser.add_argument('--name', default=None, help='name of the biome shown to the players')
    parser.add_argument('--server-name', default=None, help='name of the game server, kept in server.json')
    parser.add_argument('--first', action='store_true', help='move the biome to the start of the list, clients come there when they connect')
    parser.add_argument('--remove', action='store_true', help='remove the biome from the list and delete its archive')
    parser.add_argument('--list', action='store_true', help='print the server name and the biomes')
    args = parser.parse_args()
    os.makedirs(args.biomes_dir_path, exist_ok=True)
    config = read_config(args.biomes_dir_path)
    if args.server_name:
        config['name'] = args.server_name
    if args.list or not args.biome_id:
        if args.server_name:
            write_config(args.biomes_dir_path, config)
        print(f'server {config.get("name")!r}, biomes in {args.biomes_dir_path}:')
        for i, info in enumerate(config.get('biomes') or []):
            archive_path = os.path.join(args.biomes_dir_path, info['archive'])
            size = os.path.getsize(archive_path) / 1e6 if os.path.isfile(archive_path) else 0
            print(f'  {i + 1}. {info["id"]} "{info.get("name") or info["id"]}" {info["archive"]} {size:.1f} MB{"  <- clients come here" if i == 0 else ""}')
        return 0
    if not re.match(r'^[A-Za-z0-9_-]+$', args.biome_id):
        print(f'wrong biome id {args.biome_id!r}, only letters, digits, "-" and "_" are allowed')
        return 1
    known = config.setdefault('biomes', [])
    existing = [info for info in known if info['id'] == args.biome_id]
    archive_name = f'{args.biome_id}.tar.gz'
    archive_path = os.path.join(args.biomes_dir_path, archive_name)
    if args.remove:
        if not existing:
            print(f'biome {args.biome_id} not found')
            return 1
        known.remove(existing[0])
        write_config(args.biomes_dir_path, config)
        if os.path.isfile(archive_path):
            os.remove(archive_path)
        print(f'removed biome {args.biome_id}')
        return 0
    if not os.path.isdir(args.source_dir_path):
        print(f'folder {args.source_dir_path} not found')
        return 1
    missing = [f for f in biomes.REQUIRED_FILES if not os.path.isfile(os.path.join(args.source_dir_path, f))]
    if missing:
        print(f'{args.source_dir_path} has no {missing}, those files are needed by the server and the client')
        return 1
    names = pack(args.source_dir_path, archive_path)
    skipped = skipped_files(args.source_dir_path)
    if skipped:
        print(f'skipped {len(skipped)} files not needed by the client: {", ".join(skipped)}')
    # read it back the same way the server does
    archive = biomes.BiomeArchive(args.biome_id, args.name, archive_path)
    if existing:
        info = existing[0]
        if args.name:
            info['name'] = args.name
        info['archive'] = archive_name
    else:
        info = {'id': args.biome_id, 'name': args.name or args.biome_id, 'archive': archive_name}
        known.append(info)
    if args.first:
        known.remove(info)
        known.insert(0, info)
    write_config(args.biomes_dir_path, config)
    print(f'packed {len(names)} files from {args.source_dir_path} into {archive_path}, {archive.size / 1e6:.1f} MB, sha256 {archive.sha256}')
    print(f'biome {args.biome_id} is number {known.index(info) + 1} of {len(known)} on server {config["name"]!r}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
