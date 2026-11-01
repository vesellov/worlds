"""
Packs files of a biome (by default the ./assets/ folder) into <biomes dir>/<biome_id>.tar.gz and adds the biome
to <biomes dir>/server.json, the game server loads its biomes from there, see src/server/biomes.py.
The first biome of the list is where the clients come when they connect to the server.
Packing the same biome again replaces its archive and keeps its place in the list.

    PYTHONPATH=src venv/bin/python3 tools/pack_biome.py main --server-name alice --name "Main land"
    PYTHONPATH=src venv/bin/python3 tools/pack_biome.py desert --source ./desert_assets/
    PYTHONPATH=src venv/bin/python3 tools/pack_biome.py --list
    PYTHONPATH=src venv/bin/python3 tools/pack_biome.py desert --remove
"""

import os
import re
import sys
import gzip
import json
import tarfile
import argparse

from server import biomes

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_config(biomes_dir):
    config_path = os.path.join(biomes_dir, biomes.CONFIG_FILE_NAME)
    if os.path.isfile(config_path):
        with open(config_path, 'rt') as f:
            return json.loads(f.read())
    return {'name': 'local', 'biomes': []}


def write_config(biomes_dir, config):
    config_path = os.path.join(biomes_dir, biomes.CONFIG_FILE_NAME)
    with open(config_path + '.part', 'wt') as f:
        f.write(json.dumps(config, indent=2))
    os.replace(config_path + '.part', config_path)


def pack(source_dir, archive_path):
    """
    Writes all files of the source folder into the .tar.gz archive, files are at the root of the archive.
    """
    names = []
    for folder, _, files in os.walk(source_dir):
        for file_name in files:
            names.append(os.path.relpath(os.path.join(folder, file_name), source_dir))
    names.sort()
    part_path = archive_path + '.part'
    # no time in the gzip header: the same files give the same archive and the clients do not download it again
    with open(part_path, 'wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w') as tar:
        for name in names:
            tar.add(os.path.join(source_dir, name), arcname=name, recursive=False)
    os.replace(part_path, archive_path)
    return names


def main():
    parser = argparse.ArgumentParser(description='pack files of a biome into an archive for the game server')
    parser.add_argument('biome_id', nargs='?', help='id of the biome: letters, digits, "-" and "_"')
    parser.add_argument('--source', default=os.path.join(ROOT_PATH, 'assets'), help='folder with the files of the biome, ./assets/ by default')
    parser.add_argument('--biomes-dir', default=os.path.join(ROOT_PATH, 'biomes'), help='where the server keeps its biomes, ./biomes/ by default')
    parser.add_argument('--name', default=None, help='name of the biome shown to the players')
    parser.add_argument('--server-name', default=None, help='name of the game server, kept in server.json')
    parser.add_argument('--first', action='store_true', help='move the biome to the start of the list, clients come there when they connect')
    parser.add_argument('--remove', action='store_true', help='remove the biome from the list and delete its archive')
    parser.add_argument('--list', action='store_true', help='print the server name and the biomes')
    args = parser.parse_args()
    os.makedirs(args.biomes_dir, exist_ok=True)
    config = read_config(args.biomes_dir)
    if args.server_name:
        config['name'] = args.server_name
    if args.list or not args.biome_id:
        if args.server_name:
            write_config(args.biomes_dir, config)
        print(f'server {config.get("name")!r}, biomes in {args.biomes_dir}:')
        for i, info in enumerate(config.get('biomes') or []):
            archive_path = os.path.join(args.biomes_dir, info['archive'])
            size = os.path.getsize(archive_path) / 1e6 if os.path.isfile(archive_path) else 0
            print(f'  {i + 1}. {info["id"]} "{info.get("name") or info["id"]}" {info["archive"]} {size:.1f} MB{"  <- clients come here" if i == 0 else ""}')
        return 0
    if not re.match(r'^[A-Za-z0-9_-]+$', args.biome_id):
        print(f'wrong biome id {args.biome_id!r}, only letters, digits, "-" and "_" are allowed')
        return 1
    known = config.setdefault('biomes', [])
    existing = [info for info in known if info['id'] == args.biome_id]
    archive_name = f'{args.biome_id}.tar.gz'
    archive_path = os.path.join(args.biomes_dir, archive_name)
    if args.remove:
        if not existing:
            print(f'biome {args.biome_id} not found')
            return 1
        known.remove(existing[0])
        write_config(args.biomes_dir, config)
        if os.path.isfile(archive_path):
            os.remove(archive_path)
        print(f'removed biome {args.biome_id}')
        return 0
    if not os.path.isdir(args.source):
        print(f'folder {args.source} not found')
        return 1
    missing = [f for f in biomes.REQUIRED_FILES if not os.path.isfile(os.path.join(args.source, f))]
    if missing:
        print(f'{args.source} has no {missing}, those files are needed by the server and the client')
        return 1
    names = pack(args.source, archive_path)
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
    write_config(args.biomes_dir, config)
    print(f'packed {len(names)} files from {args.source} into {archive_path}, {archive.size / 1e6:.1f} MB, sha256 {archive.sha256}')
    print(f'biome {args.biome_id} is number {known.index(info) + 1} of {len(known)} on server {config["name"]!r}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
