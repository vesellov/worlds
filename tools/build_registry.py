"""
Builds the registry of the units of the original game "Evil Islands" for the catalog/ folder:
races, monster prototypes, NPCs, items, spells, skills, animations of the figures, sounds,
textures of the weapons and armors for every figure, units placed on the maps of the game,
and figures_registry.json which joins all of that for every figure.
It also builds the files of the catalog which the server and the client read: figures.json (the tree of the parts
of every model), animations.json (the .adb of every figure), armors.json, weapons.json and materials.json.

The data of a few versions of the game is kept together, every version is a store, see lib.models.DEFAULT_STORE_ID:
store 1 is "Evil Islands" and store 2 is "Lost in Astral". Every file of the catalog is {"<store id>": data}
and every record there knows its store: "store_id": 1. Models and textures of a store are downloaded from it,
the templates of the units say which store they are of, see lib.models.Template.

The files are read right from the game folder, the archives are not unpacked to the disk:
database.res and databaselmp.res (the database of the multiplayer maps), figures.res, redress.res, sfx.res
and the maps/*.mob files. Run inside repack/EIrepack, see "make ei_registry" in the Makefile.

A mod keeps only the files it changes, so the folder of the game is given first and the folder of the mod after it,
for example "Evil Islands: Rebirth" and its Starter/Mods/LostInAstral/ ("Lost in Astral"):
the archives of all the folders are joined, a later folder replaces the files of the same name: the database
of "Lost in Astral" has its own tables and only the .adb of its new figure, the rest come from the game.
The maps are taken from the last folder which has them.

usage: build_registry.py --store <store id> <game data folder> [<mod folder> ...] [--store ...] --dest <output folder>
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import io
import os
import re
import json
import struct
import argparse
import tempfile
import contextlib
import collections

#------------------------------------------------------------------------------

import res
import mob
import extract_db_data

#------------------------------------------------------------------------------

# databases of the game: the main one and the one of the multiplayer maps (*-lmp.mob)
DATABASES = (
    ('database.res', 'main'),
    ('databaselmp.res', 'lmp'),
)

# "locomotion" of the races
LOCOMOTION_NAMES = {
    1: 'ground',
    2: 'air',
    3: 'static',
}

# fields of the database records which hold lists, the database gives "" for an empty list and a string for one item
LIST_FIELDS = (
    'spells', 'wears', 'items', 'weapons', 'quest_items', 'textures', 'steps_path',
    'required', 'optional', 'buildin_mods', 'special_mods', 'zones', 'perks',
)

# fields which hold names of the materials and types, those are in lower case in the catalog
LOWER_FIELDS = ('material_type', 'type', 'skin_type', 'base_attributes')

# misspelled names of the fields of the database
FIELD_NAMES = {
    'expirience': 'experience',
    'absorbtion': 'absorption',
    'absortion_1': 'absorption_1',
    'absortion_2': 'absorption_2',
    'real_wepon_type_id': 'real_weapon_type_id',
}

# codes of the weapons and the armors in the names of their textures in redress.res,
# <figure><code>_<texture type 1>.<material code>.<texture type 2>.mmp, see lib.dress
REDRESS_CODES = {
    'ax': 'axe',
    'bw': 'bow',
    'cb': 'crossbow',
    'dg': 'dagger',
    'hm': 'hammer',
    'sp': 'spear',
    'sw': 'sword',
    'bt': 'boots',
    'gl': 'gloves',
    'hl': 'helm',
    'lg': 'leggings',
    'pl': 'plate',
    'pt': 'pants',
    'sh': 'shirt',
}

RES_MAGIC = b'\x3C\xE2\x9C\x01'

# archives with the models: menus.res has the figures of the menus of the game
FIGURES_ARCHIVES = ('figures.res', 'menus.res')

# files of the catalog: file name -> how deep the records are, every record gets "store_id" of its store,
# 0 are the files without records: lists of the parts of the models, of the sounds and so on
CATALOG_FILES = (
    ('races.json', 1),
    ('monsters.json', 1),
    ('npcs.json', 1),
    ('items.json', 2),
    ('spells.json', 2),
    ('skills.json', 2),
    ('hit_locations.json', 1),
    ('figures_animations.json', 1),
    ('figures_redress.json', 0),
    ('sounds.json', 0),
    ('units_samples.json', 0),
    ('figures_registry.json', 1),
    ('figures.json', 0),
    ('animations.json', 1),
    ('armors.json', 1),
    ('weapons.json', 1),
    ('materials.json', 1),
)

#------------------------------------------------------------------------------


def write_json(dest_dir_path, file_name, data):
    with open(os.path.join(dest_dir_path, file_name), 'wt') as f:
        f.write(json.dumps(data, indent=2, sort_keys=True))


def mark_store(data, store_id, depth):
    """
    Adds "store_id" to every record of the data, the records are that deep in it.
    """
    if depth == 0:
        return data
    for value in data.values():
        if depth == 1:
            value['store_id'] = store_id
        else:
            mark_store(value, store_id, depth - 1)
    return data


def read_res(file_path):
    """
    Files of a .res archive: name -> bytes.
    """
    with open(file_path, 'rb') as f:
        return read_res_data(f)


def read_res_data(f):
    result = {}
    for name, offset, size in res.read_filetree(f):
        f.seek(offset)
        result[name] = f.read(size)
    return result


def is_res(data):
    return data[:4] == RES_MAGIC


def last_path(game_dir_paths, *names):
    """
    The path to the file or folder in the last of the folders of the game which has it, None if none of them has.
    """
    for game_dir_path in reversed(game_dir_paths):
        path = os.path.join(game_dir_path, *names)
        if os.path.exists(path):
            return path
    return None


def res_paths(game_dir_paths, res_name):
    """
    Paths to the archive of this name in all the folders of the game which have it, in the order of the folders.
    """
    paths = [os.path.join(game_dir_path, 'res', res_name) for game_dir_path in game_dir_paths]
    return [path for path in paths if os.path.isfile(path)]


def read_joined_res(game_dir_paths, res_name):
    """
    Files of the archive of all the folders of the game: lower case name -> bytes, a later folder replaces the files.
    """
    result = {}
    for res_path in res_paths(game_dir_paths, res_name):
        for name, data in read_res(res_path).items():
            result[name.lower()] = data
    return result


def joined_res_names(game_dir_paths, res_name):
    """
    Lower case names of the files of the archive of all the folders of the game, without reading the files.
    """
    result = set()
    for res_path in res_paths(game_dir_paths, res_name):
        with open(res_path, 'rb') as f:
            result.update(name.lower() for name, _, _ in res.read_filetree(f))
    return sorted(result)

#--- DATABASE ---


def field_name(name):
    # "typeID" -> "type_id", "SFX path" -> "sfx_path"
    name = re.sub(r'([a-z])ID', r'\1_ID', name).lower().replace(' ', '_')
    return FIELD_NAMES.get(name, name)


def clean_record(record):
    """
    The record of a database table with the names of the fields in lower case, without the unknown fields
    and the fields which hold the raw bytes.
    """
    result = {}
    for key, value in record.items():
        key = field_name(key)
        if isinstance(value, str) and value.startswith("b'"):
            continue
        if key.startswith('unknown'):
            # one of them holds the folder of the sounds of a hero, the rest are numbers nobody knows
            if not (isinstance(value, str) and value):
                continue
        if key in LIST_FIELDS:
            if value == '':
                value = []
            elif not isinstance(value, list):
                value = [value, ]
        if key in LOWER_FIELDS and isinstance(value, str):
            value = value.lower()
        result[key] = value
    return result


def read_databases(game_dir_paths):
    """
    Tables of the databases: (database name, table name) -> list of records, and the animations of the figures
    of the .adb files of the main database: figure name -> lib.extract_db_data.split_adb_data().
    """
    tables = {}
    animations = {}
    with tempfile.TemporaryDirectory() as temp_dir_path:
        for res_name, db_name in DATABASES:
            for file_name, data in read_joined_res(game_dir_paths, res_name).items():
                base_name, _, ext = file_name.rpartition('.')
                if ext not in ('idb', 'pdb', 'sdb', 'udb', 'adb'):
                    continue
                file_path = os.path.join(temp_dir_path, f'{db_name}_{file_name}')
                with open(file_path, 'wb') as f:
                    f.write(data)
                if ext == 'adb':
                    if db_name != 'main':
                        continue
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            info = extract_db_data.read_adb_data(file_path)
                    except struct.error:
                        # inmm1.adb of the game is cut short
                        print(f'skipped broken {file_name}')
                        continue
                    if info:
                        animations[base_name] = extract_db_data.split_adb_data(info)
                    continue
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        data = extract_db_data.read_data(file_path)
                except struct.error:
                    # spells.sdb of the russian version of 2000 has an older format
                    print(f'skipped broken {db_name} {file_name}')
                    continue
                for table_name, records in extract_db_data.data_tables(data).items():
                    tables[(db_name, f'{base_name}_{table_name}')] = records
    return tables, animations


def merged_table(tables, table_name, key='name'):
    """
    Records of the table of the main database and the records of the multiplayer database which the main one does not have,
    every record knows its database: "database" is "main" or "lmp". Records are keyed by their names.
    """
    result = {}
    for db_name in ('main', 'lmp'):
        for record in tables.get((db_name, table_name), []):
            record = clean_record(record)
            name = str(record.pop(key))
            if name in result:
                continue
            record['database'] = db_name
            result[name] = record
    return result


def build_races(tables):
    races = {}
    for name, record in merged_table(tables, 'units_race_models').items():
        race = {
            'figure': (record.pop('mask_name') or '').lower(),
            'database': record.pop('database'),
            'locomotion': LOCOMOTION_NAMES.get(record.get('locomotion'), record.get('locomotion')),
            'body': {},
            'attack': {},
            'defence': {},
            'animation_speeds': {},
        }
        record.pop('locomotion', None)
        for part in ('head', 'torso', 'right_arm', 'left_arm', 'right_leg', 'left_leg'):
            race['body'][part] = {
                'type': record.pop(f'{part}_type', None),
                'weight': record.pop(f'{part}_weight', None),
                'hp': record.pop(f'{part}_hp', None),
            }
        for key in list(record.keys()):
            if key.startswith('atk_'):
                race['attack'][key[4:]] = record.pop(key)
            elif key.startswith('def_'):
                race['defence'][key[4:]] = record.pop(key)
            elif key.startswith('anm_') and key.endswith('_speed'):
                race['animation_speeds'][key[4:-6]] = record.pop(key)
        race.update(record)
        races[name] = race
    return races


def build_monsters(tables, races):
    monsters = merged_table(tables, 'units_monster_prototypes')
    for monster in monsters.values():
        race = races.get(monster.get('base_race'))
        monster['figure'] = race['figure'] if race else None
        complection = [monster.pop(f'complection_{axis}', 0) for axis in 'xyz']
        monster['complection'] = complection
        for key in ('weapon', 'second_weapon'):
            if monster.get(key) == '':
                monster[key] = None
    return monsters


def build_npcs(tables, monsters):
    npcs = merged_table(tables, 'units_npcs')
    for name, npc in npcs.items():
        # NPCs are the heroes and the mercenaries, their looks are in the monster prototype of the same name
        npc['prototype'] = name if name in monsters else None
    return npcs


def build_items(tables):
    return {
        'weapons': merged_table(tables, 'items_weapons'),
        'armors': merged_table(tables, 'items_armors'),
        'materials': merged_table(tables, 'items_materials'),
        'quick_items': merged_table(tables, 'items_quick_items'),
        'quest_items': merged_table(tables, 'items_quest_items'),
        'loot_items': merged_table(tables, 'items_loot_items'),
    }


def build_spells(tables):
    return {
        'prototypes': merged_table(tables, 'spells_spell_prototypes'),
        'modifiers': merged_table(tables, 'spells_spell_modifiers'),
        'templates': merged_table(tables, 'spells_spell_templates', key='prototype'),
        'armor_templates': merged_table(tables, 'spells_armor_spell_templates', key='prototype'),
        'weapon_templates': merged_table(tables, 'spells_weapon_spell_templates', key='prototype'),
    }


def build_skills(tables):
    return {
        'skills': merged_table(tables, 'perks_skills'),
        'perks': merged_table(tables, 'perks_perks'),
    }

#--- FILES OF THE SERVER AND THE CLIENT ---


def build_armors(tables):
    # armors.json: the armors which units can wear, the prototypes are the sets of the armors of the monsters
    return dict((r['name'], {
        'type': r['type'],
        'material_type': r['material type'].lower(),
        'texture_type_1': r['texture type 1'],
        'texture_type_2': r['texture type 2'],
    }) for r in tables.get(('main', 'items_armors'), []) if not r['name'].count('prototype'))


def build_weapons(tables):
    return dict((r['name'], {
        'type': r['type'],
        'material_type': r['material type'].lower(),
        'texture_type_1': r['texture type 1'],
        'texture_type_2': r['texture type 2'],
        'range': r['range'],
        'min_damage': r['min damage'],
        'max_damage': r['max damage'],
        'attack': r['attack'],
        'defence': r['defence'],
        'actions': r['actions'],
    }) for r in tables.get(('main', 'items_weapons'), []))


def build_materials(tables):
    return dict((r['name'], {
        'code': r['code'].lower(),
        'type': r['type'].lower(),
    }) for r in tables.get(('main', 'items_materials'), []))


def read_lnk(data):
    """
    The tree of the parts of a model of its .lnk file: [part name, [child tree, ...]], the same as lnk.read_info().
    The file is the list of the parts, every part with the name of its parent, the root part has no parent.
    """
    f = io.BytesIO(data)
    tree = []
    for _ in range(read_uint(f)):
        name = read_name(f, read_uint(f))
        parent_name_length = read_uint(f)
        if parent_name_length == 0:
            tree = [name, []]
        else:
            add_part(tree, read_name(f, parent_name_length), name)
    return tree


def read_uint(f):
    return struct.unpack('<I', f.read(4))[0]


def read_name(f, length):
    return f.read(length).split(b'\x00')[0].decode('latin-1')


def add_part(tree, parent_name, name):
    if tree[0] == parent_name:
        tree[1].append([name, []])
        return
    for child in tree[1]:
        add_part(child, parent_name, name)


def build_figures(game_dir_paths):
    """
    figures.json: the tree of the parts of every model of figures.res and menus.res, see read_lnk().
    A model with a few parts has <model>.lnk next to its <model>.mod, a model of one part
    has the .lnk named as the model inside its .mod.
    """
    result = {}
    for res_name in FIGURES_ARCHIVES:
        for file_name, data in sorted(read_joined_res(game_dir_paths, res_name).items()):
            model_name, _, ext = file_name.rpartition('.')
            if ext == 'lnk':
                result[model_name] = read_lnk(data)
        for file_name, data in sorted(read_joined_res(game_dir_paths, res_name).items()):
            model_name, _, ext = file_name.rpartition('.')
            if ext != 'mod' or model_name in result or not is_res(data):
                continue
            for name, part_data in read_res_data(io.BytesIO(data)).items():
                if name.lower() == model_name:
                    result[model_name] = read_lnk(part_data)
    return result

#--- FIGURES ---


def animation_frames(data):
    """
    Frames of an animation of figures.res: the animation is an archive with the movement of every part of the figure.
    """
    parts = read_res_data(io.BytesIO(data))
    frames = 0
    for part_data in parts.values():
        frames = max(frames, struct.unpack_from('<I', part_data, 0)[0])
    return frames, len(parts)


def build_figures_animations(game_dir_paths, adb_animations):
    """
    For every figure: its animations of figures.res with the number of frames and the parts they move,
    joined with the actions of its .adb file: what kind of action the animation is, which weapons it is for,
    at which frame the hit happens and so on.
    """
    figures_res = read_joined_res(game_dir_paths, 'figures.res')
    result = {}
    for file_name, data in sorted(figures_res.items()):
        figure, _, ext = file_name.rpartition('.')
        if ext != 'anm' or not is_res(data):
            continue
        animations = {}
        for animation_name, animation_data in read_res_data(io.BytesIO(data)).items():
            if not is_res(animation_data):
                continue
            frames, parts = animation_frames(animation_data)
            animations[animation_name.lower()] = {'frames': frames, 'parts': parts}
        result[figure] = {'animations': animations}
    for figure, info in adb_animations.items():
        figure_info = result.setdefault(figure, {'animations': {}})
        figure_info['name'] = info['name']
        figure_info['height'] = [info['minimal_height'], info['average_height'], info['maximal_height']]
        for action in info['actions']:
            animation = figure_info['animations'].setdefault(action['action_name'].lower(), {'frames': None, 'parts': None})
            action_type, _, modifier = action['action_type'].partition(':')
            animation.update({
                'action_type': action_type,
                'action_modifier': int(modifier or 0),
                'action_number': action['action_number'],
                'stage': action['animation_stage'],
                'allowed_states': action['allowed_states'],
                'forms': [f for f in action['action_forms'].split(',') if f],
                'weapons': [w for w in action['weapons'].split(',') if w],
                'probability': action['action_probability'],
                'length': action['animation_length'],
                'movement_speed': action['movement_speed'],
                'show_hide_frames': [action.get('show_hide_frame_1'), action.get('show_hide_frame_2')],
                'hit_frame': action.get('hit_frame'),
            })
    return result


def build_figures_redress(game_dir_paths):
    """
    For every figure: the textures of the weapons and the armors redress.res has for it,
    weapon or armor type -> ["<texture type 1>.<material code>.<texture type 2>", ...]
    """
    # some versions of the game, for example "Evil Islands: Rebirth", do not have redress.res
    names = joined_res_names(game_dir_paths, 'redress.res')
    pattern = re.compile(r'^(?P<figure>.+?)(?P<code>' + '|'.join(REDRESS_CODES) + r')_(?P<type1>\d+)\.(?P<material>[a-z0-9]+)\.(?P<type2>\d+)\.mmp$')
    result = {}
    unknown = []
    for name in names:
        m = pattern.match(name)
        if not m:
            unknown.append(name)
            continue
        figure_info = result.setdefault(m.group('figure'), {})
        variants = figure_info.setdefault(REDRESS_CODES[m.group('code')], [])
        variant = f"{int(m.group('type1')):02}.{m.group('material')}.{m.group('type2')}"
        if variant not in variants:
            variants.append(variant)
    for figure_info in result.values():
        for variants in figure_info.values():
            variants.sort()
    if unknown:
        if _Debug:
            print(f'unknown names in redress.res: {unknown}')
    return result

#--- SOUNDS ---


def build_sounds(game_dir_paths):
    """
    Sounds of sfx.res: folder -> names of the files. Races have the folders of their sounds in "sfx_path":
    <sfx_path>/attack, <sfx_path>/hit, <sfx_path>/death ..., the steps are in "steps_path" for every kind of ground.
    """
    result = {}
    for name in joined_res_names(game_dir_paths, 'sfx.res'):
        folder, _, file_name = name.rpartition('/')
        result.setdefault(folder, []).append(file_name)
    for files in result.values():
        files.sort()
    return result


def race_sounds(sounds, sfx_path):
    """
    Kinds of the sounds of the race: folder name under its sfx_path -> number of the sounds.
    """
    if not sfx_path:
        return {}
    prefix = sfx_path.lower().rstrip('/') + '/'
    result = {}
    for folder, files in sounds.items():
        if folder.startswith(prefix):
            kind = folder[len(prefix):]
            if '/' not in kind:
                result[kind] = len(files)
    return result

#--- MAPS ---


def map_units(nodes, result):
    for node in nodes:
        if isinstance(node, list) and len(node) == 2 and isinstance(node[1], list):
            if node[0] == 'UNIT':
                result.append(dict((n[0], n[1]) for n in node[1] if isinstance(n, list) and len(n) == 2))
            map_units(node[1], result)
    return result


def build_units_samples(game_dir_paths):
    """
    Units placed on the maps of the game, every figure has the list of the different looks the units of it have there:
    monster prototype, textures, parts of the figure, weapons, armors, spells and items, how many units on which maps.
    """
    # some copies of the game have only the res/ folder, a mod has the maps of its own campaign
    maps_dir_path = last_path(game_dir_paths, 'maps')
    looks = {}
    map_file_names = sorted(os.listdir(maps_dir_path)) if maps_dir_path else []
    for file_name in map_file_names:
        if not file_name.lower().endswith('.mob'):
            continue
        map_name = file_name[:-4].lower()
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                info = mob.read_info(os.path.join(maps_dir_path, file_name))
            except Exception as exc:
                info = None
                if _Debug:
                    print(f'failed to read {file_name}: {exc}')
        if not info:
            continue
        for unit in map_units(info, []):
            figure = (unit.get('OBJTEMPLATE') or '').lower()
            if not figure:
                continue
            look = {
                'prototype': unit.get('UNIT_PROTOTYPE') or unit.get('PARENTTEMPLATE'),
                'texture': unit.get('OBJPRIMTXTR'),
                'texture2': unit.get('OBJSECTXTR'),
                'body_parts': sorted(p.lower() for p in unit.get('OBJBODYPARTS') or ()),
                'complection': [round(c, 2) for c in unit.get('OBJCOMPLECTION') or ()],
                'weapons': list(unit.get('UNIT_WEAPONS') or ()),
                'armors': list(unit.get('UNIT_ARMORS') or ()),
                'spells': list(unit.get('UNIT_SPELLS') or ()),
                'quick_items': list(unit.get('UNIT_QUICK_ITEMS') or ()),
                'quest_items': list(unit.get('UNIT_QUEST_ITEMS') or ()),
            }
            key = json.dumps(look, sort_keys=True)
            entry = looks.setdefault(figure, {}).setdefault(key, dict(look, count=0, maps={}))
            entry['count'] += 1
            entry['maps'][map_name] = entry['maps'].get(map_name, 0) + 1
    result = {}
    for figure, figure_looks in looks.items():
        result[figure] = sorted(figure_looks.values(), key=lambda l: (-l['count'], str(l['prototype'])))
    return result

#--- REGISTRY ---


def build_figures_registry(races, monsters, npcs, animations, redress, sounds, samples):
    """
    Everything known about every figure: its races, monster prototypes and NPCs, how it moves,
    which kinds of actions its animations have and for which weapons, which weapons and armors it can show,
    its sounds and how many units of it the maps of the game have.
    """
    figures = set(animations) | set(redress) | set(samples)
    figures |= set(r['figure'] for r in races.values() if r['figure'])
    result = {}
    for figure in sorted(figures):
        figure_races = sorted(name for name, r in races.items() if r['figure'] == figure)
        figure_monsters = sorted(name for name, m in monsters.items() if m['figure'] == figure)
        figure_animations = animations.get(figure, {}).get('animations', {})
        actions = {}
        attack_weapons = set()
        for animation_name, animation in sorted(figure_animations.items()):
            action_type = animation.get('action_type', 'unknown')
            actions.setdefault(action_type, []).append(animation_name)
            if action_type == 'attack':
                attack_weapons.update(animation.get('weapons') or ['none', ])
        locomotions = sorted(set(str(races[r]['locomotion']) for r in figure_races))
        figure_sounds = {}
        for r in figure_races:
            for kind, count in race_sounds(sounds, races[r].get('sfx_path')).items():
                figure_sounds[kind] = max(figure_sounds.get(kind, 0), count)
        result[figure] = {
            'name': animations.get(figure, {}).get('name'),
            'races': figure_races,
            'locomotion': locomotions,
            'flying': 'air' in locomotions,
            'monsters': figure_monsters,
            'npcs': sorted(name for name, n in npcs.items() if n.get('prototype') in figure_monsters),
            'animations_count': len(figure_animations),
            'actions': actions,
            'attack_weapons': sorted(attack_weapons),
            'can_shoot': bool(attack_weapons & {'bow', 'crossbow'}),
            'redress': sorted(redress.get(figure, {}).keys()),
            'sounds': figure_sounds,
            'sfx_paths': sorted(set(races[r].get('sfx_path') for r in figure_races if races[r].get('sfx_path'))),
            'map_units': sum(look['count'] for look in samples.get(figure, [])),
            'map_looks': len(samples.get(figure, [])),
        }
    return result


def build_store(game_dir_paths):
    """
    All files of the catalog for one store: file name -> data, see CATALOG_FILES.
    """
    tables, adb_animations = read_databases(game_dir_paths)
    races = build_races(tables)
    monsters = build_monsters(tables, races)
    npcs = build_npcs(tables, monsters)
    animations = build_figures_animations(game_dir_paths, adb_animations)
    redress = build_figures_redress(game_dir_paths)
    sounds = build_sounds(game_dir_paths)
    samples = build_units_samples(game_dir_paths)
    return {
        'races.json': races,
        'monsters.json': monsters,
        'npcs.json': npcs,
        'items.json': build_items(tables),
        'spells.json': build_spells(tables),
        'skills.json': build_skills(tables),
        'hit_locations.json': merged_table(tables, 'units_hit_locations'),
        'figures_animations.json': animations,
        'figures_redress.json': redress,
        'sounds.json': sounds,
        'units_samples.json': samples,
        'figures_registry.json': build_figures_registry(races, monsters, npcs, animations, redress, sounds, samples),
        'figures.json': build_figures(game_dir_paths),
        'animations.json': adb_animations,
        'armors.json': build_armors(tables),
        'weapons.json': build_weapons(tables),
        'materials.json': build_materials(tables),
    }


def main():
    parser = argparse.ArgumentParser(description='builds the catalog of the units of the original game from a few versions of it')
    parser.add_argument('--store', dest='stores', action='append', nargs='+', required=True, metavar='STORE_ID GAME_DIR',
                        help='id of the store, the folder with the res/ and maps/ folders of the game, then the folders of its mods')
    parser.add_argument('--dest', dest='dest_dir_path', required=True, metavar='DEST_DIR', help='folder for the JSON files')
    args = parser.parse_args()
    stores = {}
    for store in args.stores:
        if len(store) < 2 or not store[0].isdigit():
            parser.error(f'--store needs the id of the store and the folder of the game: {" ".join(store)}')
        game_dir_paths = store[1:]
        if not res_paths(game_dir_paths, 'database.res'):
            parser.error('none of the folders has res/database.res: %s' % ', '.join(game_dir_paths))
        stores[int(store[0])] = game_dir_paths
    os.makedirs(args.dest_dir_path, exist_ok=True)
    files = {}
    for store_id, game_dir_paths in sorted(stores.items()):
        print(f'store {store_id}: {", ".join(game_dir_paths)}')
        for file_name, data in build_store(game_dir_paths).items():
            files.setdefault(file_name, {})[str(store_id)] = data
    for file_name, depth in CATALOG_FILES:
        stores_data = files[file_name]
        for store_id, data in stores_data.items():
            mark_store(data, int(store_id), depth)
        write_json(args.dest_dir_path, file_name, stores_data)
        print(f'{file_name}: ' + ', '.join(f'{len(data)} in store {store_id}' for store_id, data in sorted(stores_data.items())))


if __name__ == '__main__':
    main()
