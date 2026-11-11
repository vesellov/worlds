"""
The list of the saved games of the folder saves/ and the window which chooses one (game.exe 633770 and
634750).

A slot is a folder with info.sav, scenario.sav, mission.sav and shot.sav. info.sav starts with
u32 signature 0x111, u32 number of strings, float hour of the world and then that many strings (the allod,
the zone and so on), each ending with 0; the rest is a stamp. scenario.sav starts with 0xdead, mission.sav
with 0x102 and shot.sav with a compressed picture (eilib.sav reads their bodies).

The remake saves its own state into remake.json of a slot (capture, save_slot, load_slot). A slot of game.exe
without remake.json is loaded by read_game_save: the zone of mission.sav, the hour of info.sav, the party
of scenario.sav (the hero, its numbers and its items) and the place of the hero in mission.sav; the other
objects of the map come from its .mob as on the entry into the zone.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import json
import os

#------------------------------------------------------------------------------

from eilib import sav as sav_module
from engine import combat
from engine import equipment
from engine import experience
from engine import roster as roster_module
from engine import unit as unit_module

#------------------------------------------------------------------------------

SAVES_DIR_NAME = 'saves'

INFO_NAME = sav_module.INFO_NAME
SCENARIO_NAME = sav_module.SCENARIO_NAME
MISSION_NAME = sav_module.MISSION_NAME
SHOT_NAME = sav_module.SHOT_NAME

# the state of the remake in a slot of its own: JSON next to info.sav
REMAKE_NAME = 'remake.json'
REMAKE_VERSION = 1

INFO_SIGNATURE = sav_module.INFO_SIGNATURE
SCENARIO_SIGNATURE = sav_module.SCENARIO_SIGNATURE
MISSION_SIGNATURE = sav_module.MISSION_SIGNATURE
SHOT_SIGNATURE = sav_module.SHOT_SIGNATURE

# the window on the screen of 800 x 600
PANEL_RECT = (150, 100, 650, 500)
TITLE_BOX = (150, 106, 650, 142)
ROWS_TOP = 160
ROW_HEIGHT = 24
VISIBLE_ROWS = 11
ROWS_X = (180, 620)
OK_RECT = (213, 526, 287, 574)
CANCEL_RECT = (513, 526, 587, 574)

# the slots which game.exe names itself (653a10 gives them the texts of texts.res in the list): F5 saves
# to saveq; the autosave of the entry into a zone goes to savea (6360b0), the one of the exit from a zone,
# when the map of the allod opens, to saveb (62eb94)
QUICK_SLOT_NAME = 'saveq'
AUTO_ENTER_SLOT_NAME = 'savea'
AUTO_EXIT_SLOT_NAME = 'saveb'
SLOT_TEXT_NAMES = {
    QUICK_SLOT_NAME: 'string save_quick',
    AUTO_ENTER_SLOT_NAME: 'string save_auto_enter',
    AUTO_EXIT_SLOT_NAME: 'string save_auto_exit',
}

# CWorldClient::vf06 (5641f0, partly) counts the ticks of the world in the zone and sends the string event
# "autosave" to the interface on the 10th one
AUTOSAVE_TICK = 10

LOAD_TITLE = 'Load game'
SAVE_TITLE = 'Save game'

#------------------------------------------------------------------------------


class Slot(object):
    """
    One folder of saves/: its name and what info.sav says.
    """

    def __init__(self, name, dir_path, hour=0.0, strings=(), remake=False):
        self.name = name
        self.dir_path = dir_path
        self.hour = hour
        self.strings = tuple(strings)
        # the slot has the state of the remake (remake.json) and can be loaded by it
        self.remake = remake

    def __repr__(self):
        return 'Slot(%s, %s%s)' % (self.name, self.strings, ', remake' if self.remake else '')

    @property
    def allod(self):
        return self.strings[0] if self.strings else ''

    @property
    def zone(self):
        return self.strings[1] if len(self.strings) > 1 else ''


def slot_title(name, text_of):
    """
    The name of a slot in the list: the text of texts.res for saveq, savea and saveb (653a10), else the name
    of the folder. text_of is GameData.text.
    """
    text_name = SLOT_TEXT_NAMES.get(name)
    text = text_of(text_name) if text_name else None
    return text.strip() if text and text.strip() else name


def autosave_due(ticks_before, ticks_after):
    """
    True when the ticks of the world in the zone went over AUTOSAVE_TICK in this frame (5641f0, partly).
    """
    return ticks_before < AUTOSAVE_TICK <= ticks_after


def read_info(data):
    """
    (hour, strings) of info.sav; (0.0, ()) when the data is not an info.sav.
    """
    info = sav_module.read_info(data)
    if info is None:
        return 0.0, ()
    return info.hour, info.strings


def slots(data_dir_path):
    """
    The slots of data/saves, sorted by name; a game slot has mission.sav, a remake one has remake.json.
    """
    saves_dir_path = os.path.join(data_dir_path, SAVES_DIR_NAME)
    if not os.path.isdir(saves_dir_path):
        return []
    result = []
    for name in sorted(os.listdir(saves_dir_path)):
        dir_path = os.path.join(saves_dir_path, name)
        if not os.path.isdir(dir_path):
            continue
        remake = os.path.isfile(os.path.join(dir_path, REMAKE_NAME))
        if not remake and not os.path.isfile(os.path.join(dir_path, MISSION_NAME)):
            continue
        hour = 0.0
        strings = ()
        info_path = os.path.join(dir_path, INFO_NAME)
        if os.path.isfile(info_path):
            with open(info_path, 'rb') as f:
                hour, strings = read_info(f.read())
        result.append(Slot(name, dir_path, hour, strings, remake))
    return result


class Dialog(object):
    """
    The list of the slots: the selection, the clicks and the keys of the window of 633770 and 634750.
    """

    def __init__(self, entries, title=LOAD_TITLE):
        self.entries = list(entries)
        self.title = title
        self.selected = 0
        self.scroll = 0

    def __repr__(self):
        return 'SaveDialog(%s, %d slots)' % (self.title, len(self.entries))

    @property
    def selected_slot(self):
        if 0 <= self.selected < len(self.entries):
            return self.entries[self.selected]
        return None

    def select(self, index):
        if not self.entries:
            return
        index = min(max(index, 0), len(self.entries) - 1)
        self.selected = index
        if index < self.scroll:
            self.scroll = index
        elif index >= self.scroll + VISIBLE_ROWS:
            self.scroll = index - VISIBLE_ROWS + 1

    def row_rect(self, row):
        top = ROWS_TOP + row * ROW_HEIGHT
        return (ROWS_X[0], top, ROWS_X[1], top + ROW_HEIGHT)

    def click(self, x, y):
        """
        "select", "confirm", "cancel" or None.
        """
        for row in range(VISIBLE_ROWS):
            if _inside(self.row_rect(row), x, y):
                index = self.scroll + row
                if index < len(self.entries):
                    self.select(index)
                    return 'select'
        if _inside(OK_RECT, x, y):
            return 'confirm'
        if _inside(CANCEL_RECT, x, y):
            return 'cancel'
        return None

    def on_key(self, name):
        if name == 'up':
            self.select(self.selected - 1)
        elif name == 'down':
            self.select(self.selected + 1)
        elif name in ('enter', 'spacebar'):
            return 'confirm'
        elif name == 'escape':
            return 'cancel'
        return None

#------------------------------------------------------------------------------
# the state of the remake in remake.json


def _value(value):
    """
    A value of the scripts or of the world in JSON: the lists and tuples become lists, a dictionary a map of
    strings, everything else is kept by repr for the next load to report.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    return repr(value)


def _script_value(value):
    """
    The value of a script back from JSON; the remake saves only the simple kinds (the groups of objects are
    rebuilt by the scripts).
    """
    if isinstance(value, list):
        return tuple(_script_value(item) for item in value)
    return value


def item_state(item):
    """
    The state of an engine.equipment.Item: its name, its durability and its count.
    """
    return {'name': item.name, 'durability': float(item.durability), 'count': int(item.count)}


def restore_item(item, data):
    """
    The durability and the count of an Item made from its name; the shape gives the maximum durability.
    """
    if item is None or data is None:
        return item
    item.durability = min(float(data.get('durability', item.max_durability)), item.max_durability)
    item.count = max(1, int(data.get('count', 1)))
    return item


def hero_stats_state(stats):
    return {
        'experience': stats.experience,
        'start_experience': stats.start_experience,
        'waiting_experience': stats.waiting_experience,
        'perk_experience': getattr(stats, 'perk_experience', 0.0),
        'base_strength': stats.base_strength,
        'base_dexterity': stats.base_dexterity,
        'base_intelligence': stats.base_intelligence,
        'science': stats.science,
        'skills': list(stats.skills),
        'perk_levels': list(stats.perk_levels),
    }


def restore_hero_stats(stats, data):
    if stats is None or data is None:
        return stats
    stats.experience = float(data.get('experience', stats.experience))
    stats.start_experience = float(data.get('start_experience', stats.start_experience))
    stats.waiting_experience = float(data.get('waiting_experience', stats.waiting_experience))
    stats.perk_experience = float(data.get('perk_experience', 0.0))
    stats.base_strength = float(data.get('base_strength', stats.base_strength))
    stats.base_dexterity = float(data.get('base_dexterity', stats.base_dexterity))
    stats.base_intelligence = float(data.get('base_intelligence', stats.base_intelligence))
    stats.science = int(data.get('science', stats.science))
    stats.skills = list(data.get('skills', stats.skills))
    stats.perk_levels = list(data.get('perk_levels', stats.perk_levels))
    return stats


def record_state(record):
    """
    A UnitRecord of the roster: its names, its things and the wear of the worn things, its HeroStats.
    """
    return {
        'name': record.name,
        'prototype_name': record.prototype_name,
        'stats_name': record.stats_name,
        'model_name': record.model_name,
        'item_names': None if record.item_names is None else list(record.item_names),
        'quick_items': None if record.quick_items is None else list(record.quick_items),
        'worn_items': {name: item_state(item) for name, item in record.worn_items.items()},
        'hero_stats': None if record.hero_stats is None else hero_stats_state(record.hero_stats),
    }


def restore_record(record, data, game_data):
    record.name = data.get('name', record.name)
    record.prototype_name = data.get('prototype_name', record.prototype_name)
    record.stats_name = data.get('stats_name', record.stats_name)
    record.model_name = data.get('model_name', record.model_name)
    item_names = data.get('item_names')
    record.item_names = None if item_names is None else list(item_names)
    quick_items = data.get('quick_items')
    record.quick_items = None if quick_items is None else list(quick_items)
    record.worn_items = {}
    for name, state in (data.get('worn_items') or {}).items():
        item = equipment.make_item(game_data.items(), name)
        if item is not None:
            record.worn_items[name] = restore_item(item, state)
    if data.get('hero_stats') is not None:
        if record.hero_stats is None:
            settings = experience.RpgSettings.from_registry(game_data.registry('config/ai.reg'))
            record.hero_stats = experience.from_prototype(game_data.units(), record.stats_name,
                                                          game_data.perks(), settings)
        restore_hero_stats(record.hero_stats, data['hero_stats'])
    return record


def roster_state(roster):
    if roster is None:
        return None
    parties = [roster.default] + [p for p in roster.parties if p is not roster.default]
    return {
        'current_name': roster.current_name,
        'parties': [{
            'name': party.name,
            'loot': list(party.loot),
            'units': [record_state(record) for record in party.units],
        } for party in parties],
    }


def restore_roster(roster, data, game_data):
    if roster is None or data is None:
        return roster
    roster.default = roster_module.UnitParty(roster_module.DEFAULT_PARTY)
    roster.parties = []
    roster.current_name = data.get('current_name', roster_module.DEFAULT_PARTY)
    for party_data in data.get('parties', []):
        party = roster.default if party_data.get('name', '') == roster_module.DEFAULT_PARTY else roster_module.UnitParty(
            party_data.get('name', ''))
        party.loot = list(party_data.get('loot', []))
        for data_record in party_data.get('units', []):
            record = roster_module.UnitRecord(data_record.get('name', ''),
                                              data_record.get('prototype_name', ''),
                                              data_record.get('stats_name'),
                                              data_record.get('item_names'),
                                              data_record.get('model_name'))
            restore_record(record, data_record, game_data)
            party.units.append(record)
        if party is not roster.default:
            roster.parties.append(party)
    return roster


def party_state(party):
    return {
        'money': int(party.money),
        'items': list(party.items),
        'quest_items': [{'name': item.name, 'number': item.number} for item in party.quest_items],
        'completed_quests': list(party.completed_quests),
    }


def restore_party(party, data, game_data):
    if party is None or data is None:
        return party
    party.money = int(data.get('money', 0))
    party.items = list(data.get('items', []))
    database = game_data.items()
    party.quest_items = []
    for item_data in data.get('quest_items', []):
        number = int(item_data.get('number', -1))
        found = next((q for q in database.quest_items if q.number == number), None)
        if found is None:
            found = database.quest_item(item_data.get('name', ''))
        if found is not None:
            party.quest_items.append(found)
    party.completed_quests = list(data.get('completed_quests', []))
    return party


def fighter_state(unit):
    """
    The numbers of the fight of a unit: the parts, the mana, the spells on it and the strike delay.
    """
    fighter = unit.fighter
    state = {'dead': unit.is_dead}
    if fighter is None:
        return state
    state.update({
        'mana': float(fighter.mana),
        'max_mana': float(fighter.max_mana),
        'start_health': float(fighter.start_health),
        'strike_wait': float(unit.strike_wait),
        'parts': [{'name': part.name, 'health': float(part.health), 'state': int(part.state)}
                  for part in fighter.parts],
        'enchantments': [{'index': e.index, 'ticks': e.ticks, 'effect': e.effect} for e in fighter.enchantments],
    })
    return state


def restore_fighter(unit, data):
    if unit is None or data is None:
        return
    fighter = unit.fighter
    if fighter is None:
        return
    fighter.mana = min(float(data.get('mana', fighter.mana)), fighter.max_mana)
    fighter.max_mana = float(data.get('max_mana', fighter.max_mana))
    fighter.start_health = float(data.get('start_health', fighter.start_health))
    unit.strike_wait = float(data.get('strike_wait', unit.strike_wait))
    parts = {part.name: part for part in fighter.parts}
    for part_data in data.get('parts', []):
        part = parts.get(part_data.get('name'))
        if part is not None:
            part.health = min(float(part_data.get('health', part.health)), part.max_health)
            part.state = int(part_data.get('state', part.state))
    fighter.enchantments = [combat_enchantment(item) for item in data.get('enchantments', [])]


def combat_enchantment(data):
    return combat.Enchantment(int(data.get('index', 0)), float(data.get('ticks', 0.0)),
                              float(data.get('effect', 0.0)))


def unit_state(unit, script_host):
    """
    One unit of the map: its NID in the scripts, where it stands, its hidden flags and its fight.
    """
    return {
        'nid': script_host.object_id(unit) if script_host is not None else 0,
        'x': float(unit.x),
        'y': float(unit.y),
        'yaw': float(unit.yaw),
        'hidden': bool(unit.hidden),
        'looted': bool(unit.looted),
        'blocked': bool(unit.blocked),
        'aim_part': unit.aim_part,
        'quest_items': [item.number for item in unit.quest_items],
        'fighter': fighter_state(unit),
    }


def restore_unit(unit, data, game_data):
    unit.x = float(data.get('x', unit.x))
    unit.y = float(data.get('y', unit.y))
    unit.z = unit.ground_height()
    unit.yaw = unit.target_yaw = float(data.get('yaw', unit.yaw))
    unit.hidden = bool(data.get('hidden', False))
    unit.looted = bool(data.get('looted', False))
    unit.blocked = bool(data.get('blocked', False))
    unit.aim_part = data.get('aim_part')
    database = game_data.items()
    numbers = set(int(number) for number in data.get('quest_items', []))
    unit.quest_items = [q for q in database.quest_items if q.number in numbers]
    restore_fighter(unit, data.get('fighter') or {})
    unit.mode = unit_module.MODE_IDLE
    unit.target = None
    unit.waypoints = []
    unit.foe = None


def lie_dead(unit):
    """
    The body of a unit which died before the save: the last frame of its death, no hooks of a new death.
    """
    unit.die()
    player = unit.player
    if player.clip is not None:
        player.time = float(player.clip.length)
        player.finished = True


def lever_state(app):
    result = []
    for lever in app.levers:
        result.append({
            'nid': app.script_host.object_id(lever),
            'state': lever.lever.state,
            'enabled': lever.lever.enabled,
        })
    return result


def trap_state(app):
    result = []
    for trap in app.traps:
        result.append({
            'nid': app.script_host.object_id(trap),
            'active': bool(trap.active),
            'counter': float(getattr(trap, 'counter', 0.0)),
        })
    return result


def camera_state(camera):
    return {
        'target': [float(v) for v in camera.target],
        'yaw': float(camera.yaw),
        'pitch': float(camera.pitch),
        'distance': float(camera.distance),
    }


def restore_camera(camera, data):
    if camera is None or data is None:
        return
    camera.target[:] = [float(v) for v in data.get('target', camera.target)]
    camera.yaw = float(data.get('yaw', camera.yaw))
    camera.pitch = float(data.get('pitch', camera.pitch))
    camera.distance = float(data.get('distance', camera.distance))


def capture(app):
    """
    The whole state of the game of the remake: the map and its hour, the party and the roster, the hero,
    the units of the map, the levers and the traps, the variables of the scripts and the camera.
    """
    game_data = app.game_data
    zone = game_data.zone_of_map(app.map_name) if app.map_name else None
    record = app.hero_record if app.hero_record is not None else (
        app.roster.current().lead() if app.roster is not None else None)
    if record is not None and app.hero_item_names is not None:
        record.item_names = list(app.hero_item_names)
    state = {
        'version': REMAKE_VERSION,
        'map_name': app.map_name,
        'mob_name': zone.mob_name if zone is not None else app.map_name,
        'zone_name': zone.name if zone is not None else '',
        'allod': zone.allod if zone is not None else '',
        'hour': float(app.hour),
        'fast_hours': bool(app.fast_hours),
        'game_vars': {name: _value(value) for name, value in (app.game_vars or {}).items()},
        'party': party_state(app.party),
        'roster': roster_state(app.roster),
        'hero': None,
        'units': [unit_state(unit, app.script_host) for unit in app.map_units],
        'companions': [dict(unit_state(unit, app.script_host), name=unit.name)
                       for unit in getattr(app, 'companions', ())],
        'levers': lever_state(app),
        'traps': trap_state(app),
        'camera': camera_state(app.scene.camera) if app.scene is not None else None,
    }
    if app.unit is not None:
        hero = unit_state(app.unit, app.script_host)
        hero['prototype_name'] = app.unit.prototype_name
        state['hero'] = hero
    return state


def restore_world(app, state):
    """
    The units, the hero and the levers after setup_map made the map of the save: the units take their places and
    numbers, the dead ones lie where they fell (game.exe keeps the bodies with their loot), the levers their states.
    """
    saved_units = {int(item.get('nid', 0)): item for item in state.get('units', [])}
    for unit in list(app.map_units):
        nid = app.script_host.object_id(unit)
        data = saved_units.get(nid)
        if data is None:
            continue
        restore_unit(unit, data, app.game_data)
        if data.get('fighter', {}).get('dead'):
            lie_dead(unit)
    saved_companions = {data.get('name'): data for data in state.get('companions', [])}
    for unit in getattr(app, 'companions', ()):
        data = saved_companions.get(unit.name)
        if data is None:
            continue
        restore_unit(unit, data, app.game_data)
        record = getattr(unit, 'record', None)
        if record is not None and record.hero_stats is not None:
            unit.hero_stats = record.hero_stats
            experience.update_fighter(unit.fighter, unit.hero_stats)
    hero_data = state.get('hero')
    if hero_data is not None and app.unit is not None:
        restore_unit(app.unit, hero_data, app.game_data)
        app.unit.hero_stats = app.hero_record.hero_stats if app.hero_record is not None else app.unit.hero_stats
        experience.update_fighter(app.unit.fighter, app.unit.hero_stats)
        for part_data in hero_data.get('fighter', {}).get('parts', []):
            part = next((p for p in app.unit.fighter.parts if p.name == part_data.get('name')), None)
            if part is not None:
                part.health = min(float(part_data.get('health', part.health)), part.max_health)
                part.state = int(part_data.get('state', part.state))
        app.unit.fighter.mana = min(float(hero_data.get('fighter', {}).get('mana', app.unit.fighter.mana)),
                                   app.unit.fighter.max_mana)
    for data in state.get('levers', []):
        obj = app.script_host.object_by_id(int(data.get('nid', 0)))
        if obj is not None and hasattr(obj, 'lever'):
            obj.lever.enabled = bool(data.get('enabled', True))
            obj.lever.switch(int(data.get('state', 0)), ticks=0)
            obj.lever.shown_phase = obj.lever.phase
    for data in state.get('traps', []):
        obj = app.script_host.object_by_id(int(data.get('nid', 0)))
        if obj is not None and hasattr(obj, 'active'):
            obj.active = bool(data.get('active', True))
            obj.counter = float(data.get('counter', 0.0))
    restore_camera(app.scene.camera if app.scene is not None else None, state.get('camera'))
    if app.unit is not None and app.args.follow:
        app.scene.camera.target[:] = (app.unit.x, app.unit.y, app.unit.z + 1.0)


def save_slot(dir_path, state):
    """
    Write remake.json and the game-compatible info.sav of a slot. saveq, savea and saveb are the slots of
    game.exe too: when the folder holds a save of game.exe (mission.sav), its info.sav stays and only
    remake.json is written next to it, so the save of the game is not broken.
    """
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, REMAKE_NAME), 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    if os.path.isfile(os.path.join(dir_path, MISSION_NAME)):
        return dir_path
    data = sav_module.build_info(float(state.get('hour', 0.0)),
                                 [state.get('allod', ''), state.get('zone_name', '')])
    with open(os.path.join(dir_path, INFO_NAME), 'wb') as f:
        f.write(data)
    return dir_path


def load_slot(dir_path):
    """
    The state of a slot, None when it has no remake.json.
    """
    path = os.path.join(dir_path, REMAKE_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or int(state.get('version', 0)) > REMAKE_VERSION:
        return None
    state['game_vars'] = {name: _script_value(value) for name, value in (state.get('game_vars') or {}).items()}
    return state


def slot_dir(data_dir_path, name):
    return os.path.join(data_dir_path, SAVES_DIR_NAME, name)


def next_name(data_dir_path, prefix='save'):
    saves_dir_path = os.path.join(data_dir_path, SAVES_DIR_NAME)
    number = 1
    while os.path.isdir(os.path.join(saves_dir_path, '%s%d' % (prefix, number))):
        number += 1
    return '%s%d' % (prefix, number)


#------------------------------------------------------------------------------


def _inside(rect, x, y):
    left, top, right, bottom = rect
    return left <= x < right and top <= y < bottom


#------------------------------------------------------------------------------


class GameSave(object):
    """
    What the remake takes from a save of game.exe (read_game_save).
    """

    def __init__(self, map_name, mob_name, zone, hour, roster, other_item_names=(), hero_place=None):
        self.map_name = map_name
        self.mob_name = mob_name
        self.zone = zone
        self.hour = hour
        self.roster = roster
        # the items which are not worn (the quick items, the bag)
        self.other_item_names = list(other_item_names)
        # ((x, y, z), yaw) of the hero, None when mission.sav does not tell it
        self.hero_place = hero_place

    def __repr__(self):
        return 'GameSave(%s, %s, %s)' % (self.map_name, self.zone, self.roster)


def _base_name(file_name):
    return os.path.splitext(os.path.basename(file_name))[0].lower()


def read_game_save(dir_path, game_data):
    """
    GameSave of a slot of game.exe, None when it has no readable scenario.sav and mission.sav.
    """
    try:
        with open(os.path.join(dir_path, SCENARIO_NAME), 'rb') as f:
            scenario = sav_module.read_body(f.read())
        with open(os.path.join(dir_path, MISSION_NAME), 'rb') as f:
            mission = sav_module.read_body(f.read())
        mob_file_name, mpr_file_name, zone = sav_module.mission_files(mission.body)
    except (OSError, sav_module.SaveError):
        return None
    hour = None
    try:
        with open(os.path.join(dir_path, INFO_NAME), 'rb') as f:
            info = sav_module.read_info(f.read())
        hour = info.hour if info is not None else None
    except OSError:
        pass
    parties = sav_module.scenario_parties(scenario.body)
    party = None
    for candidate in parties:
        if candidate.zone.lower() == zone.lower() and candidate.units:
            party = candidate
            break
    if party is None:
        return None
    roster = roster_module.Roster()
    others = []
    for unit in party.units:
        record, other = roster_module.record_from_saved(unit, party, game_data.items(), game_data.quick_items(),
                                                        perks=game_data.perks())
        roster.default.units.append(record)
        others.extend(other)
    hero = party.units[0]
    place = sav_module.find_unit_place(mission.body, hero.model_name, hero.prototype_name, hero.stats_name or None)
    return GameSave(_base_name(mpr_file_name), _base_name(mob_file_name), zone, hour, roster, others, place)


def game_save_is_newer(dir_path):
    """
    True when scenario.sav of game.exe in the slot was written after remake.json (or there is no remake.json):
    the slots of the game (saveq, savea, saveb) are shared by both.
    """
    scenario_path = os.path.join(dir_path, SCENARIO_NAME)
    remake_path = os.path.join(dir_path, REMAKE_NAME)
    if not os.path.isfile(scenario_path):
        return False
    if not os.path.isfile(remake_path):
        return True
    return os.path.getmtime(scenario_path) > os.path.getmtime(remake_path)
