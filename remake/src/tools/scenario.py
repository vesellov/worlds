"""
Extracts the scenario of a game into one JSON file: the quests and the briefings of quests.qdb with their texts,
the characters who own the briefings and speak in them, the named units of all maps, and the links between them
found in the scripts of the maps.

    PYTHONPATH=src venv/bin/python -m tools.scenario <data folder> -o ei1.json [--texts other/texts.res]

The data folder is the one with res/ and maps/ (the names are compared without the case). --texts takes the texts of another language
version, for example the Russian texts.res for the English data.

How the game keeps the scenario (all of it is in the global variables of the player, GSGetVar/GSSetVar):
    "b.<owner>.<briefing>"   1 the briefing is available, 2 it was heard; the owner is the name (OBJNAME) of
                             the unit of a briefing zone who tells it, or the name of the zone itself for
                             a briefing which starts when the party comes in
    "q.<zone>.<quest>"       1 the quest is given, 2 it is done; "q.<zone>.<quest>.<n>" are its subobjectives
    "z.<zone>"               1 the zone is open on the map of the allod, "z.<zone>_<zone>" a road between two
quests.qdb section 1 holds the quests (the name, the experience, the comment of the editor, the index),
section 2 the briefings: the tag 2 money (negative: the party pays), 3 items, 5 quest items, 6 the variables
of the quests set to 1, 7 to 2, 8 the zones opened, 9 the zones closed.
The briefing zones of EI1 also keep VSS_BS_CHECK / VSS_BS_COMMANDS: the commands run after a briefing was heard.

EI2 does not use OBJNAME to find a unit by its name: the NID of a named unit is the hash of its name in capitals
(game.exe FUN_00431580: h = h * 5 + c over the bytes, then h % 1000000000 + 1000000000), so the unit "Baron"
of a briefing zone has the NID name_hash("Baron") and its OBJNAME is any text of the editor.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import re
import sys
import json
import struct
import argparse
import collections

#------------------------------------------------------------------------------

from eilib import idb
from eilib import mob
from eilib import res
from eilib import zones
from eilib import gamedata

#------------------------------------------------------------------------------

ENCODING = 'cp1251'

TOKEN_RE = re.compile(r'"[^"]*"|[A-Za-z_#][\w#.]*|-?\d+(?:\.\d+)?|[(),:=]|\S')

VAR_RE = re.compile(r'^([bqz])\.(.+)$')

# commands of the scripts which matter for the scenario besides the variables
NOTABLE_COMMANDS = (
    'QuestComplete', 'GiveItem', 'GiveQuestItem', 'EraseQuestItem', 'RemoveQuestItem', 'GiveUnitQuestItem',
    'GiveMoney', 'AddUnitToParty', 'RemoveUnitFromParty', 'CreateParty', 'SetCurrentParty', 'LeaveToZone',
    'PlayMovie', 'GameOver', 'AddMob', 'SendStringEvent', 'RecalcMercBriefings', 'RedeployParty',
)

# conditions on the units which tell the targets of the quests
UNIT_CONDITIONS = ('IsDead', 'IsAlive', 'UnitInSquare', 'UnitSee', 'GroupSee', 'IsUnitInArea', 'HaveItem')

QUEST_FIELD_EXPERIENCE = 1
QUEST_FIELD_STEPS = 3
QUEST_FIELD_COMMENT = 4
QUEST_FIELD_INDEX = 6

BRIEF_FIELD_EXPERIENCE = 1
BRIEF_FIELD_MONEY = 2
BRIEF_FIELD_ITEMS = 3
BRIEF_FIELD_COMMENT = 4
BRIEF_FIELD_QUEST_ITEMS = 5
BRIEF_FIELD_QUESTS_OPEN = 6
BRIEF_FIELD_QUESTS_DONE = 7
BRIEF_FIELD_ZONES_OPEN = 8
BRIEF_FIELD_ZONES_CLOSE = 9

# the stats of UNIT_STATS of a .mob shown for a character (index -> name), see engine.stats
STAT_FIELDS = (
    (1, 'health'), (3, 'mana'), (16, 'attack'), (17, 'parry'), (19, 'damage_min'), (20, 'damage_spread'),
    (28, 'armor'), (15, 'attack_range'),
)
INTEGER_STATS = (0, 1, 2, 3, 14, 28)

# the characters who have no named unit on the maps: the id of a character -> parts of prototype names
# tried in order. The heroes join the party from the prototype, "shava" is the old name of "spletnica".
CHARACTER_PROTOTYPES = {
    'hero': ('human hero', ),
    'hero2': ('hero2', ),
    'shava': ('bz11g human shava', ),
}

# the ids of the cast which are generic voices, not characters
NOT_CHARACTERS = ('annofcam', 'human male', 'human female')

#------------------------------------------------------------------------------


def name_hash(name):
    """
    The NID which EI2 gives to the unit with that name.
    """
    value = 0
    for c in name.upper().encode(ENCODING, errors='replace'):
        value = (value * 5 + c) & 0xffffffff
    return value % 1000000000 + 1000000000

#------------------------------------------------------------------------------


class Texts(object):
    """
    The texts of texts.res by their names, the case of the names is ignored.
    """

    def __init__(self, archive):
        self.archive = archive

    def get(self, name):
        data = self.archive.get(name)
        if data is None:
            return ''
        return bytes(data).decode(ENCODING, errors='replace').replace('\r\n', '\n').replace('\r', '\n')

    def first_line(self, name):
        text = self.get(name).strip()
        return text.split('\n', 1)[0].strip() if text else ''

    def names(self, kind):
        prefix = kind.lower() + ' '
        return [n for n in self.archive.names if n.lower().startswith(prefix)]

#------------------------------------------------------------------------------


def parse_briefing_text(text):
    """
    The title, the cast (#show), the phrases [(speaker, text, description)] of a "briefing x" text.
    """
    lines = text.split('\n')
    title = ''
    cast = []
    phrases = []
    current = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            words = stripped[1:].split()
            keyword = words[0].lower() if words else ''
            if keyword == 'show' and len(words) > 1:
                if words[1] not in cast:
                    cast.append(words[1])
            elif keyword == 'phrase' and len(words) > 1:
                current = [words[1], '', '']
                phrases.append(current)
            elif keyword == 'desc' and current is not None:
                current[2] = stripped[5:].strip()
            continue
        if not title and current is None:
            title = stripped
            continue
        if current is not None:
            current[1] = (current[1] + ' ' + stripped).strip()
    return title, cast, [tuple(p) for p in phrases]


def parse_quest_text(text):
    """
    The title, the description and the subobjectives [(number, title, text)] of a "quest x" text.
    """
    blocks = re.split(r'^#subobj\s+', text, flags=re.M)
    head = [line.strip() for line in blocks[0].strip().split('\n') if line.strip()]
    title = head[0] if head else ''
    description = ' '.join(head[1:])
    subobjectives = []
    for block in blocks[1:]:
        block_lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
        if not block_lines:
            continue
        number = block_lines[0]
        sub_title = block_lines[1] if len(block_lines) > 1 else ''
        subobjectives.append((number, sub_title, ' '.join(block_lines[2:])))
    return title, description, subobjectives

#------------------------------------------------------------------------------


def node_text(data):
    return bytes(data).split(b'\0', 1)[0].decode(ENCODING, errors='replace')


def node_texts(data):
    return [node_text(child.data) for child in idb.iter_nodes(data)]


def node_float(data):
    if len(data) < 4:
        return 0.0
    return struct.unpack_from('<f', data)[0]


def node_dword(data):
    if len(data) < 4:
        return 0
    return struct.unpack_from('<i', data)[0]


def records(sections, tag):
    section = sections.get(tag)
    return section.children() if section is not None else []


def read_quest_database(data):
    sections = {node.tag: node for node in idb.root(data).children()}
    quests = []
    for record in records(sections, 1):
        fields = record.fields()
        name = node_text(fields.get(0, b''))
        if not name or name == 'dummy':
            continue
        quests.append({
            'id': name,
            'experience': round(node_float(fields.get(QUEST_FIELD_EXPERIENCE, b'')), 1),
            'steps': node_dword(fields.get(QUEST_FIELD_STEPS, b'')),
            'comment': node_text(fields.get(QUEST_FIELD_COMMENT, b'')),
            'index': node_dword(fields.get(QUEST_FIELD_INDEX, b'')),
        })
    briefings = []
    for record in records(sections, 2):
        fields = record.fields()
        name = node_text(fields.get(0, b''))
        if not name:
            continue
        briefings.append({
            'id': name,
            'experience': round(node_float(fields.get(BRIEF_FIELD_EXPERIENCE, b'')), 1),
            'money': round(node_float(fields.get(BRIEF_FIELD_MONEY, b'')), 1),
            'items': node_texts(fields.get(BRIEF_FIELD_ITEMS, b'')),
            'comment': node_text(fields.get(BRIEF_FIELD_COMMENT, b'')),
            'quest_items': node_texts(fields.get(BRIEF_FIELD_QUEST_ITEMS, b'')),
            'quests_open': node_texts(fields.get(BRIEF_FIELD_QUESTS_OPEN, b'')),
            'quests_done': node_texts(fields.get(BRIEF_FIELD_QUESTS_DONE, b'')),
            'zones_open': node_texts(fields.get(BRIEF_FIELD_ZONES_OPEN, b'')),
            'zones_close': node_texts(fields.get(BRIEF_FIELD_ZONES_CLOSE, b'')),
        })
    return quests, briefings

#------------------------------------------------------------------------------


def strip_comments(text):
    lines = []
    for line in text.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        position = 0
        in_string = False
        cut = len(line)
        while position < len(line):
            c = line[position]
            if c == '"':
                in_string = not in_string
            elif not in_string and line.startswith('//', position):
                cut = position
                break
            position += 1
        lines.append(line[:cut])
    return '\n'.join(lines)


def tokenize(text):
    return TOKEN_RE.findall(strip_comments(text))


def parse_expression(tokens, position):
    """
    (node, next position): ('call', name, [args], block) / ('str', text) / ('num', text) / ('id', name).
    """
    token = tokens[position]
    position += 1
    if token.startswith('"'):
        return ('str', token[1:-1]), position
    if position < len(tokens) and tokens[position] == '(' and token not in '(),':
        args, position = parse_arguments(tokens, position)
        return ('call', token, args, None), position
    if re.match(r'^-?\d', token):
        return ('num', token), position
    return ('id', token), position


def parse_arguments(tokens, position):
    args = []
    position += 1
    while position < len(tokens) and tokens[position] != ')':
        if tokens[position] == ',':
            position += 1
            continue
        arg, position = parse_expression(tokens, position)
        args.append(arg)
    return args, position + 1


def parse_block(tokens, position):
    """
    The commands of "( ... )" starting at the "(": a list of nodes, "For(...)( ... )" keeps its block.
    """
    commands = []
    position += 1
    while position < len(tokens) and tokens[position] != ')':
        if tokens[position] in (',', '='):
            position += 1
            continue
        node, position = parse_expression(tokens, position)
        if node[0] == 'call' and position < len(tokens) and tokens[position] == '(':
            block, position = parse_block(tokens, position)
            node = ('call', node[1], node[2], block)
        commands.append(node)
    return commands, position + 1


def skip_balanced(tokens, position):
    depth = 0
    while position < len(tokens):
        if tokens[position] == '(':
            depth += 1
        elif tokens[position] == ')':
            depth -= 1
            if depth == 0:
                return position + 1
        position += 1
    return position


def parse_program(text):
    """
    name -> [(conditions, commands)] of the scripts of a SS_TEXT, the world script under "WorldScript".
    The parser skips what it does not know, so it reads the scripts of both games.
    """
    tokens = tokenize(text)
    scripts = {}
    position = 0
    while position < len(tokens):
        token = tokens[position]
        if token in ('GlobalVars', 'DeclareScript'):
            while position < len(tokens) and tokens[position] != '(':
                position += 1
            position = skip_balanced(tokens, position)
        elif token == 'Script' and position + 2 < len(tokens):
            name = tokens[position + 1]
            position += 2
            if tokens[position] != '(':
                continue
            branches = []
            position += 1
            while position < len(tokens) and tokens[position] != ')':
                if tokens[position] == 'if' and tokens[position + 1] == '(':
                    conditions, position = parse_block(tokens, position + 1)
                    if position < len(tokens) and tokens[position] == 'then':
                        commands, position = parse_block(tokens, position + 1)
                    else:
                        commands = []
                    branches.append((conditions, commands))
                else:
                    position += 1
            position += 1
            scripts[name] = branches
        elif token == 'WorldScript' and position + 1 < len(tokens):
            commands, position = parse_block(tokens, position + 1)
            scripts['WorldScript'] = [([], commands)]
        else:
            position += 1
    return scripts


def render(node):
    kind = node[0]
    if kind == 'str':
        return '"%s"' % node[1]
    if kind in ('num', 'id'):
        return node[1]
    return '%s(%s)' % (node[1], ', '.join(render(a) for a in node[2]))


def walk_calls(nodes):
    for node in nodes:
        if node[0] != 'call':
            continue
        yield node
        for found in walk_calls(node[2]):
            yield found
        if node[3]:
            for found in walk_calls(node[3]):
                yield found


def variable_name(node):
    if node[0] == 'str':
        return node[1].strip().lower()
    return None


def condition_tests(conditions):
    """
    [(variable, operator, value)] of the tests of the variables of the scenario in the conditions of a branch.
    """
    tests = []
    for condition in conditions:
        negate = False
        node = condition
        while node[0] == 'call' and node[1] == 'Not' and node[2]:
            negate = not negate
            node = node[2][0]
        if node[0] != 'call' or len(node[2]) != 2:
            for call in walk_calls([condition]):
                if call[1] in ('GSGetVar', 'GetGSVar') and len(call[2]) >= 2:
                    name = variable_name(call[2][1])
                    if name and VAR_RE.match(name):
                        tests.append((name, 'uses', ''))
            continue
        left, right = node[2]
        if node[1] == 'IsEqualString' and left == ('id', 'szComplete'):
            # EI2: the branch of "#OnBriefingComplete" which runs after the briefing was heard
            name = variable_name(right)
            if name:
                tests.append((name, 'not ==' if negate else '==', '2'))
            continue
        operator = {'IsEqual': '==', 'IsGreater': '>', 'IsLess': '<', 'IsGreaterOrEqual': '>=',
                    'IsLessOrEqual': '<='}.get(node[1], node[1])
        if left[0] == 'call' and left[1] in ('GSGetVar', 'GetGSVar') and len(left[2]) >= 2:
            name = variable_name(left[2][1])
            if name:
                if negate:
                    operator = 'not ' + operator
                tests.append((name, operator, render(right)))
    return tests


def command_effects(commands, script_names):
    """
    The assignments [(variable, value)], the calls of other scripts and the notable commands of a branch.
    """
    assignments = []
    calls = []
    notable = []
    for call in walk_calls(commands):
        name = call[1]
        if name in ('GSSetVar', 'GSSetVarMax') and len(call[2]) >= 3:
            variable = variable_name(call[2][1])
            if variable:
                assignments.append((variable, render(call[2][2])))
        elif name == 'GSDelVar' and len(call[2]) >= 2:
            variable = variable_name(call[2][1])
            if variable:
                assignments.append((variable, 'del'))
        elif name in script_names:
            calls.append(name)
        elif name in NOTABLE_COMMANDS:
            notable.append(render(call))
    return assignments, calls, notable


def unit_conditions(conditions):
    found = []
    for call in walk_calls(conditions):
        if call[1] in UNIT_CONDITIONS:
            found.append(render(call))
    return found

#------------------------------------------------------------------------------


def analyze_program(map_name, text):
    """
    Events of the scripts of one map: every branch which sets a variable of the scenario or does a notable
    command, with its conditions and the conditions of the scripts which call it.
    """
    scripts = parse_program(text)
    script_names = set(scripts)
    callers = collections.defaultdict(list)
    branch_info = {}
    for name, branches in scripts.items():
        infos = []
        for conditions, commands in branches:
            assignments, calls, notable = command_effects(commands, script_names)
            infos.append({
                'tests': condition_tests(conditions),
                'units': unit_conditions(conditions),
                'conditions': [render(c) for c in conditions],
                'assignments': assignments,
                'notable': notable,
            })
            for callee in calls:
                callers[callee].append((name, len(infos) - 1))
        branch_info[name] = infos
    events = []
    for name, infos in branch_info.items():
        for index, info in enumerate(infos):
            if not info['assignments'] and not info['notable']:
                continue
            context = inherited_context(name, callers, branch_info, set())
            events.append({
                'map': map_name,
                'script': name,
                'tests': info['tests'] + context['tests'],
                'units': info['units'] + context['units'],
                'conditions': info['conditions'] + context['conditions'],
                'assignments': info['assignments'],
                'notable': info['notable'],
            })
    return events


def inherited_context(name, callers, branch_info, seen):
    """
    The conditions of the branches which call the script, and of their callers.
    """
    context = {'tests': [], 'units': [], 'conditions': []}
    if name in seen:
        return context
    seen.add(name)
    for caller, index in callers.get(name, ()):
        info = branch_info[caller][index]
        context['tests'].extend(info['tests'])
        context['units'].extend(info['units'])
        context['conditions'].extend(info['conditions'])
        upper = inherited_context(caller, callers, branch_info, seen)
        for key in context:
            context[key].extend(upper[key])
    return context


def after_briefing_events(map_name, section):
    """
    EI1: the commands of VSS_BS_COMMANDS which run after the briefing of VSS_BS_CHECK was heard.
    """
    checks = section.get('VSS_BS_CHECK') or ()
    commands = section.get('VSS_BS_COMMANDS') or ()
    events = []
    for variable, text in zip(checks, commands):
        nodes = parse_block(tokenize('(' + text + ')'), 0)[0]
        assignments, _calls, notable = command_effects(nodes, set())
        if not assignments and not notable:
            continue
        events.append({
            'map': map_name,
            'script': 'after briefing',
            'tests': [(variable.lower(), '==', '2')],
            'units': [],
            'conditions': ['heard %s' % variable],
            'assignments': assignments,
            'notable': notable,
        })
    return events

#------------------------------------------------------------------------------


def decode_stats(dwords):
    if not dwords or len(dwords) < 41:
        return {}
    values = {}
    for index, name in STAT_FIELDS:
        raw = struct.pack('<I', dwords[index] & 0xffffffff)
        if index in INTEGER_STATS:
            values[name] = struct.unpack('<i', raw)[0]
        else:
            values[name] = round(struct.unpack('<f', raw)[0], 2)
    return values


def read_units(map_name, root):
    units = []
    for unit in root.walk('UNIT'):
        values = {c.name: c.value for c in unit.children if c.children is None}
        name = values.get('OBJNAME') or ''
        nid = values.get('NID', 0)
        # an EI2 unit can be named by its NID alone
        if not name and not 1000000000 <= nid <= 1999999999:
            continue
        position = values.get('OBJPOSITION') or (0.0, 0.0, 0.0)
        units.append({
            'map': map_name,
            'name': name,
            'nid': nid,
            'prototype': values.get('UNIT_PROTOTYPE') or values.get('PARENTTEMPLATE') or '',
            'model': (values.get('OBJTEMPLATE') or '').lower(),
            'texture': values.get('OBJPRIMTXTR') or '',
            'second_texture': values.get('OBJSECTXTR') or '',
            'complection': [round(v, 3) for v in (values.get('OBJCOMPLECTION') or (0.5, 0.5, 0.5))],
            'armors': list(values.get('UNIT_ARMORS') or ()),
            'weapons': list(values.get('UNIT_WEAPONS') or ()),
            'spells': list(values.get('UNIT_SPELLS') or ()),
            'quick_items': list(values.get('UNIT_QUICK_ITEMS') or ()),
            'quest_items': list(values.get('UNIT_QUEST_ITEMS') or ()),
            'player': values.get('OBJ_PLAYER', 0),
            'position': [round(v, 1) for v in position],
            'stats': decode_stats(values.get('UNIT_STATS')),
        })
    return units


def read_maps(maps_dir_path):
    """
    The units with names and the events of the scripts of all .mob files.
    """
    units = []
    events = []
    for file_name in sorted(os.listdir(maps_dir_path)):
        if not file_name.lower().endswith('.mob'):
            continue
        map_name = file_name[:-4].lower()
        with open(os.path.join(maps_dir_path, file_name), 'rb') as f:
            data = f.read()
        try:
            root = mob.parse(data)
        except Exception as exc:
            if _Debug:
                print('%s: %s' % (file_name, exc))
            continue
        units.extend(read_units(map_name, root))
        for node in root.walk('SS_TEXT'):
            events.extend(analyze_program(map_name, node.value or ''))
        for section in root.walk('VSS_SECTION'):
            values = {c.name: c.value for c in section.children if c.children is None}
            events.extend(after_briefing_events(map_name, values))
    return units, events

#------------------------------------------------------------------------------


def read_zones(texts):
    result = []
    for zone in zones.parse(texts.archive.read('map.txt')):
        result.append({
            'id': zone.name,
            'allod': zone.allod,
            'type': zone.type,
            'mob': (zone.mob_name or '').lower(),
            'mpr': (zone.mpr_name or '').lower(),
            'title': texts.first_line('zone ' + zone.name),
            'text': texts.get('zone ' + zone.name).strip().split('\n', 1)[-1].strip(),
            'exits': [e.neighbour_name for e in zone.exits if e.neighbour_name and e.neighbour_name != 'none'],
        })
    return result


def zone_of_variable(name):
    parts = name.split('.')
    return parts[1] if len(parts) > 2 else ''


def is_briefing_map(map_name):
    return map_name.startswith(('bz', 'cz'))


def prototype_record(prototype, unit_database):
    """
    The numbers of a prototype of units.udb shown for a character.
    """
    unit_type = unit_database.unit_type(prototype.type_name)
    return {
        'name': prototype.name,
        'type': prototype.type_name,
        'model': unit_type.model_name if unit_type is not None else '',
        'health': int(round(prototype.health)),
        'mana': int(round(prototype.mana)),
        'attack': round(prototype.attack, 1),
        'parry': round(prototype.defense, 1),
        'damage': '%d-%d' % (round(prototype.damage_min), round(prototype.damage_min + prototype.damage_max)),
        'armor': round(prototype.armor, 1),
        'skin': prototype.skin,
        'hair': prototype.hair,
        'weapons': list(prototype.weapons),
    }


def find_unit(character_id, units_by_name, units_by_prototype, unit_database, units_by_script_name):
    """
    The unit of a map which shows the character: by the hash of its name in the NID (EI2), by the hints of
    CHARACTER_PROTOTYPES, then by its OBJNAME, the units of the briefing zones first.
    A unit made from the prototype alone when no map has one.
    """
    found = units_by_script_name.get(character_id, [])
    if found:
        counts = collections.Counter(u['prototype'] for u in found)
        found = sorted(found, key=lambda u: (not is_briefing_map(u['map']), -counts[u['prototype']], u['map']))
        return found[0], 'nid'
    for hint in CHARACTER_PROTOTYPES.get(character_id, ()):
        found = [u for prototype, us in units_by_prototype.items() if hint in prototype for u in us]
        if found:
            found.sort(key=lambda u: (not is_briefing_map(u['map']), u['map']))
            return found[0], 'prototype'
        prototype = unit_database.prototype(hint)
        if prototype is not None:
            unit_type = unit_database.unit_type(prototype.type_name)
            return {
                'map': '', 'name': character_id, 'prototype': prototype.name,
                'model': unit_type.model_name if unit_type is not None else '',
                'texture': 'default0', 'second_texture': 'default0', 'complection': [0.5, 0.5, 0.5],
                'armors': [], 'weapons': list(prototype.weapons), 'spells': [], 'quick_items': [],
                'quest_items': [], 'player': 0, 'position': [0, 0, 0], 'stats': {},
            }, 'prototype only'
    found = units_by_name.get(character_id, [])
    if not found:
        return None, ''
    counts = collections.Counter(u['prototype'] for u in found)
    found = sorted(found, key=lambda u: (not is_briefing_map(u['map']), -counts[u['prototype']], u['map']))
    return found[0], 'name'


def quest_targets(events, units_by_map):
    """
    Named units in the conditions of the branches which move the quests on: quest -> {unit name: unit}.
    """
    targets = collections.defaultdict(dict)
    for event in events:
        quests = set()
        for variable, _value in event['assignments']:
            parts = variable.split('.')
            if parts[0] == 'q' and len(parts) >= 3:
                quests.add(parts[2])
        if not quests:
            continue
        for condition in event['units']:
            names = re.findall(r'[\(,]\s*([A-Za-z_][\w#]*)', condition)
            names += re.findall(r'"([^"]+)"', condition)
            names += ['#%s' % nid for nid in re.findall(r'GetObject\((\d+)\)', condition)]
            for name in names:
                unit = units_by_map.get(event['map'], {}).get(name.lower())
                if unit is None:
                    continue
                for quest in quests:
                    targets[quest][unit.get('script_name') or unit['name']] = unit
    return targets


def get_character(characters, persons, character_id):
    if character_id not in characters:
        characters[character_id] = {
            'id': character_id, 'name': persons.get(character_id, ''), 'kind': 'npc',
            'owns': [], 'speaks_in': [], 'lines': 0, 'quests_given': [], 'quests_target': [],
            'with': collections.Counter(), 'money': 0.0, 'items': [], 'brief_zones': set(),
        }
    return characters[character_id]


def assign_script_names(units, persons, briefings, events):
    """
    Sets 'script_name' of the units whose NID is the hash of a name known from the texts, the briefings
    or the scripts.
    """
    candidates = set(persons)
    for briefing in briefings:
        candidates.update(briefing['owners'])
        candidates.update(briefing['speakers'])
        candidates.update(briefing['cast'])
    for event in events:
        for condition in event['conditions'] + event['notable']:
            candidates.update(re.findall(r'[A-Za-z_][\w]*', condition))
            candidates.update(re.findall(r'"([^"]+)"', condition))
    by_hash = {}
    for name in sorted(candidates):
        # the first spelling wins, the texts of the persons spell the names as the briefings
        by_hash.setdefault(name_hash(name), name)
    for name in persons:
        by_hash[name_hash(name)] = name
    for unit in units:
        name = by_hash.get(unit['nid'])
        if name:
            unit['script_name'] = name


def build_characters(briefings, quests, persons, units, events, zone_ids, unit_database, unit_names):
    assign_script_names(units, persons, briefings, events)
    persons = {k.lower(): v for k, v in persons.items()}
    units_by_script_name = collections.defaultdict(list)
    for unit in units:
        if unit.get('script_name'):
            units_by_script_name[unit['script_name'].lower()].append(unit)
    units_by_name = collections.defaultdict(list)
    units_by_prototype = collections.defaultdict(list)
    units_by_map = collections.defaultdict(dict)
    for unit in units:
        units_by_name[unit['name'].lower()].append(unit)
        units_by_prototype[unit['prototype'].lower()].append(unit)
        units_by_map[unit['map']].setdefault(unit['name'].lower(), unit)
        units_by_map[unit['map']].setdefault('#%s' % unit['nid'], unit)
        if unit.get('script_name'):
            units_by_map[unit['map']][unit['script_name'].lower()] = unit
    quest_by_variable = {q['variable']: q for q in quests if q['variable']}
    characters = {}

    for briefing in briefings:
        owners = [o for o in briefing['owners'] if o not in zone_ids]
        cast = []
        for speaker in briefing['speakers'] + briefing['cast']:
            key = speaker.lower()
            if key in NOT_CHARACTERS or key in cast:
                continue
            if key in persons or key in units_by_name or key in units_by_script_name or key in owners:
                cast.append(key)
        for owner in owners:
            record = get_character(characters, persons, owner)
            record['owns'].append(briefing['id'])
            record['money'] += briefing['money']
            record['items'].extend(briefing['items'] + briefing['quest_items'])
            for variable in briefing['quests_open']:
                quest = quest_by_variable.get(variable.lower())
                if quest is not None and quest['id'] not in record['quests_given']:
                    record['quests_given'].append(quest['id'])
        for key in cast:
            record = get_character(characters, persons, key)
            if briefing['id'] not in record['speaks_in']:
                record['speaks_in'].append(briefing['id'])
            record['lines'] += sum(1 for s, _t, _d in briefing['phrases'] if s.lower() == key)
            for other in cast:
                if other != key:
                    record['with'][other] += 1
    for quest_id, found in quest_targets(events, units_by_map).items():
        for name, unit in found.items():
            key = name.lower()
            if key in NOT_CHARACTERS:
                continue
            record = get_character(characters, persons, key)
            if key not in persons and not record['owns'] and not record['speaks_in']:
                record['kind'] = 'target'
                record['name'] = record['name'] or unit_names.get(unit['prototype'].lower(), '') or name
            if quest_id not in record['quests_target']:
                record['quests_target'].append(quest_id)
    for key, record in characters.items():
        if key in ('hero', 'hero2'):
            record['kind'] = 'hero'
        unit, how = find_unit(key, units_by_name, units_by_prototype, unit_database, units_by_script_name)
        record['unit'] = unit
        record['unit_found_by'] = how
        prototype = unit_database.prototype(unit['prototype']) if unit else None
        record['prototype'] = prototype_record(prototype, unit_database) if prototype is not None else None
        record['prototype_name'] = unit_names.get(unit['prototype'].lower(), '') if unit else ''
        record['maps'] = sorted(set(u['map'] for u in units_by_name.get(key, []) + units_by_script_name.get(key, [])))
        record['with'] = dict(record['with'].most_common())
        record['money'] = round(record['money'], 1)
        record['brief_zones'] = sorted(m for m in record['maps'] if is_briefing_map(m))
    return sorted(characters.values(), key=lambda c: (c['kind'] != 'hero', c['kind'] == 'target',
                                                      -(len(c['owns']) * 3 + c['lines'])))


def build(data_dir_path, texts_file_path=None):
    database = res.ResArchive.open(gamedata.resolve_path(data_dir_path, 'res/database.res'))
    texts_path = gamedata.resolve_path(data_dir_path, 'res/texts.res')
    texts = Texts(res.ResArchive.open(texts_file_path or texts_path))
    original_texts = Texts(res.ResArchive.open(texts_path))
    quests, briefings = read_quest_database(database.read('quests.qdb'))
    zone_list = read_zones(original_texts)
    for zone in zone_list:
        zone['title'] = texts.first_line('zone ' + zone['id']) or zone['title']
        translated = texts.get('zone ' + zone['id']).strip()
        if translated:
            zone['text'] = translated.split('\n', 1)[-1].strip()
    mob_zones = collections.defaultdict(list)
    for zone in zone_list:
        if zone['mob']:
            mob_zones[zone['mob']].append(zone['id'])
    units, events = read_maps(gamedata.resolve_path(data_dir_path, 'maps'))
    # quests
    quest_by_id = {}
    for quest in quests:
        title, description, subobjectives = parse_quest_text(texts.get('quest ' + quest['id']))
        quest.update(title=title, description=description, subobjectives=subobjectives,
                     variable='', zone='', given_by=[], done_by=[])
        quest_by_id[quest['id'].lower()] = quest
    # briefings
    owners = collections.defaultdict(set)
    for event in events:
        for variable, _value in event['assignments']:
            match = VAR_RE.match(variable)
            if match and match.group(1) == 'b' and '.' in match.group(2):
                owner, brief = match.group(2).split('.', 1)
                owners[brief].add(owner)
        for variable, _operator, _value in event['tests']:
            match = VAR_RE.match(variable)
            if match and match.group(1) == 'b' and '.' in match.group(2):
                owner, brief = match.group(2).split('.', 1)
                owners[brief].add(owner)
    for briefing in briefings:
        title, cast, phrases = parse_briefing_text(texts.get('briefing ' + briefing['id']))
        speakers = []
        for speaker, _text, _desc in phrases:
            if speaker not in speakers:
                speakers.append(speaker)
        briefing.update(title=title, cast=cast, speakers=speakers, phrases=phrases,
                        owners=sorted(owners.get(briefing['id'].lower(), ())))
        for variable in briefing['quests_open']:
            quest = quest_by_id.get(variable.split('.')[-1].lower()) if variable.count('.') == 2 else None
            if quest is not None:
                quest['variable'] = variable.lower()
                quest['zone'] = zone_of_variable(variable.lower())
                quest['given_by'].append(briefing['id'])
        for variable in briefing['quests_done']:
            quest = quest_by_id.get(variable.split('.')[-1].lower()) if variable.count('.') == 2 else None
            if quest is not None:
                quest['done_by'].append(briefing['id'])
    # the quests which scripts give or finish
    for event in events:
        for variable, value in event['assignments']:
            parts = variable.split('.')
            if parts[0] == 'q' and len(parts) == 3:
                quest = quest_by_id.get(parts[2])
                if quest is not None and not quest['variable']:
                    quest['variable'] = variable
                    quest['zone'] = parts[1]
    persons = {name.split(' ', 1)[1]: texts.first_line(name) for name in texts.names('pers')}
    unit_names = {name.split(' ', 1)[1].replace('_', ' ').lower(): texts.first_line(name)
                  for name in texts.names('UNIT')}
    unit_database = idb.parse_units(database.read('units.udb'))
    characters = build_characters(briefings, quests, persons, units, events, set(z['id'] for z in zone_list),
                                  unit_database, unit_names)
    return {
        'characters': characters,
        'quests': sorted(quests, key=lambda q: q['index']),
        'briefings': briefings,
        'zones': zone_list,
        'mob_zones': mob_zones,
        'allods': {name.split(' ', 1)[1]: {'title': texts.first_line(name),
                                           'text': texts.get(name).strip().split('\n', 1)[-1].strip()}
                   for name in texts.names('allod')},
        'persons': persons,
        'unit_names': unit_names,
        'quest_items': {name.split(' ', 1)[1].replace('_', ' ').lower(): texts.first_line(name)
                        for name in texts.names('QUESTITEM')},
        'units': units,
        'events': events,
    }

#------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
    p.add_argument('data_dir_path', metavar='DATA_DIR')
    p.add_argument('-o', dest='output_file_path', metavar='OUTPUT', required=True)
    p.add_argument('--texts', dest='texts_file_path', metavar='TEXTS_RES', default=None)
    args = p.parse_args()
    result = build(args.data_dir_path, args.texts_file_path)
    with open(args.output_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    if _Debug:
        print('%d quests, %d briefings, %d units, %d events' % (
            len(result['quests']), len(result['briefings']), len(result['units']), len(result['events'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
