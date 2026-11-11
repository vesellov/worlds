"""
Objects of a map read from its .mob file: trees, buildings, levers and units.

OBJPOSITION is x, y and the height above the land, OBJROTATION is a quaternion (w, x, y, z),
OBJCOMPLECTION are the body proportions of the figure (the coefficients of fig.trilinear(), they can be above 1).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

#------------------------------------------------------------------------------

from engine import quat

#------------------------------------------------------------------------------

OBJECT_NODES = ('OBJECT', 'UNIT', 'LEVER', 'TORCH', 'MAGIC_TRAP')

DEFAULT_HOUR = 12.0

#------------------------------------------------------------------------------


class WorldSettings(object):
    """
    WORLD_SET of a .mob: the hour of the day, the wind and two light coefficients.
    """

    def __init__(self, node=None):
        get = node.get if node is not None else (lambda name, default=None: default)
        self.hour = float(get('WS_TIME', DEFAULT_HOUR) or DEFAULT_HOUR) % 24.0
        self.wind_direction = tuple(get('WS_WIND_DIR', (0.0, 1.0, 0.0)) or (0.0, 1.0, 0.0))
        self.wind_strength = float(get('WS_WIND_STR', 0.0) or 0.0)
        self.ambient = float(get('WS_AMBIENT', 0.2) or 0.0)
        self.sun_light = float(get('WS_SUN_LIGHT', 0.6) or 0.0)

    def __repr__(self):
        return 'WorldSettings(%.2f h, wind %.2f)' % (self.hour, self.wind_strength)


class MapObject(object):

    def __init__(self, node):
        self.kind = node.name
        self.nid = node.get('NID')
        self.name = node.get('OBJNAME', '')
        self.template = (node.get('OBJTEMPLATE') or '').lower()
        self.parent_template = node.get('PARENTTEMPLATE', '')
        self.texture = (node.get('OBJPRIMTXTR') or '').lower()
        self.second_texture = (node.get('OBJSECTXTR') or '').lower()
        self.position = tuple(node.get('OBJPOSITION') or (0.0, 0.0, 0.0))
        self.rotation = tuple(node.get('OBJROTATION') or (1.0, 0.0, 0.0, 0.0))
        self.complection = tuple(node.get('OBJCOMPLECTION') or (0.5, 0.5, 0.5))
        self.player = node.get('OBJ_PLAYER', 0)
        self.object_type = node.get('OBJTYPE')
        self.prototype = node.get('UNIT_PROTOTYPE')
        self.armors = node.get('UNIT_ARMORS') or ()
        self.weapons = node.get('UNIT_WEAPONS') or ()
        # the names of the quest items of section 5 of items.idb which the unit carries
        self.quest_items = node.get('UNIT_QUEST_ITEMS') or ()
        # the 43 dwords of engine.stats
        self.stats = node.get('UNIT_STATS')
        # the records UNIT_LOGIC of the alarms 0..4 (engine.ai.UnitLogic.from_node)
        self.logic_nodes = [c for c in (node.children or []) if c.name == 'UNIT_LOGIC']
        self.logic_node = self.logic_nodes[0] if self.logic_nodes else None
        # LEVER: the state, the number of the states, the states go round, it is a door, the grid is built again
        self.lever_state = node.get('LEVER_CUR_STATE', 0)
        self.lever_states = node.get('LEVER_TOTAL_STATE', 0)
        self.lever_cycled = bool(node.get('LEVER_IS_CYCLED', 0))
        self.lever_is_door = bool(node.get('LEVER_IS_DOOR', 0))
        self.lever_recalc_graph = bool(node.get('LEVER_RECALC_GRAPH', 0))
        # LEVER_SCIENCE_STATS_NEW: (the flags, the number of the key, the level) which the user must have
        self.lever_science = tuple(node.get('LEVER_SCIENCE_STATS_NEW') or (0, 0, 0))
        # MAGIC_TRAP: the bits of the players it casts at, the spell "name{modifier;...}", the ticks between the casts
        # (below 1: at every unit which comes in), one target only (LEVER_CAST_ONCE of EIrepack, +0xcc),
        # the circles (x, y, radius) which it looks at and the points of the spells of the point
        self.trap_players = node.get('MT_DIPLOMACY', 0xffffffff)
        self.trap_spell = node.get('MT_SPELL', '')
        self.trap_interval = node.get('MT_CAST_INTERVAL', 15)
        self.trap_one_target = bool(node.get('LEVER_CAST_ONCE', 0))
        self.trap_areas = tuple(node.get('MT_AREAS') or ())
        self.trap_targets = tuple(node.get('MT_TARGETS') or ())

    def __repr__(self):
        return 'MapObject(%s %s %s at %.1f,%.1f)' % (self.kind, self.template, self.texture, self.position[0], self.position[1])

    @property
    def is_unit(self):
        return self.kind == 'UNIT'

    @property
    def is_lever(self):
        return self.kind == 'LEVER'

    @property
    def is_trap(self):
        return self.kind == 'MAGIC_TRAP'

    @property
    def yaw(self):
        w, _, _, z = self.rotation
        return 2.0 * math.atan2(z, w)

    def rotation_matrix(self):
        return quat.to_matrix(self.rotation)

#------------------------------------------------------------------------------


def map_objects(mob_root):
    found = []
    for name in OBJECT_NODES:
        for node in mob_root.walk(name):
            if node.children is not None and node.find('OBJTEMPLATE') is not None:
                found.append(MapObject(node))
    return found


def world_settings(mob_root):
    if mob_root is not None:
        for node in mob_root.walk('WORLD_SET'):
            if node.children is not None:
                return WorldSettings(node)
    return WorldSettings()
