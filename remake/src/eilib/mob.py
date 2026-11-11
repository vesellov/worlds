"""
Reader of the .mob files: the objects of a map (units, buildings, trees, levers, traps...),
the scripts and the diplomacy. A .mob file is a tree of tagged nodes: a 4 byte tag, a 4 byte length
(with these 8 bytes) and the data, a record node holds other nodes.

The table of the tags is taken from EIrepack (Formats/mob.py, MIT license, (c) 2018 Asp Adm).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

#------------------------------------------------------------------------------

MAGIC = b'\x00\xA0\x00\x00'

ENCODING = 'cp1251'

# tag -> (name, kind of the data)
TAGS = {
    4294967295: ('UNKNOWN', 'Unknown'),
    0: ('ROOT', 'Record'),
    43984: ('WORLD_SET', 'Record'),
    45072: ('OBJ_DEF_LOGIC', 'Null'),
    53248: ('PR_OBJECTDBFILE', 'Null'),
    57346: ('DIR_NAME', 'String'),
    3722304977: ('DIPLOMATION', 'Record'),
    43985: ('WS_WIND_DIR', 'Plot'),
    45056: ('OBJECTSECTION', 'Record'),
    45066: ('OBJROTATION', 'Quaternion'),
    45073: ('OBJ_PLAYER', 'Byte'),
    57347: ('DIR_NINST', 'Dword'),
    3722304978: ('DIPLOMATION_FOF', 'Diplomacy'),
    40960: ('OBJECTDBFILE', 'Record'),
    43520: ('LIGHT_SECTION', 'Null'),
    43986: ('WS_WIND_STR', 'Float'),
    45057: ('OBJECT', 'Record'),
    45067: ('OBJTEXTURE', 'Null'),
    45074: ('OBJ_PARENT_ID', 'Dword'),
    52224: ('SOUND_SECTION', 'Null'),
    52234: ('SOUND_RESNAME', 'StringArray'),
    56576: ('PARTICL_SECTION', 'Null'),
    57348: ('DIR_PARENT_FOLDER', 'Dword'),
    65280: ('SEC_RANGE', 'Record'),
    3722304979: ('DIPLOMATION_PL_NAMES', 'StringArray'),
    43521: ('LIGHT', 'Record'),
    43987: ('WS_TIME', 'Float'),
    45058: ('NID', 'Dword'),
    45068: ('OBJCOMPLECTION', 'Plot'),
    45075: ('OBJ_USE_IN_SCRIPT', 'Byte'),
    52225: ('SOUND', 'Record'),
    52235: ('SOUND_RANGE2', 'Dword'),
    56577: ('PARTICL', 'Record'),
    57349: ('DIR_TYPE', 'Byte'),
    65281: ('MAIN_RANGE', 'Record'),
    7696: ('VSS_BS_COMMANDS', 'StringArray'),
    43522: ('LIGHT_RANGE', 'Float'),
    43988: ('WS_AMBIENT', 'Float'),
    45059: ('OBJTYPE', 'Dword'),
    45069: ('OBJBODYPARTS', 'StringArray'),
    45076: ('OBJ_IS_SHADOW', 'Byte'),
    52226: ('SOUND_ID', 'Dword'),
    56578: ('PARTICL_ID', 'Dword'),
    65282: ('RANGE', 'Record'),
    7680: ('VSS_SECTION', 'Record'),
    7690: ('VSS_ISSTART', 'Byte'),
    7697: ('VSS_CUSTOM_SRIPT', 'String'),
    43523: ('LIGHT_NAME', 'String'),
    43989: ('WS_SUN_LIGHT', 'Float'),
    45060: ('OBJNAME', 'String'),
    45070: ('PARENTTEMPLATE', 'String'),
    45077: ('OBJ_R', 'Null'),
    52227: ('SOUND_POSITION', 'Plot'),
    52237: ('SOUND_AMBIENT', 'Byte'),
    56579: ('PARTICL_POSITION', 'Plot'),
    3149594624: ('UNIT', 'Record'),
    3149594634: ('UNIT_NEED_IMPORT', 'Byte'),
    7681: ('VSS_TRIGER', 'Record'),
    7691: ('VSS_LINK', 'Record'),
    43524: ('LIGHT_POSITION', 'Plot'),
    45061: ('OBJINDEX', 'Null'),
    45071: ('OBJCOMMENTS', 'String'),
    45078: ('OBJ_QUEST_INFO', 'String'),
    52228: ('SOUND_RANGE', 'Dword'),
    52238: ('SOUND_IS_MUSIC', 'Byte'),
    56580: ('PARTICL_COMMENTS', 'String'),
    3148546048: ('MAGIC_TRAP', 'Record'),
    3149594625: ('UNIT_R', 'Null'),
    3149660160: ('UNIT_LOGIC', 'Record'),
    3149660170: ('UNIT_LOGIC_WAIT', 'Float'),
    7682: ('VSS_CHECK', 'Record'),
    7692: ('VSS_GROUP', 'String'),
    43525: ('LIGHT_ID', 'Dword'),
    45062: ('OBJTEMPLATE', 'String'),
    52229: ('SOUND_NAME', 'String'),
    56581: ('PARTICL_NAME', 'String'),
    65285: ('MIN_ID', 'Dword'),
    3148546049: ('MT_DIPLOMACY', 'Dword'),
    3148611584: ('LEVER', 'Record'),
    3149594626: ('UNIT_PROTOTYPE', 'String'),
    3149660161: ('UNIT_LOGIC_AGRESSIV', 'Null'),
    3149660171: ('UNIT_LOGIC_ALARM_CONDITION', 'Byte'),
    3149725696: ('GUARD_PT', 'Record'),
    7683: ('VSS_PATH', 'Record'),
    7693: ('VSS_IS_USE_GROUP', 'Byte'),
    43526: ('LIGHT_SHADOW', 'Byte'),
    45063: ('OBJPRIMTXTR', 'String'),
    52230: ('SOUND_MIN', 'Dword'),
    56582: ('PARTICL_TYPE', 'Dword'),
    65286: ('MAX_ID', 'Dword'),
    3148546050: ('MT_SPELL', 'String'),
    3148611585: ('LEVER_SCIENCE_STATS', 'Null'),
    3149594627: ('UNIT_ITEMS', 'Null'),
    3149660162: ('UNIT_LOGIC_CYCLIC', 'Byte'),
    3149660172: ('UNIT_LOGIC_HELP', 'Float'),
    3149725697: ('GUARD_PT_POSITION', 'Plot'),
    3149791232: ('ACTION_PT', 'Record'),
    7684: ('VSS_ID', 'Dword'),
    7694: ('VSS_VARIABLE', 'Record'),
    43527: ('LIGHT_COLOR', 'Plot'),
    45064: ('OBJSECTXTR', 'String'),
    52231: ('SOUND_MAX', 'Dword'),
    56583: ('PARTICL_SCALE', 'Float'),
    826366246: ('AIGRAPH', 'AiGraph'),
    3148546051: ('MT_AREAS', 'AreaArray'),
    3148546052: ('MT_TARGETS', 'PointArray'),
    3148611586: ('LEVER_CUR_STATE', 'Byte'),
    3149594628: ('UNIT_STATS', 'UnitStats'),
    3149660163: ('UNIT_LOGIC_MODEL', 'Dword'),
    3149660173: ('UNIT_LOGIC_ALWAYS_ACTIVE', 'Byte'),
    3149725698: ('GUARD_PT_ACTION', 'Null'),
    3149791233: ('ACTION_PT_LOOK_PT', 'Plot'),
    3149856768: ('TORCH', 'Record'),
    7685: ('VSS_RECT', 'Rectangle'),
    7695: ('VSS_BS_CHECK', 'StringArray'),
    43528: ('LIGHT_COMMENTS', 'String'),
    45065: ('OBJPOSITION', 'Plot'),
    52232: ('SOUND_COMMENTS', 'String'),
    3148611587: ('LEVER_TOTAL_STATE', 'Byte'),
    3149594629: ('UNIT_QUEST_ITEMS', 'StringArray'),
    3149660164: ('UNIT_LOGIC_GUARD_R', 'Float'),
    3149660174: ('UNIT_LOGIC_AGRESSION_MODE', 'Byte'),
    3149791234: ('ACTION_PT_WAIT_SEG', 'Dword'),
    3149856769: ('TORCH_STRENGHT', 'Float'),
    7686: ('VSS_SRC_ID', 'Dword'),
    52233: ('SOUND_VOLUME', 'Null'),
    3148546053: ('MT_CAST_INTERVAL', 'Dword'),
    3148611588: ('LEVER_IS_CYCLED', 'Byte'),
    3149594630: ('UNIT_QUICK_ITEMS', 'StringArray'),
    3149660165: ('UNIT_LOGIC_GUARD_PT', 'Plot'),
    3149791235: ('ACTION_PT_TURN_SPEED', 'Dword'),
    3149856770: ('TORCH_PTLINK', 'Plot'),
    7687: ('VSS_DST_ID', 'Dword'),
    3148611589: ('LEVER_CAST_ONCE', 'Byte'),
    3149594631: ('UNIT_SPELLS', 'StringArray'),
    3149660166: ('UNIT_LOGIC_NALARM', 'Byte'),
    3149791236: ('ACTION_PT_FLAGS', 'Byte'),
    3149856771: ('TORCH_SOUND', 'String'),
    7688: ('VSS_TITLE', 'String'),
    3148611590: ('LEVER_SCIENCE_STATS_NEW', 'LeverStats'),
    3149594632: ('UNIT_WEAPONS', 'StringArray'),
    3149660167: ('UNIT_LOGIC_USE', 'Byte'),
    7689: ('VSS_COMMANDS', 'String'),
    61440: ('DIRICTORY_ELEMENTS', 'Record'),
    3148611591: ('LEVER_IS_DOOR', 'Byte'),
    3149594633: ('UNIT_ARMORS', 'StringArray'),
    3149660168: ('UNIT_LOGIC_REVENGE', 'Null'),
    57344: ('DIRICTORY', 'Record'),
    2899242186: ('SS_TEXT_OLD', 'String'),
    3148611592: ('LEVER_RECALC_GRAPH', 'Byte'),
    3149660169: ('UNIT_LOGIC_FEAR', 'Null'),
    49152: ('SC_OBJECTDBFILE', 'Null'),
    57345: ('FOLDER', 'Record'),
    2899242187: ('SS_TEXT', 'StringEncrypted'),
}

#------------------------------------------------------------------------------


class Node(object):

    def __init__(self, name, value=None, children=None):
        self.name = name
        self.value = value
        self.children = children

    def __repr__(self):
        if self.children is not None:
            return 'Node(%s, %d children)' % (self.name, len(self.children))
        return 'Node(%s=%r)' % (self.name, self.value)

    def find(self, name):
        for child in self.children or []:
            if child.name == name:
                return child
        return None

    def get(self, name, default=None):
        child = self.find(name)
        return default if child is None else child.value

    def walk(self, name):
        """
        Every node with that name in the whole tree.
        """
        if self.name == name:
            yield self
        for child in self.children or []:
            for found in child.walk(name):
                yield found

#------------------------------------------------------------------------------


def _text(raw):
    return bytes(raw).split(b'\0', 1)[0].decode(ENCODING, errors='replace')


def _decrypt(data, offset, length):
    key = struct.unpack_from('<I', data, offset)[0]
    out = bytearray(length)
    for i in range(length):
        key = (key + (((((key * 13) << 4) + key) << 8) - key) * 4 + 2531011) & 0xffffffff
        out[i] = data[offset + 4 + i] ^ ((key >> 16) & 255)
    return out.decode(ENCODING, errors='replace')


def _value(kind, data, offset, size):
    if kind == 'Null':
        return None
    if kind == 'Dword':
        return struct.unpack_from('<I', data, offset)[0]
    if kind == 'Byte':
        return data[offset]
    if kind == 'Float':
        return struct.unpack_from('<f', data, offset)[0]
    if kind == 'String':
        return _text(data[offset:offset + size])
    if kind in ('Quaternion', 'Rectangle'):
        return struct.unpack_from('<4f', data, offset)
    if kind == 'Plot':
        return struct.unpack_from('<3f', data, offset)
    if kind == 'StringArray':
        count = struct.unpack_from('<I', data, offset)[0]
        items = []
        p = offset + 4
        for _ in range(count):
            length = struct.unpack_from('<I', data, p + 4)[0]
            items.append(_text(data[p + 8:p + length]))
            p += length
        return tuple(items)
    if kind == 'UnitStats':
        return struct.unpack_from('<%dI' % (size // 4), data, offset)
    if kind == 'Diplomacy':
        return struct.unpack_from('<%di' % (size // 4), data, offset)
    if kind == 'StringEncrypted':
        return _decrypt(data, offset, size - 4)
    if kind == 'AreaArray':
        # MAGIC_TRAP: the circles (x, y, radius)
        count = struct.unpack_from('<I', data, offset)[0]
        return tuple(struct.unpack_from('<3f', data, offset + 4 + 12 * i) for i in range(count))
    if kind == 'PointArray':
        # MAGIC_TRAP: the points (x, y) which the spells of the point go to
        count = struct.unpack_from('<I', data, offset)[0]
        return tuple(struct.unpack_from('<2f', data, offset + 4 + 8 * i) for i in range(count))
    if kind == 'LeverStats':
        return struct.unpack_from('<3i', data, offset)
    return bytes(data[offset:offset + size])


def _node(data, offset):
    tag, length = struct.unpack_from('<II', data, offset)
    name, kind = TAGS.get(tag, ('TAG_%08X' % tag, 'Unknown'))
    start, end = offset + 8, offset + length
    if kind == 'Record':
        children = []
        p = start
        while p < end:
            child, p = _node(data, p)
            children.append(child)
        return Node(name, children=children), end
    return Node(name, value=_value(kind, data, start, length - 8)), end


def parse(data):
    data = bytes(data)
    if data[:4] != MAGIC:
        raise ValueError('not a .mob file')
    root, _ = _node(data, 0)
    return root
