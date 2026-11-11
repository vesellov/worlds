"""
The saved games of Evil Islands: a slot is a folder data/saves/<name> with info.sav, scenario.sav,
mission.sav and shot.sav.

info.sav is read completely: u32 signature 0x111, u32 number of strings, float hour of the world and that
many strings ending with 0. The first two strings are the allod and the zone; the last one of the saves
of the game is a 4 byte stamp (an Unix time) which is kept as raw bytes. The rest of the slot files:

- scenario.sav starts with u32 0xdead and u32 0x74 (the version), mission.sav with u32 0x102 and the version;
  the body follows (579b80 writes it, 579830 reads it): with a version above 0x6b a u32 checksum of the
  unpacked body (over its whole dwords: h = (h ^ w) * ((h ^ w) & 0xfc | 3)), above 0x71 a byte 1 when the body
  is packed and 0 when it is not, then the body. A body of a version up to 0x64 is never packed.
- the packed body (430960) is u32 the size of the body and a stream of LZ codes over a window of 0x400
  bytes: the bits are read from the lowest one of a byte, the bytes of the literals sit between the bytes of
  the bits. The bit 0 is a literal byte, 1 is a copy: the length and then the distance by the trees of
  copy_length and copy_distance, the copy starts distance + 1 bytes back.
- the body of scenario.sav starts with a blob (4a5250: u16 size * 2 below 30001, else u32 size * 2 + 1) of
  the nodes of CDbReader (409070: byte id, byte size * 2 or u32 size * 2 + 1, the data), the nodes of the
  party and of the scenario; the raw state (the quest log and so on) follows.
- the body of mission.sav starts with the names of the map files and the zone ("zone1.mob", "zone1.mpr",
  "gz1g") ending with 0, then the objects of the map.
- shot.sav is a compressed picture, its first u32 is 0xe47217f7.

The reader of game.exe is FUN_0057b3b0 (load) and FUN_0057a780/FUN_0057aba0 (save).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import struct

#------------------------------------------------------------------------------

INFO_NAME = 'info.sav'
SCENARIO_NAME = 'scenario.sav'
MISSION_NAME = 'mission.sav'
SHOT_NAME = 'shot.sav'
FILE_NAMES = (INFO_NAME, SCENARIO_NAME, MISSION_NAME, SHOT_NAME)

INFO_SIGNATURE = 0x111
SCENARIO_SIGNATURE = 0xdeaD
MISSION_SIGNATURE = 0x102
SHOT_SIGNATURE = 0xe47217f7

# the second dword of scenario.sav and mission.sav of the saves of the game
VERSION = 0x74
# 579830: the versions above these have a packed body, a checksum, the byte of the packing
PACKED_VERSION = 0x64
CHECKSUM_VERSION = 0x6b
PACKED_FLAG_VERSION = 0x71

# 430960: the window of the copies
WINDOW_SIZE = 0x400
# 4a5250: the size of a blob below this takes 2 bytes, else 4
SHORT_BLOB = 30001

#------------------------------------------------------------------------------


class SaveInfo(object):
    """
    What info.sav says: the hour of the world, the allod, the zone and the stamp.
    """

    def __init__(self, hour=0.0, strings=(), stamp=b''):
        self.hour = hour
        # the decoded strings after the header (allod, zone, ...)
        self.strings = tuple(strings)
        # the raw bytes of the last string when it is a stamp, not a text
        self.stamp = bytes(stamp)

    def __repr__(self):
        return 'SaveInfo(%g, %r)' % (self.hour, self.strings)

    @property
    def allod(self):
        return self.strings[0] if self.strings else ''

    @property
    def zone(self):
        return self.strings[1] if len(self.strings) > 1 else ''


class SaveHeader(object):
    """
    The signature, the version and the size of a .sav.
    """

    def __init__(self, name='', signature=0, version=0, size=0):
        self.name = name
        self.signature = signature
        self.version = version
        self.size = size

    def __repr__(self):
        return 'SaveHeader(%s, signature 0x%x, version 0x%x, %d bytes)' % (
            self.name, self.signature, self.version, self.size)

    @property
    def recognized(self):
        return self.signature in SIGNATURES.values()

#------------------------------------------------------------------------------

SIGNATURES = {
    INFO_NAME: INFO_SIGNATURE,
    SCENARIO_NAME: SCENARIO_SIGNATURE,
    MISSION_NAME: MISSION_SIGNATURE,
    SHOT_NAME: SHOT_SIGNATURE,
}

#------------------------------------------------------------------------------


def read_header(data, name=''):
    """
    (signature, version, size) of a .sav; version is 0 for info.sav and shot.sav without one.
    """
    data = bytes(data)
    if len(data) < 4:
        return SaveHeader(name, 0, 0, len(data))
    signature = struct.unpack_from('<I', data, 0)[0]
    version = struct.unpack_from('<I', data, 4)[0] if len(data) >= 8 else 0
    return SaveHeader(name, signature, version, len(data))


def read_info(data):
    """
    SaveInfo of info.sav, None when the data is not an info.sav.
    """
    data = bytes(data)
    if len(data) < 12:
        return None
    signature, count, hour = struct.unpack_from('<IIf', data, 0)
    if signature != INFO_SIGNATURE:
        return None
    strings = []
    raw = []
    offset = 12
    for _ in range(count):
        end = data.find(b'\0', offset)
        if end < 0:
            break
        chunk = data[offset:end]
        raw.append(chunk)
        strings.append(chunk.decode('cp1251', errors='replace'))
        offset = end + 1
    # the last string of the saves of the game is a 4 byte stamp which is not a text
    stamp = raw[-1] if raw and len(raw[-1]) == 4 and not _is_text(raw[-1]) else b''
    return SaveInfo(hour, strings, stamp)


def build_info(hour, strings):
    """
    The bytes of info.sav: the signature, the count, the hour and the strings ending with 0.
    """
    chunks = [struct.pack('<IIf', INFO_SIGNATURE, len(strings), float(hour))]
    for text in strings:
        chunks.append(str(text).encode('cp1251', errors='replace') + b'\0')
    return b''.join(chunks)


def _is_text(data):
    """
    True when every byte of the string is a printable character of cp1251.
    """
    return all(0x20 <= byte < 0x80 or byte >= 0xc0 for byte in bytes(data))


def read_signature(data):
    """
    The signature of a .sav, 0 for too short data.
    """
    data = bytes(data)
    return struct.unpack_from('<I', data, 0)[0] if len(data) >= 4 else 0


def is_save(data, name):
    """
    True when the data starts with the signature of that file of a save slot.
    """
    signature = SIGNATURES.get(name)
    return signature is not None and read_signature(data) == signature


class SaveError(ValueError):
    pass


class SaveBody(object):
    """
    The body of scenario.sav or mission.sav unpacked, with its checksum.
    """

    def __init__(self, signature, version, body, checksum=None, packed=False):
        self.signature = signature
        self.version = version
        self.body = bytes(body)
        self.checksum = checksum
        self.packed = packed

    def __repr__(self):
        return 'SaveBody(0x%x, version 0x%x, %d bytes%s)' % (
            self.signature, self.version, len(self.body), ', packed' if self.packed else '')


class _Bits(object):
    """
    430fe0, 431040: the bits of the packed stream from the lowest one; the bytes of the literals come from
    the same place between them.
    """

    def __init__(self, data, offset):
        self.data = data
        self.offset = offset
        self.value = 0
        self.count = 0

    def take_byte(self):
        if self.offset >= len(self.data):
            raise SaveError('the packed body ends too early')
        value = self.data[self.offset]
        self.offset += 1
        return value

    def bit(self):
        if self.count == 0:
            self.value = self.take_byte()
            self.count = 8
        result = self.value & 1
        self.value >>= 1
        self.count -= 1
        return result

    def bits(self, count):
        while self.count < count:
            self.value += self.take_byte() << self.count
            self.count += 8
        result = self.value & ((1 << count) - 1)
        self.value >>= count
        self.count -= count
        return result


def copy_length(bits):
    """
    430960: the length of a copy, 2..226.
    """
    if not bits.bit():
        return 4 if bits.bit() else 2
    if not bits.bit():
        if not bits.bit():
            return bits.bits(3) + 11 if bits.bit() else 3
        return bits.bits(2) + 7
    if not bits.bit():
        return bits.bits(6) + 35 if bits.bit() else 5
    if not bits.bit():
        return 6
    return bits.bits(7) + 99 if bits.bit() else bits.bits(4) + 19


def copy_distance(bits):
    """
    430960: the distance of a copy back from the byte before the next one, 0..5473.
    """
    if not bits.bit():
        if not bits.bit():
            if not bits.bit():
                return bits.bits(2) + 2 if bits.bit() else bits.bits(3) + 10
            if not bits.bit():
                return bits.bits(2) + 6
            return bits.bits(1) if bits.bit() else bits.bits(4) + 18
        return bits.bits(8) + 98 if bits.bit() else bits.bits(6) + 34
    return bits.bits(10) + 354 if bits.bit() else bits.bits(12) + 1378


def unpack(data, offset=0):
    """
    430960: the bytes of a packed body at the offset (u32 the size, then the codes).
    """
    data = bytes(data)
    if len(data) < offset + 4:
        raise SaveError('no size of the packed body')
    size = struct.unpack_from('<I', data, offset)[0]
    bits = _Bits(data, offset + 4)
    window = bytearray(WINDOW_SIZE)
    mask = WINDOW_SIZE - 1
    at = 0
    result = bytearray()
    while len(result) < size:
        if not bits.bit():
            value = bits.take_byte()
            result.append(value)
            window[at] = value
            at = (at + 1) & mask
            continue
        length = min(copy_length(bits), size - len(result))
        distance = copy_distance(bits)
        for _ in range(length):
            value = window[(at - distance - 1) & mask]
            result.append(value)
            window[at] = value
            at = (at + 1) & mask
    return bytes(result)


def checksum(body):
    """
    579b80, 579830: the checksum of the whole dwords of the body.
    """
    body = bytes(body)
    value = 0
    for (word, ) in struct.iter_unpack('<I', body[:len(body) // 4 * 4]):
        mixed = value ^ word
        value = (mixed * ((mixed & 0xfc) | 3)) & 0xffffffff
    return value


def read_body(data):
    """
    579830: SaveBody of scenario.sav or mission.sav; SaveError when the checksum does not match.
    """
    data = bytes(data)
    if len(data) < 8:
        raise SaveError('too short for a header')
    signature, version = struct.unpack_from('<II', data, 0)
    offset = 8
    if version <= PACKED_VERSION:
        return SaveBody(signature, version, data[offset:])
    stored = None
    if version > CHECKSUM_VERSION:
        stored = struct.unpack_from('<I', data, offset)[0]
        offset += 4
    packed = True
    if version > PACKED_FLAG_VERSION:
        packed = data[offset] != 0
        offset += 1
    body = unpack(data, offset) if packed else data[offset:]
    if packed and stored is not None and checksum(body) != stored:
        raise SaveError('the checksum does not match')
    return SaveBody(signature, version, body, stored, packed)


def read_blob(data, offset=0):
    """
    The reader of 4a5250: (the bytes of the blob, the offset after it).
    """
    if len(data) < offset + 2:
        raise SaveError('no size of a blob')
    size = struct.unpack_from('<H', data, offset)[0]
    if size & 1:
        size = struct.unpack_from('<I', data, offset)[0]
        offset += 4
    else:
        offset += 2
    size >>= 1
    if len(data) < offset + size:
        raise SaveError('the blob is longer than the data')
    return data[offset:offset + size], offset + size


def read_nodes(data):
    """
    CDbReader::NextNode 409070: [(id, the bytes of the node)] of a level of the tree; a node of nodes is read by
    read_nodes of its bytes.
    """
    data = bytes(data)
    result = []
    offset = 0
    while offset + 2 <= len(data):
        node_id = data[offset]
        size = data[offset + 1]
        if size & 1:
            if offset + 5 > len(data):
                raise SaveError('a node without its size')
            size = struct.unpack_from('<I', data, offset + 1)[0]
            offset += 5
        else:
            offset += 2
        size >>= 1
        if offset + size > len(data):
            raise SaveError('a node is longer than the data')
        result.append((node_id, data[offset:offset + size]))
        offset += size
    if offset != len(data):
        raise SaveError('a byte is left after the nodes')
    return result


def mission_files(body):
    """
    The names at the start of the body of mission.sav: (the map, the map of the textures, the zone).
    """
    names = []
    offset = 0
    for _ in range(3):
        end = body.find(b'\0', offset)
        if end < 0:
            raise SaveError('no names of the map')
        names.append(body[offset:end].decode('cp1251', errors='replace'))
        offset = end + 1
    return tuple(names)

#------------------------------------------------------------------------------

# 404760, 404860: the id which ends the bodies of the objects of a manager (a real id like it is followed by 1)
END_OF_OBJECTS = 0xfffffadf

# the classes of the objects of the managers of a party (6a5d20)
CLASS_SPELL = 0x3002
CLASS_WEAPON = 0x3004
CLASS_ARMOR = 0x3005
CLASS_QUICK_ITEM = 0x3008
CLASS_QUEST_ITEM = 0x3009

# 528800: the stats of a unit of a party are its CUnitStats as they are in the memory
STATS_SIZE = 0x704
STATS_SKILLS_COUNT = 8
STATS_PERK_GROUPS_COUNT = 0x24


class SavedObject(object):
    """
    An object of a manager of a party (404760): its id, its class and its code (an item: the material in the low
    word, the shape in the high one; a spell: its prototype in the high word).
    """

    def __init__(self, object_id, class_id, code):
        self.object_id = object_id
        self.class_id = class_id
        self.code = code

    def __repr__(self):
        return 'SavedObject(%d, 0x%x, %d, %d)' % (self.object_id, self.class_id, self.low, self.high)

    @property
    def low(self):
        return self.code & 0xffff

    @property
    def high(self):
        return self.code >> 16


class SavedStats(object):
    """
    The CUnitStats of a unit of a party (0x704 bytes, see engine.experience): the experience +4 and +8, the attributes
    +0xc, +0x14, +0x1c (their values with the perks +0x10, +0x18, +0x20), the science +0x6c8, the skills +0x6cb,
    the levels of the groups of the perks +0x6d3, the waiting experience +0x700. +0x24 and +0x28 are 1.0 in the
    saves of the hero: the multiplier of the speed (engine.stats), not the experience of the perks.
    """

    def __init__(self, data):
        self.data = bytes(data)
        (self.experience, self.start_experience) = struct.unpack_from('<ff', self.data, 4)
        (self.strength, _, self.dexterity, _, self.intelligence, _) = struct.unpack_from('<6f', self.data, 0xc)
        self.science = self.data[0x6c8]
        self.skills = list(self.data[0x6cb:0x6cb + STATS_SKILLS_COUNT])
        self.perk_levels = list(self.data[0x6d3:0x6d3 + STATS_PERK_GROUPS_COUNT])
        self.waiting_experience = struct.unpack_from('<f', self.data, 0x700)[0]

    def __repr__(self):
        return 'SavedStats(exp %g, str %g dex %g int %g, skills %s)' % (
            self.experience, self.strength, self.dexterity, self.intelligence, self.skills)


class SavedUnit(object):
    """
    A record of a unit of a party (65fda0, 0x784 bytes in the memory, engine.roster.UnitRecord): the texts +0, +8
    (the model), +0xc, +0x10, the prototype +0x14, the hash of the name +0x1c, the stats +0x20, 12 bytes +0x724,
    the four lists of the ids of its items +0x730, +0x740, +0x750, +0x760, the name +4, the fields +0x770..+0x780
    and the prototype of its stats +0x18 (CopyStats).
    """

    def __init__(self):
        self.text = ''
        self.model_name = ''
        self.texts = ('', '')
        self.prototype_name = ''
        self.name_hash = 0
        self.stats = None
        self.extra = b''
        self.item_lists = []
        self.name = ''
        self.fields = ()
        self.stats_name = ''

    def __repr__(self):
        return 'SavedUnit(%r, %r, %r)' % (self.name, self.prototype_name, self.model_name)

    @property
    def item_ids(self):
        return [object_id for ids in self.item_lists for object_id in ids]


class SavedParty(object):
    """
    A party of scenario.sav (CParty 661bd0): the zone where it is, the items and the spells of its units by their
    ids and the records of the units.
    """

    def __init__(self, zone='', items=(), spells=(), units=()):
        self.zone = zone
        self.items = list(items)
        self.spells = list(spells)
        self.units = list(units)

    def __repr__(self):
        return 'SavedParty(%r, %d units, %d items, %d spells)' % (
            self.zone, len(self.units), len(self.items), len(self.spells))

    def item(self, object_id):
        for item in self.items:
            if item.object_id == object_id:
                return item
        return None


class _Reader(object):

    def __init__(self, data, offset=0):
        self.data = bytes(data)
        self.offset = offset

    def take(self, size):
        if self.offset + size > len(self.data):
            raise SaveError('the record ends too early')
        result = self.data[self.offset:self.offset + size]
        self.offset += size
        return result

    def u32(self):
        return struct.unpack('<I', self.take(4))[0]

    def byte(self):
        return self.take(1)[0]

    def text(self):
        end = self.data.find(b'\0', self.offset)
        if end < 0:
            raise SaveError('a text without its end')
        result = self.data[self.offset:end].decode('cp1251', errors='replace')
        self.offset = end + 1
        return result


def _read_objects(reader):
    """
    404760: the count and (id, class, code) of every object of a manager.
    """
    return [SavedObject(reader.u32(), reader.u32(), reader.u32()) for _ in range(reader.u32())]


def _skip_bodies(reader, objects):
    """
    404860 (partly): the bodies of the objects (the id and what the object writes, its vf 0x28) end with the id
    END_OF_OBJECTS and the byte 0. The remake does not read the bodies (the wear of the items, the state of the
    spells), it looks for the end; the first body starts with the id of the first object.
    """
    if objects and struct.unpack_from('<I', reader.data, reader.offset)[0] != objects[0].object_id:
        raise SaveError('the bodies do not start with the first object')
    end = reader.data.find(struct.pack('<IB', END_OF_OBJECTS, 0), reader.offset)
    if end < 0:
        raise SaveError('the bodies of the objects have no end')
    reader.offset = end + 5


def _read_unit(reader):
    """
    65fda0: a record of a unit of a party.
    """
    unit = SavedUnit()
    unit.text = reader.text()
    unit.model_name = reader.text()
    unit.texts = (reader.text(), reader.text())
    unit.prototype_name = reader.text()
    unit.name_hash = reader.u32()
    unit.stats = SavedStats(reader.take(STATS_SIZE))
    unit.extra = reader.take(12)
    unit.item_lists = [[reader.u32() for _ in range(reader.u32())] for _ in range(4)]
    unit.name = reader.text()
    unit.fields = (reader.u32(), reader.u32(), reader.byte(), reader.u32(), reader.u32())
    unit.stats_name = reader.text()
    return unit


def read_party(data):
    """
    661bd0: SavedParty of the blob of a party (the record of a party in the tree of scenario.sav); 661880 reads it.
    The name of the zone, a dword, the managers of the items (+0x1c) and of the spells (+0x24) of the party (their
    lists and then their bodies), the ids of the units in the zone (+0x40), the records of the units (+0x30).
    """
    reader = _Reader(data)
    party = SavedParty(reader.text())
    reader.u32()
    party.items = _read_objects(reader)
    party.spells = _read_objects(reader)
    _skip_bodies(reader, party.items)
    _skip_bodies(reader, party.spells)
    for _ in range(reader.u32()):
        reader.u32()
    party.units = [_read_unit(reader) for _ in range(reader.u32())]
    return party


def scenario_parties(body):
    """
    The SavedParty of every party in the body of scenario.sav: the blobs of the records of the tree which start
    with a party (the node 1 of a record: the node 2 with the size and the blob, 6617b0).
    """
    tree, _ = read_blob(body)
    nodes = read_nodes(tree)
    result = []
    for node_id, data in nodes:
        if node_id != 2:
            continue
        for record_id, record in read_nodes(data):
            if record_id != 1:
                continue
            try:
                fields = dict(read_nodes(record))
                inner = dict(read_nodes(fields.get(1, b'')))
                blob = dict(read_nodes(inner.get(2, b''))).get(2)
                if blob is not None:
                    result.append(read_party(blob))
            except (SaveError, struct.error):
                # the other records (the traders) are not parties
                continue
    return result

#------------------------------------------------------------------------------

# 50f900: the place of a map object before the name of its model: the position (12 bytes) and the rotation
# (w, x, y, z, 16 bytes)
PLACE_SIZE = 28


def object_places(body, model_name):
    """
    [(offset, (x, y, z), (w, x, y, z))] of the map objects of mission.sav with the model: the bytes before the name
    of the model (+0xa4, written by 50f900 (partly) after the position +0x1c and the rotation +0x34) hold a rotation of
    the length 1.
    """
    key = model_name.encode('cp1251') + b'\0'
    result = []
    offset = body.find(key)
    while offset >= 0:
        if offset >= PLACE_SIZE:
            position = struct.unpack_from('<3f', body, offset - PLACE_SIZE)
            rotation = struct.unpack_from('<4f', body, offset - 16)
            if abs(sum(v * v for v in rotation) - 1.0) < 1e-3:
                result.append((offset, position, rotation))
        offset = body.find(key, offset + 1)
    return result


def find_unit_place(body, model_name, prototype_name, stats_name=None):
    """
    ((x, y, z), yaw) of the unit of mission.sav with the model whose record of the unit (5321c0 (partly)) ends with the
    names of its prototype and of its numbers, before the next object with that model; None when there is none.
    The remake looks for the names, it does not read the objects one by one.
    """
    tail = ('%s\0%s\0' % (prototype_name, stats_name or prototype_name)).encode('cp1251')
    places = object_places(body, model_name)
    for index, (offset, position, rotation) in enumerate(places):
        end = places[index + 1][0] if index + 1 < len(places) else len(body)
        if body.find(tail, offset, end) >= 0:
            w, _, _, z = rotation
            return position, 2.0 * math.atan2(z, w)
    return None
