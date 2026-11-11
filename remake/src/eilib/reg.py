"""
Reader of the binary .reg files of Evil Islands: menus.reg (the screen rects of the menu buttons), music.reg,
ai.reg, smessbase.reg and others. game.exe reads them with CFileRegistry (6945d0 checks the signature).

    u32 signature 0x45ab3efb, u16 sections count,
    sections count x (i16 tree link, u32 offset of the section from the start of the file)

A section: u16 keys count, u16 name length, name, keys count x (i16 tree link, u32 offset of the key
from the start of the section). A key: u8 type, u16 name length, name, the value. The type 0 is int32,
1 is float32, 2 is a string with its u16 length; the bit 0x80 makes an array: u16 count, then the values.
The tree links only speed up the search by name, the order of the table is kept.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

#------------------------------------------------------------------------------

SIGNATURE = 0x45ab3efb

TYPE_INT = 0
TYPE_FLOAT = 1
TYPE_STRING = 2
ARRAY = 0x80

ENCODING = 'cp1251'

#------------------------------------------------------------------------------


class RegError(Exception):
    pass

#------------------------------------------------------------------------------


def parse(data):
    """
    Section name -> {key name -> value}, an array becomes a tuple.
    """
    data = bytes(data)
    if len(data) < 6 or struct.unpack_from('<I', data, 0)[0] != SIGNATURE:
        raise RegError('not a .reg file')
    sections_count = struct.unpack_from('<H', data, 4)[0]
    sections = {}
    for n in range(sections_count):
        _, section_offset = struct.unpack_from('<hI', data, 6 + n * 6)
        name, keys = _section(data, section_offset)
        sections[name] = keys
    if _Debug:
        print('.reg: %d sections' % len(sections))
    return sections


def _string(data, offset):
    length = struct.unpack_from('<H', data, offset)[0]
    return data[offset + 2:offset + 2 + length].decode(ENCODING, errors='replace'), offset + 2 + length


def _section(data, offset):
    keys_count = struct.unpack_from('<H', data, offset)[0]
    name, table_offset = _string(data, offset + 2)
    keys = {}
    for n in range(keys_count):
        _, key_offset = struct.unpack_from('<hI', data, table_offset + n * 6)
        key_name, value = _key(data, offset + key_offset)
        keys[key_name] = value
    return name, keys


def _key(data, offset):
    kind = data[offset]
    name, offset = _string(data, offset + 1)
    if kind & ARRAY:
        count = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        values = []
        for _ in range(count):
            value, offset = _value(data, offset, kind & ~ARRAY)
            values.append(value)
        return name, tuple(values)
    return name, _value(data, offset, kind)[0]


def _value(data, offset, kind):
    if kind == TYPE_INT:
        return struct.unpack_from('<i', data, offset)[0], offset + 4
    if kind == TYPE_FLOAT:
        return struct.unpack_from('<f', data, offset)[0], offset + 4
    if kind == TYPE_STRING:
        return _string(data, offset)
    raise RegError('unknown value type %d at %d' % (kind, offset))
