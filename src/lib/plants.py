"""
Compact binary form of the plants of a biome: plants.bin keeps the same data as plants.json did.

plants.json is {"template#texture#parts": ["c1:c2:c3 x y direction plant_name template#texture#parts", ...]},
see server.land.build_plants(). decode() returns exactly that dictionary, string by string, so the server
and the client read the plants the same way as before. A plant which is not material, units walk through it,
has one more word at the end of its string: "c1:c2:c3 x y direction plant_name template#texture#parts 0",
see split(). A coordinate of a plant is the number of the segment
and the shift inside of it in 1/100 of a segment, a direction is in whole degrees, the coefficients and the name
of the plant are taken from the table of strings.

The file is packed into the .tar.gz archive of the biome, the columns are laid out so gzip compresses them well:
the numbers of the segments are kept as the differences with the previous plant, the neighbors in the list are
close to each other on the map, and the bytes of 2-byte numbers are split: all low bytes, then all high bytes.
All numbers are little-endian:

    b'WPLB', version: u8
    strings count: u16, every string: length u8 + utf-8 bytes
    groups count: u16, every group: index of its key in the strings u16, plants count u32
    columns of all plants of all groups one after another, in the order of the groups:
        x segment difference i16, x shift u8, y segment difference i16, y shift u8,
        direction u16, index of the coefficients in the strings u16, index of the plant name in the strings u16,
        material u8: 1 when units can not walk through the plant, 0 when they can

The files of version 1 have no column of the material, all their plants are material.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

#------------------------------------------------------------------------------

MAGIC = b'WPLB'

VERSION = 2

# versions of the files decode() reads
VERSIONS = (1, 2)

# the last word of the string of a plant which is not material, see split()
NOT_MATERIAL = '0'

#------------------------------------------------------------------------------


def coordinate_string(segment, shift):
    """
    Text of the coordinate, the same way server.land.build_plants() writes it: x + shift_x.
    """
    return f'{segment + shift / 100.0}'


def split(plant_coded):
    """
    Parts of the string of a plant: (coefs, x, y, direction, plant_name, plant_template, material),
    material is False when units walk through the plant, see the module description.
    """
    words = plant_coded.split(' ')
    if len(words) == 7 and words[6] == NOT_MATERIAL:
        return tuple(words[:6]) + (False, )
    if len(words) != 6:
        raise ValueError(f'plant {plant_coded!r} is not known')
    return tuple(words) + (True, )


def join(coefs, x, y, direction, plant_name, plant_template, material=True):
    """
    String of a plant, see split().
    """
    text = f'{coefs} {x} {y} {direction} {plant_name} {plant_template}'
    if not material:
        text += ' ' + NOT_MATERIAL
    return text


def split_bytes(fmt, values):
    """
    2-byte numbers packed with the struct format letter, low bytes first and then high bytes.
    """
    data = struct.pack(f'<{len(values)}{fmt}', *values)
    return data[0::2] + data[1::2]


def join_bytes(fmt, data, offset, count):
    """
    Reads count 2-byte numbers written by split_bytes(), returns them and the offset after them.
    """
    joined = bytearray(2 * count)
    joined[0::2] = data[offset:offset + count]
    joined[1::2] = data[offset + count:offset + 2 * count]
    return struct.unpack(f'<{count}{fmt}', joined), offset + 2 * count


def encode(plants):
    """
    Returns plants.bin data of the plants dictionary, see the module description.
    Raises ValueError when a plant can not be written so that decode() gives back the same text.
    """
    strings = []
    string_index = {}

    def _index(text):
        if text not in string_index:
            if len(text.encode('utf-8')) > 255:
                raise ValueError(f'string {text!r} is too long for plants.bin')
            string_index[text] = len(strings)
            strings.append(text)
        return string_index[text]

    groups = []
    columns = {'xs': [], 'xf': [], 'ys': [], 'yf': [], 'd': [], 'c': [], 'n': [], 'm': []}
    last_x = 0
    last_y = 0
    for plant_key, plants_list in plants.items():
        groups.append((_index(plant_key), len(plants_list)))
        for plant_coded in plants_list:
            coefs, x, y, direction, plant_name, plant_template, material = split(plant_coded)
            x_segment, x_shift = divmod(round(float(x) * 100), 100)
            y_segment, y_shift = divmod(round(float(y) * 100), 100)
            if coordinate_string(x_segment, x_shift) != x or coordinate_string(y_segment, y_shift) != y:
                raise ValueError(f'position of plant {plant_coded!r} can not be written into plants.bin')
            if str(int(direction)) != direction or not 0 <= int(direction) <= 65535 or plant_template != plant_key:
                raise ValueError(f'plant {plant_coded!r} of {plant_key!r} can not be written into plants.bin')
            columns['xs'].append(x_segment - last_x)
            columns['xf'].append(x_shift)
            columns['ys'].append(y_segment - last_y)
            columns['yf'].append(y_shift)
            columns['d'].append(int(direction))
            columns['c'].append(_index(coefs))
            columns['n'].append(_index(plant_name))
            columns['m'].append(1 if material else 0)
            last_x = x_segment
            last_y = y_segment
    if len(strings) > 65535 or len(groups) > 65535:
        raise ValueError(f'too many kinds of plants for plants.bin: {len(groups)} groups, {len(strings)} strings')
    parts = [MAGIC, struct.pack('<BH', VERSION, len(strings))]
    for text in strings:
        data = text.encode('utf-8')
        parts.append(struct.pack('<B', len(data)))
        parts.append(data)
    parts.append(struct.pack('<H', len(groups)))
    for key_index, plants_count in groups:
        parts.append(struct.pack('<HI', key_index, plants_count))
    parts.append(split_bytes('h', columns['xs']))
    parts.append(bytes(columns['xf']))
    parts.append(split_bytes('h', columns['ys']))
    parts.append(bytes(columns['yf']))
    parts.append(split_bytes('H', columns['d']))
    parts.append(split_bytes('H', columns['c']))
    parts.append(split_bytes('H', columns['n']))
    parts.append(bytes(columns['m']))
    return b''.join(parts)


def decode(data):
    """
    Returns the plants dictionary from plants.bin data, the same as json.loads() of plants.json gave.
    """
    if data[:4] != MAGIC:
        raise ValueError('not a plants.bin file')
    version, strings_count = struct.unpack_from('<BH', data, 4)
    if version not in VERSIONS:
        raise ValueError(f'plants.bin version {version} is not supported, expected one of {VERSIONS}')
    offset = 7
    strings = []
    for _ in range(strings_count):
        length = data[offset]
        strings.append(data[offset + 1:offset + 1 + length].decode('utf-8'))
        offset += 1 + length
    groups_count, = struct.unpack_from('<H', data, offset)
    offset += 2
    groups = []
    for _ in range(groups_count):
        groups.append(struct.unpack_from('<HI', data, offset))
        offset += 6
    count = sum(plants_count for _, plants_count in groups)
    x_steps, offset = join_bytes('h', data, offset, count)
    x_shifts = data[offset:offset + count]
    offset += count
    y_steps, offset = join_bytes('h', data, offset, count)
    y_shifts = data[offset:offset + count]
    offset += count
    directions, offset = join_bytes('H', data, offset, count)
    coefs_indexes, offset = join_bytes('H', data, offset, count)
    names_indexes, offset = join_bytes('H', data, offset, count)
    materials = data[offset:offset + count] if version >= 2 else bytes([1]) * count
    offset += count if version >= 2 else 0
    plants = {}
    i = 0
    x_segment = 0
    y_segment = 0
    for key_index, plants_count in groups:
        plant_key = strings[key_index]
        plants_list = plants.setdefault(plant_key, [])
        for _ in range(plants_count):
            x_segment += x_steps[i]
            y_segment += y_steps[i]
            plants_list.append(join(
                strings[coefs_indexes[i]],
                coordinate_string(x_segment, x_shifts[i]),
                coordinate_string(y_segment, y_shifts[i]),
                directions[i],
                strings[names_indexes[i]],
                plant_key,
                material=bool(materials[i]),
            ))
            i += 1
    if _Debug:
        print(f'decoded {count} plants of {len(plants)} kinds from {len(data)} bytes')
    return plants
