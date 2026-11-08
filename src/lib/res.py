"""
Readers of the game resource files: .res archives, models, bones, animations and textures,
and the paths to the data and catalog folders.
"""

#------------------------------------------------------------------------------

_Debug = True

#------------------------------------------------------------------------------

import os
import time
import re
import json
import struct
import shutil
import tempfile
import collections

import numpy as np
from PIL import Image

#------------------------------------------------------------------------------

from lib import const

#------------------------------------------------------------------------------

ENCODE = "cp1251"
ROOT_PATH = ''
DATA_PATH = ''
# catalog/ folder is part of the client, files there are never downloaded, see client.main
CATALOG_PATH = ''

#------------------------------------------------------------------------------


def data_path(data_path):
    global DATA_PATH
    return os.path.join(DATA_PATH, data_path)


def catalog_path(file_name):
    return os.path.join(CATALOG_PATH, file_name)


def read_byte(file, count=1):
    # Unsigned 1b integer
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('B', file.read(1))[0])        
    return array if count > 1 else array[0]


def read_char(file, count=1):
    # Signed 1b integer
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('b', file.read(1))[0])        
    return array if count > 1 else array[0]

def read_short(file, count=1):
    # Signed 2b integer
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('h', file.read(2))[0])        
    return array if count > 1 else array[0]


def read_ushort(file, count=1):
    # Unsigned 2b integer
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('H', file.read(2))[0])
    return array if count > 1 else array[0]


def read_int(file, count=1):
    # Signed 4b integer
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('i', file.read(4))[0])
    return array if count > 1 else array[0]


def read_uint(file, count=1):
    # Unsigned 4b integer
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('I', file.read(4))[0])
    return array if count > 1 else array[0]


def read_int64(file, count=1):
    # Signed 4b integer
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('q', file.read(8))[0])
    return array if count > 1 else array[0]


def read_uint64(file, count=1):
    # Unsigned 8b integer
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('Q', file.read(8))[0])
    return array if count > 1 else array[0]


def half_to_float(h):
    # Conversion between 2b half to 4b float
    s = int((h >> 15) & 0x00000001)    # sign
    e = int((h >> 10) & 0x0000001f)    # exponent
    f = int(h & 0x000003ff)            # fraction
    if e == 0:
        if f == 0:
            return int(s << 31)
        else:
            while not (f & 0x00000400):
                f <<= 1
                e -= 1
            e += 1
            f &= ~0x00000400
    elif e == 31:
        if f == 0:
            return int((s << 31) | 0x7f800000)
        else:
            return int((s << 31) | 0x7f800000 | (f << 13))
    e = e + (127 -15)
    f = f << 13
    return int((s << 31) | (e << 23) | f)


def read_half(file, count=1):
    # Hard-float 2b real
    array = [] 
    for _ in range(count):
        nz = file.read(2)
        v = struct.unpack('H', nz)
        x = half_to_float(v[0])
        # hack to coerce int to float
        pck = struct.pack('I', x)
        f = struct.unpack('f', pck)
        array.append(f[0])
    return array if count > 1 else array[0]


def read_float(file, count=1):
    # Floating-point 4b real
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('f', file.read(4))[0])
    return array if count > 1 else array[0]


def read_double(file, count=1):
    # Floating-point 8b real
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('d', file.read(4))[0])        
    return array if count > 1 else array[0]


def read_str(file, length, code="cp1251"):
    # Non-terminated char array
    buf = ""
    for _ in range(length):
        b = struct.unpack('c', file.read(1))[0]
        if ord(b) != 0:
            buf += b.decode(code)
    return buf


def read_cstr(file, code="cp1251"):
    # Zero-terminated char array (C-style string)
    buf = ""
    while True:
        b = struct.unpack('c', file.read(1))[0]
        
        if b is None or ord(b) == 0:
            return buf
        else:
            buf += b.decode(code)


def floats_list(data, offset, shape):
    """
    Little-endian 4b floats from the bytes at the offset as nested lists of the given shape and the offset after them.
    All numbers are converted at once, reading them one by one is many times slower.
    """
    count = 1
    for size in shape:
        count *= size
    values = np.frombuffer(data, dtype='<f4', count=count, offset=offset).reshape(shape).tolist()
    return values, offset + 4 * count


def floats_array(data, offset, shape):
    """
    The same as floats_list() but the numbers are kept in one numpy array of 8b floats instead of many Python lists.
    """
    count = 1
    for size in shape:
        count *= size
    values = np.frombuffer(data, dtype='<f4', count=count, offset=offset).astype(np.float64).reshape(shape)
    return values, offset + 4 * count


def ushorts_array(data, offset, shape):
    count = 1
    for size in shape:
        count *= size
    values = np.frombuffer(data, dtype='<u2', count=count, offset=offset).reshape(shape)
    return values, offset + 2 * count


def read_bon_info(file_path):
    if not os.path.isfile(file_path):
        return None
    with open(file_path, "rb") as file:
        return parse_bon_info(file.read())


def parse_bon_info(data):
    # 8 shifts of the part, one for every body variant
    return floats_list(data, 0, (len(data) // 12, 3))[0]


def read_anm_info(file_path, arrays=False):
    with open(file_path, "rb") as file:
        return parse_anm_info(file.read(), arrays=arrays)


def parse_anm_info(data, arrays=False):
    """
    With arrays=True the rotations, the shifts and the morphing of the frames are numpy arrays instead of lists.
    """
    read_floats = floats_array if arrays else floats_list
    info = []
    rotations_count = struct.unpack_from('<I', data, 0)[0]
    info.append(rotations_count)
    rotations, offset = read_floats(data, 4, (rotations_count, 4))
    info.append(rotations)
    shifts_count = struct.unpack_from('<I', data, offset)[0]
    info.append(shifts_count)
    shifts, offset = read_floats(data, offset + 4, (shifts_count, 3))
    info.append(shifts)
    info.extend(struct.unpack_from('<II', data, offset))
    if info[4] != 0 and info[5] != 0:
        info.append(read_floats(data, offset + 8, (info[4], info[5], 3))[0])
    return info


def read_anm_first_shift(file_path):
    """
    Shift of the part in the first frame of the animation without reading all frames, None when there are no shifts.
    """
    with open(file_path, "rb") as file:
        rotations_count = read_uint(file)
        file.seek(4 + 16 * rotations_count)
        if read_uint(file) == 0:
            return None
        return read_float(file, 3)


def read_fig_info(file_path):
    if not os.path.isfile(file_path):
        return None
    with open(file_path, "rb") as file:
        return parse_fig_info(file.read())


def parse_fig_info(data):
    if data[:3] != b'\x46\x49\x47':
        raise Exception("Incorrect magic!")
    n = data[3] - 48
    info = [n]
    info.extend(struct.unpack_from('<6I', data, 4))
    # one number is skipped
    info.extend(struct.unpack_from('<2I', data, 32))
    info[4] //= 3
    offset = 40
    # center, min and max
    for _ in range(3):
        values, offset = floats_list(data, offset, (n, 3))
        info.append(values)
    # radius
    values, offset = floats_list(data, offset, (n, ))
    info.append(values if n != 1 else values[0])
    # Vertex blocks
    values, offset = floats_list(data, offset, (info[1], 3, n, 4))
    info.append(values)
    # Normals
    values, offset = floats_list(data, offset, (info[2], 4, 4))
    info.append(values)
    # UV
    values, offset = floats_list(data, offset, (info[3], 2))
    info.append(values)
    # Indices
    values, offset = ushorts_array(data, offset, (info[4], 3))
    info.append(values.tolist())
    # Vertex components
    values, offset = ushorts_array(data, offset, (info[5], 3))
    values = values.astype(np.int64)
    info.append(np.stack([values[:, 0] >> 2, values[:, 0] & 3, values[:, 1] >> 2, values[:, 1] & 3, values[:, 2]], axis=1).tolist())
    # Deformation
    values, offset = ushorts_array(data, offset, (info[6], 2))
    info.append(values.tolist())
    return info


def read_mod_info(file_path):
    mod_name = file_path.replace('.mod', '')
    with open(file_path, "rb") as file:
        tree = read_res_filetree(file)
        # f_name = os.path.splitext(os.path.basename(file_path))[0]
        for element in tree:
            element[0] += ".fig" if mod_name != element[0] else ".lnk"
        # unpack_res(file, tree, file_path)
    return tree


def unpack_mod_info(file_path, destination_dir_path):
    with open(file_path, "rb") as file:
        tree = read_res_filetree(file)
        f_name = os.path.splitext(os.path.basename(file_path))[0]
        for element in tree:
            element[0] += ".fig" if f_name != element[0] else ".lnk"
        unpack_res(file, tree, destination_dir_path)
    return tree


def add_child_to_list(arr, parent_name, child_name):
    if arr[0] == parent_name:
        arr[1].append([child_name, []])
    else:
        for part_name in arr[1]:
            add_child_to_list(part_name, parent_name, child_name)


def add_child_to_dict(dct, parent_name, child_name):
    if parent_name in dct:
        if child_name in dct[parent_name]:
            return
        dct[parent_name][child_name] = {}
        return
    for other_name in dct.keys():
        add_child_to_dict(dct[other_name], parent_name, child_name)


def add_children(tree, parent_name, child_name):
    if parent_name in tree:
        if child_name in tree[parent_name]:
            return
        tree[parent_name][child_name] = {}
    else:
        tree[parent_name] = {}
        for child_name in tree.keys():
            tree[parent_name][child_name] = {}
            add_children(tree[parent_name][child_name], parent_name, child_name)


def flat_tree(tree, arr=None):
    if arr is None:
        arr = []
    arr.append(tree[0])
    if len(tree[1]) != 0:
        for leaf in tree[1]:
            flat_tree(leaf, arr)
    return arr


def walk_tree(tree, parent, visitor, arr=None, dct=None):
    if arr is None:
        arr = []
    if dct is None:
        dct = {}
    arr.append(tree[0])
    visitor(parent, tree[0])
    if len(tree[1]) != 0:
        for leaf in tree[1]:
            walk_tree(leaf, tree[0], visitor, arr)


def read_lnk_info(file_path):
    lst = []
    parents = {}
    tree = {}
    childs = {}
    with open(file_path, "rb") as file:
        for _ in range(read_uint(file)):
            name_len = read_uint(file)
            new_name = read_str(file, name_len)
            parent_name_len = read_uint(file)
            if parent_name_len == 0:
                lst = [new_name, []]
                parents[new_name] = None
                if new_name not in childs:
                    childs[new_name] = []
                tree[new_name] = {}
            else:
                parent_name = read_str(file, parent_name_len)
                parents[new_name] = parent_name
                if new_name not in childs:
                    childs[new_name] = []
                if parent_name not in childs:
                    childs[parent_name] = []
                childs[parent_name].append(new_name)
                add_child_to_list(lst, parent_name, new_name)
                add_child_to_dict(tree, parent_name, new_name)
    return lst, tree, parents, childs


def read_res_filetree(file, return_dict=False):
    magic = file.read(4)
    if magic != b'\x3C\xE2\x9C\x01':
        raise Exception("Incorrect magic!")
    filetree = []
    filetree_dict = {}
    buf = []
    table_size = read_uint(file)
    table_offset = read_uint(file)
    read_uint(file)  # names_len
    file.seek(table_offset)
    for i in range(table_size):
        buf.append(read_uint(file, 4))
        buf[i].append(read_ushort(file))
        buf[i].append(read_uint(file))
    names_offset = file.tell()
    for i in range(table_size):
        file.seek(buf[i][5] + names_offset)
        try:
            _name = read_str(file, buf[i][4], ENCODE).replace("\\", "/")
        except Exception as e:
            print(file, i, e)
            continue
        filetree.append([_name, buf[i][2], buf[i][1]])
        if _name in filetree_dict:
            raise ValueError(f'File {_name} is duplicated')
        filetree_dict[_name] = [_name, buf[i][2], buf[i][1]]
    if return_dict:
        return filetree_dict
    return filetree


def unpack_res(file, filetree, destination_dir_path):
    for element in filetree:
        dest_file_path = os.path.join(destination_dir_path, element[0])
        if os.path.isfile(dest_file_path):
            continue
        if not os.path.exists(destination_dir_path):
            os.makedirs(destination_dir_path)
        with open(dest_file_path, "wb") as new_file:
            file.seek(element[1])
            buf = file.read(element[2])
            new_file.write(buf)


def unpack_res_element(file, element, dest_file_path=None):
    if not dest_file_path:
        dest_file_path = element[0]
    if os.path.isfile(dest_file_path):
        return dest_file_path
    with open(dest_file_path, "wb") as new_file:
        file.seek(element[1])
        buf = file.read(element[2])
        new_file.write(buf)
    return dest_file_path


def res_name_hash(name, count):
    # slot in the .res table where the lookup of that name starts, same as the game does
    return sum(ord(c) for c in name.lower()) % count


def write_res(dest_file_path, items, timestamp=None):
    """
    Writes a new .res file, items is a list of (name, data_bytes).
    Layout: header, data of all elements, table of elements, names.
    Table rows are placed by the hash of the name, collisions are chained via the first field of the row.
    """
    if timestamp is None:
        timestamp = int(time.time())
    count = len(items)
    names = [name.replace('/', '\\').encode(ENCODE) for name, _ in items]
    rows = []
    offset = 16
    for i, (_, data) in enumerate(items):
        rows.append([0xFFFFFFFF, len(data), offset, timestamp, len(names[i]), 0])
        offset += len(data)
    table_offset = offset
    # place every element at the slot of its hash if possible, others go to free slots and are chained
    slots = [None] * count
    pending = []
    for i, (name, _) in enumerate(items):
        h = res_name_hash(name, count)
        if slots[h] is None:
            slots[h] = i
        else:
            pending.append(i)
    free_slots = [s for s in range(count) if slots[s] is None]
    for i in pending:
        s = free_slots.pop(0)
        slots[s] = i
        j = res_name_hash(items[i][0], count)
        while rows[slots[j]][0] != 0xFFFFFFFF:
            j = rows[slots[j]][0]
        rows[slots[j]][0] = s
    # first field keeps the slot index of the next element in the chain, table is written in the slots order
    names_blob = b''
    table = b''
    for s in range(count):
        i = slots[s]
        row = rows[i]
        row[5] = len(names_blob)
        names_blob += names[i]
        table += struct.pack('<IIIIHI', *row)
    tmp_file_path = dest_file_path + '.part'
    with open(tmp_file_path, 'wb') as f:
        f.write(b'\x3C\xE2\x9C\x01')
        f.write(struct.pack('<III', count, table_offset, len(names_blob)))
        for _, data in items:
            f.write(data)
        f.write(table)
        f.write(names_blob)
    os.replace(tmp_file_path, dest_file_path)
    return dest_file_path


def figures_templates(figures_file, filetree_dict=None, tmp_dir_path=None):
    """
    Groups elements of figures.res by the model (template) they belong to, returns ({template: [element names]}, [not assigned names]).
    Templates are the names of .lnk and .mod elements, .anm and .bon with the same name belong to them,
    parts of a model are named <template><part>.fig / .bon where parts are listed in the <template>.lnk tree.
    Template names are lower case, element names are as they are in figures.res.
    """
    if filetree_dict is None:
        filetree_dict = read_res_filetree(figures_file, return_dict=True)
    lower_names = {}
    for name in filetree_dict.keys():
        lower_names[name.lower()] = name
    templates = {}
    for name_l, name in lower_names.items():
        if name_l.endswith('.lnk') or name_l.endswith('.mod'):
            templates.setdefault(name_l[:-4], [])
    assigned = set()
    for t in templates.keys():
        for ext in ('.lnk', '.mod', '.anm', '.bon'):
            if t + ext in lower_names:
                templates[t].append(lower_names[t + ext])
                assigned.add(t + ext)
    own_tmp_dir = tmp_dir_path is None
    if own_tmp_dir:
        tmp_dir_path = tempfile.mkdtemp()
    try:
        for t in templates.keys():
            if t + '.lnk' not in lower_names:
                continue
            lnk_file_path = unpack_res_element(figures_file, filetree_dict[lower_names[t + '.lnk']], dest_file_path=os.path.join(tmp_dir_path, t + '.lnk'))
            lnk_list, _, _, _ = read_lnk_info(lnk_file_path)
            if not lnk_list:
                continue
            for part_name in flat_tree(lnk_list):
                for ext in ('.fig', '.bon'):
                    k = t + part_name.lower() + ext
                    if k in lower_names and k not in assigned:
                        templates[t].append(lower_names[k])
                        assigned.add(k)
    finally:
        if own_tmp_dir:
            shutil.rmtree(tmp_dir_path, ignore_errors=True)
    # the rest of the parts belong to the template with the longest matching name prefix
    by_length = sorted(templates.keys(), key=lambda x: -len(x))
    not_assigned = []
    for name_l, name in lower_names.items():
        if name_l in assigned:
            continue
        for t in by_length:
            if name_l.startswith(t):
                templates[t].append(name)
                assigned.add(name_l)
                break
        else:
            not_assigned.append(name)
    return templates, not_assigned


def extract_figures_template(figures_file, filetree_dict, element_names, dest_file_path):
    """
    Writes elements of one template from already opened figures.res into a separate .res file.
    Names are written in lower case, because the client looks them up in lower case.
    """
    items = []
    for name in element_names:
        _, offset, size = filetree_dict[name]
        figures_file.seek(offset)
        items.append((name.lower(), figures_file.read(size)))
    return write_res(dest_file_path, items)


_texture_groups_compiled = None


def texture_group(texture):
    """
    Returns the name of the group (and of the <group>.res file) for the texture name.
    """
    global _texture_groups_compiled
    if _texture_groups_compiled is None:
        _texture_groups_compiled = [(g, re.compile(r)) for g, r in const.TEXTURE_GROUPS]
    texture = texture.lower()
    for group, regex in _texture_groups_compiled:
        m = regex.match(texture)
        if m:
            if group.endswith(':'):
                return m.group(1)
            return group
    return 'objects'


def textures_groups(textures_file, filetree_dict=None):
    """
    Groups elements of textures.res by the groups from const.TEXTURE_GROUPS: {group: [element names]}.
    Element names are as they are in textures.res.
    """
    if filetree_dict is None:
        filetree_dict = read_res_filetree(textures_file, return_dict=True)
    groups = {}
    for name in filetree_dict.keys():
        groups.setdefault(texture_group(os.path.splitext(name)[0]), []).append(name)
    return groups


def figures_template_names(figures_file, filetree_dict=None):
    """
    Returns lower case names of all models (templates) in figures.res: names of the .lnk and .mod elements.
    """
    if filetree_dict is None:
        filetree_dict = read_res_filetree(figures_file, return_dict=True)
    names = set()
    for name in filetree_dict.keys():
        name_l = name.lower()
        if name_l.endswith('.lnk') or name_l.endswith('.mod'):
            names.add(name_l[:-4])
    return names


def texture_model(texture, templates_by_length):
    """
    Returns the model the texture belongs to, texture names are <model><part / armor / weapon / material>.
    The model is the longest template name the texture starts with, templates_by_length is sorted longest first.
    Textures of unknown models go to the model named as the texture without trailing digits: wolf01 -> wolf.
    """
    texture = texture.lower()
    for t in templates_by_length:
        if texture.startswith(t):
            return t
    return re.match(r'^(.*?)\d*$', texture).group(1) or texture


def texture_model_group(texture, templates_by_length):
    """
    Returns the group of the texture inside redress.res: the model, and for models from const.REDRESS_SPLIT_MODELS
    also the kind of armor / weapon: unhufeax_01.br.1 -> unhufe_ax, unhumaskin_00 -> unhuma_skin.
    Pictures of wounds (<body part>w<number>: unhufelhw1, unhumabdw2) go to <model>_w.
    Textures of a split model which match neither rule stay in <model>.
    """
    texture = texture.lower()
    model = texture_model(texture, templates_by_length)
    if model not in const.REDRESS_SPLIT_MODELS:
        return model
    rest = texture[len(model):]
    if re.match(r'^[a-z]{2}w\d+$', rest):
        return model + '_w'
    m = re.match(r'^([a-z]+)_', rest)
    if m:
        return model + '_' + m.group(1)
    return model


def textures_models(textures_file, templates, filetree_dict=None):
    """
    Groups elements of redress.res by the model they belong to: {group: [element names]}, see texture_model_group().
    templates are the model names from figures.res, see figures_template_names().
    Element names are as they are in redress.res.
    """
    if filetree_dict is None:
        filetree_dict = read_res_filetree(textures_file, return_dict=True)
    by_length = sorted(templates, key=lambda x: -len(x))
    models = {}
    for name in filetree_dict.keys():
        # names look like unhufeax_01.br.1.mmp, only the last extension is dropped
        texture = name[:-4] if name.lower().endswith('.mmp') else name
        models.setdefault(texture_model_group(texture, by_length), []).append(name)
    return models


def extract_textures_group(textures_file, filetree_dict, element_names, dest_file_path):
    """
    Writes elements of one group from already opened textures.res into a separate .res file.
    """
    # same layout as for models: names in lower case
    return extract_figures_template(textures_file, filetree_dict, element_names, dest_file_path)


def scan_figures_data(figures_res_file_path):
    items = {}
    with open(figures_res_file_path, 'rb') as figures_file:
        res_filetree_dict = read_res_filetree(figures_file, return_dict=True)
        for k in res_filetree_dict.keys():
            ext = k[-4:].lower().replace('.', '')
            if ext not in items:
                items[ext] = []
            items[ext].append(k[:-4])
    return items


def scan_textures_data(texture_res_file_path):
    lst = []
    with open(texture_res_file_path, 'rb') as texture_file:
        res_filetree_dict = read_res_filetree(texture_file, return_dict=True)
        for k in res_filetree_dict.keys():
            if k.lower().endswith('.mmp'):
                name = k[:-4].lower()
                if name not in lst:
                    lst.append(name)
    return lst


def unpack_texture(texture_res_file_path, destination_dir_path, name):
    if not os.path.isdir(destination_dir_path):
        os.makedirs(destination_dir_path)
    if _Debug:
        print(f'    unpacking texture {name} from {texture_res_file_path} into {destination_dir_path}')
    with open(texture_res_file_path, 'rb') as texture_file:
        res_filetree_dict = read_res_filetree(texture_file, return_dict=True)
        for k in res_filetree_dict.keys():
            if k.lower().endswith('.mmp') and k[:-4].lower() == name.lower():
                mmp_file_path = os.path.join(destination_dir_path, name.lower() + '.mmp')
                png_file_path = os.path.join(destination_dir_path, name.lower() + '.png')
                unpack_res_element(texture_file, res_filetree_dict[k], dest_file_path=mmp_file_path)
                pil_img = read_mmp(mmp_file_path)
                pil_img.save(png_file_path)
                os.remove(mmp_file_path)
                if _Debug:
                    print(f'unpacked texture {name} to {png_file_path}')
                return png_file_path
    if _Debug:
        print(f'WARNING texture {name} was not found at in {texture_res_file_path}')
    return None


def extract_color(pixel, mask, shift, count):
    return int(0.5 + 255 * ((pixel & mask) >> shift) / (mask >> shift))


def get_color(pixel, a_d, r_d, g_d, b_d):
    if a_d[2] == 0:
        a = 255
    else:
        a = extract_color(pixel, a_d[0], a_d[1], a_d[2])
    r = extract_color(pixel, r_d[0], r_d[1], r_d[2])
    g = extract_color(pixel, g_d[0], g_d[1], g_d[2])
    b = extract_color(pixel, b_d[0], b_d[1], b_d[2])    
    return [r, g, b, a]


def convert_DXT(file, width, height, DXT3=False):
    data = np.zeros((height, width, 4), dtype=np.uint8)
    color = [0, 0, 0, 0]    
    for i in range(height // 4):
        for j in range(width // 4):
            if DXT3:
                for x in range(4):
                    row = read_ushort(file)
                    for y in range(4):
                        data[i * 4 + x, j * 4 + y, 3] = (row & 15) * 17
                        row >>= 4
            gen_c1 = read_ushort(file)
            gen_c2 = read_ushort(file)
            color[0] = get_color(gen_c1, [0, 0, 0], [63488, 11, 5], [2016, 5, 6], [31, 0, 5])
            color[1] = get_color(gen_c2, [0, 0, 0], [63488, 11, 5], [2016, 5, 6], [31, 0, 5])            
            if gen_c1 > gen_c2 or DXT3:
                color[2] = [(2 * color[0][i] + color[1][i]) // 3 for i in range(4)]
                color[3] = [(color[0][i] + 2 * color[1][i]) // 3 for i in range(4)]
            else:
                color[2] = [(color[0][i] + color[1][i]) // 2 for i in range(4)]
                color[3] = [0, 0, 0, 0]
            for x in range(4):
                row = read_byte(file)
                for y in range(4):
                    if DXT3:
                        for k in range(3):
                            data[i * 4 + x, j * 4 + y, k] = color[row & 3][k]
                    else:
                        data[i * 4 + x, j * 4 + y] = color[row & 3][:]
                    row >>= 2
    return data


def decompress_PNT3(file, size, width, height):
    source = file.read(size)
    src = 0
    n = 0
    destination = b""
    while src < size:
        v = int.from_bytes(source[src:src + 4], byteorder='little')
        src += 4        
        if v > 1000000 or v == 0:
            n += 1
        else:
            destination += source[src - (1 + n) * 4:src - 4]
            destination += b"\x00" * v
            n = 0
    destination += source[src - n * 4:src]
    data = np.zeros((height, width, 4), dtype=np.uint8)
    n = 0
    for i in range(height):
        for j in range(width):
            data[i, j] = [int.from_bytes(destination[n + 2:n + 3], byteorder='little'),
                          int.from_bytes(destination[n + 1:n + 2], byteorder='little'),
                          int.from_bytes(destination[n + 0:n + 1], byteorder='little'),
                          int.from_bytes(destination[n + 3:n + 4], byteorder='little')]
            n += 4            
    return data


def read_mmp(file_path):
    with open(file_path, "rb") as file:
        if file.read(4) != b"\x4D\x4D\x50\x00":
            print("Incorrect magic!")
            return
        width = read_uint(file)
        height = read_uint(file)
        mip_count = read_uint(file)
        form = file.read(4)
        if form == b"DXT1":
            file.seek(76)
            data = convert_DXT(file, width, height)
        elif form == b"DXT3":
            file.seek(76)
            data = convert_DXT(file, width, height, DXT3=True)
        elif form == b"PNT3":
            file.seek(76)
            data = decompress_PNT3(file, mip_count, width, height)
        else:
            pixel_size = read_uint(file) // 8
            a_d = read_uint(file, 3)
            r_d = read_uint(file, 3)
            g_d = read_uint(file, 3)
            b_d = read_uint(file, 3)
            file.seek(76)
            data = np.zeros((height, width, 4), dtype=np.uint8)
            if pixel_size == 2:
                pix_reader = read_ushort
            elif pixel_size == 4:
                pix_reader = read_uint
            for i in range(height):
                for j in range(width):
                    pixel = pix_reader(file)
                    data[i, j] = get_color(pixel, a_d, r_d, g_d, b_d)
        return Image.fromarray(data, 'RGBA') \
            if form != b'\x88\x88\x00\x00' else Image.fromarray(data, 'RGBA').transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def res_tree_add(t, path):
    for node in path:
        t = t[node]


def default_tree():
    return collections.defaultdict(default_tree)


def dicts(t):
    return {k: dicts(t[k]) for k in t}


_buf = ""

def generate_res_tree(t, depth = 0):
    global _buf
    for k in t.keys():
        _buf += "{}".format(depth * "  ")
        if len(t[k]) == 0:
            _buf += "- "
        _buf += "{}".format(k)
        if len(t[k]) != 0:
            _buf += ":"
        _buf += "\n"
        depth += 1
        generate_res_tree(t[k], depth)
        depth -= 1


def build_res_yaml(filetree, f_name):
    global _buf
    _buf = ""
    ftree = default_tree()
    for file in filetree:
        path = [f_name] + file[0].split("/")
        res_tree_add(ftree, path)
    generate_res_tree(ftree)
    return _buf
