import os
import json
import struct
import collections
import numpy as np
from PIL import Image

_Debug = True


ENCODE = "cp1251"


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


def read_bon_info(file_name):
    info = []
    if not os.path.isfile(file_name):
        return None
    with open(file_name, "rb") as file:
        file.seek(0)
        for _ in range(os.path.getsize(file_name) // 12):
            info.append(read_float(file, 3))
    return info


def read_anm_info(file_name):
    info = []
    with open(file_name, "rb") as file:
        info.append(read_uint(file))
        info.append([read_float(file, 4) for _ in range(info[0])])
        info.append(read_uint(file))
        info.append([read_float(file, 3) for _ in range(info[2])])
        info.extend(read_uint(file, 2))
        if info[4] != 0 and info[5] != 0:
            info.append([[
                read_float(file, 3) for _ in range(info[5])] for _ in range(info[4])])
    return info


def read_fig_info(file_name):
    info = []
    if not os.path.isfile(file_name):
        return None
    with open(file_name, "rb") as file:
        if file.read(3) != b'\x46\x49\x47':
            raise Exception("Incorrect magic!")
        n = read_byte(file) - 48
        info.append(n)
        info.extend(read_uint(file, 6))
        read_uint(file)
        info.extend(read_uint(file, 2))
        info[4] //= 3
        for _ in range(3):
            info.append([read_float(file, 3) for _ in range(n)])
        info.append(read_float(file, n))
        # Vertex blocks
        info.append([[[read_float(file, 4) for _ in range(n)] for _ in range(3)] for _ in range(info[1])])
        # Normals
        info.append([[read_float(file, 4) for _ in range(4)] for _ in range(info[2])])
        # UV
        info.append([read_float(file, 2) for _ in range(info[3])])
        # Indices
        info.append([read_ushort(file, 3) for _ in range(info[4])])
        # Vertex components
        info.append([])
        for _ in range(info[5]):
            buf = read_ushort(file, 3)
            info[-1].append([buf[0] >> 2, buf[0] & 3, buf[1] >> 2, buf[1] & 3, buf[2]])
        # Deformation
        info.append([read_ushort(file, 2) for _ in range(info[6])])
    return info


def read_mod_info(file_name):
    mod_name = file_name.replace('.mod', '')
    with open(file_name, "rb") as file:
        tree = read_res_filetree(file)
        # f_name = os.path.splitext(os.path.basename(file_name))[0]
        for element in tree:
            element[0] += ".fig" if mod_name != element[0] else ".lnk"
        # unpack_res(file, tree, file_name)
    return tree


def unpack_mod_info(file_name, destination_dir):
    with open(file_name, "rb") as file:
        tree = read_res_filetree(file)
        f_name = os.path.splitext(os.path.basename(file_name))[0]
        for element in tree:
            element[0] += ".fig" if f_name != element[0] else ".lnk"
        unpack_res(file, tree, destination_dir)
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


def read_lnk_info(file_name):
    lst = []
    parents = {}
    tree = {}
    childs = {}
    with open(file_name, "rb") as file:
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


def unpack_res(file, filetree, destination_dir):
    for element in filetree:
        dest_file_name = os.path.join(destination_dir, element[0])
        if os.path.isfile(dest_file_name):
            continue
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        with open(dest_file_name, "wb") as new_file:
            file.seek(element[1])
            buf = file.read(element[2])
            new_file.write(buf)


def unpack_res_element(file, element, dest_file_name=None):
    if not dest_file_name:
        dest_file_name = element[0]
    if os.path.isfile(dest_file_name):
        return dest_file_name
    with open(dest_file_name, "wb") as new_file:
        file.seek(element[1])
        buf = file.read(element[2])
        new_file.write(buf)
    return dest_file_name


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


def read_mmp(file_name):
    with open(file_name, "rb") as file:
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
