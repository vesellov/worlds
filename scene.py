import os
import yaml
import struct


_Debug = True


ENCODE = "cp1251"


# Unsigned 1b integer
def read_byte(file, count=1):
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('B', file.read(1))[0])        
    return array if count > 1 else array[0]


# Signed 1b integer
def read_char(file, count=1):
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('b', file.read(1))[0])        
    return array if count > 1 else array[0]

# Signed 2b integer
def read_short(file, count=1):
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('h', file.read(2))[0])        
    return array if count > 1 else array[0]


# Unsigned 2b integer
def read_ushort(file, count=1):
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('H', file.read(2))[0])
    return array if count > 1 else array[0]


# Signed 4b integer
def read_int(file, count=1):
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('i', file.read(4))[0])
    return array if count > 1 else array[0]


# Unsigned 4b integer
def read_uint(file, count=1):
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('I', file.read(4))[0])
    return array if count > 1 else array[0]


# Signed 4b integer
def read_int64(file, count=1):
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('q', file.read(8))[0])
    return array if count > 1 else array[0]


# Unsigned 8b integer
def read_uint64(file, count=1):
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('Q', file.read(8))[0])
    return array if count > 1 else array[0]


# Conversion between 2b half to 4b float
def half_to_float(h):
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


# Hard-float 2b real
def read_half(file, count=1):
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


# Floating-point 4b real
def read_float(file, count=1):
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('f', file.read(4))[0])
    return array if count > 1 else array[0]


# Floating-point 8b real
def read_double(file, count=1):
    array = [] 
    for _ in range(count): 
        array.append(struct.unpack('d', file.read(4))[0])        
    return array if count > 1 else array[0]


# Non-terminated char array
def read_str(file, length, code="cp1251"):
    buf = ""
    for _ in range(length):
        b = struct.unpack('c', file.read(1))[0]
        if ord(b) != 0:
            buf += b.decode(code)
    return buf


# Zero-terminated char array (C-style string)
def read_cstr(file, code="cp1251"):
    buf = ""
    while True:
        b = struct.unpack('c', file.read(1))[0]
        
        if b is None or ord(b) == 0:
            return buf
        else:
            buf += b.decode(code)


def read_filetree(file):
    magic = file.read(4)
    if magic != b'\x3C\xE2\x9C\x01':
        raise Exception("Incorrect magic!")
        return
    filetree = []
    buf = []
    table_size = read_uint(file)
    table_offset = read_uint(file)
    names_len = read_uint(file)
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
    return filetree


def unpack_res(file, filetree, file_name):
    f_name = os.path.splitext(file_name)[0]
    for element in filetree:
        name = os.path.dirname(element[0])
        dest_file_name = os.path.join(f_name, element[0])
        if os.path.isfile(dest_file_name):
            continue
        if not os.path.exists(os.path.join(f_name, name)):
            os.makedirs(os.path.join(f_name, name))
        with open(dest_file_name, "wb") as new_file:
            file.seek(element[1])
            buf = file.read(element[2])
            new_file.write(buf)


def read_bon_info(file_name):
    info = []
    if not os.path.isfile(file_name):
        return None
    with open(file_name, "rb") as file:
        file.seek(0)
        for _ in range(os.path.getsize(file_name) // 12):
            info.append(read_float(file, 3))
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


def add_child(arr, p_name, name):
    if arr[0] == p_name:
        arr[1].append([name, []])
    else:
        for bone in arr[1]:
            add_child(bone, p_name, name)


def flat_tree(tree, arr=None):
    if arr is None:
        arr = []
    arr.append(tree[0])
    if len(tree[1]) != 0:
        for leaf in tree[1]:
            flat_tree(leaf, arr)
    return arr


def walk_tree(tree, parent, visitor, arr=None):
    if arr is None:
        arr = []
    arr.append(tree[0])
    visitor(parent, tree[0])
    if len(tree[1]) != 0:
        for leaf in tree[1]:
            walk_tree(leaf, tree[0], visitor, arr)
    return arr


def read_lnk_info(file_name):
    info = []
    with open(file_name, "rb") as file:
        for _ in range(read_uint(file)):
            name_len = read_uint(file)
            name = read_str(file, name_len)
            parent_name_len = read_uint(file)
            if parent_name_len == 0:
                info = [name, []]
            else:
                parent_name = read_str(file, parent_name_len)
                add_child(info, parent_name, name)
    return info


def trilinear(val, coefs=[0, 0, 0]):
    # Linear interpolation by str
    t1 = val[0] + (val[1] - val[0]) * coefs[1]
    t2 = val[2] + (val[3] - val[2]) * coefs[1]
    # Bilinear interpolation by dex
    v1 = t1 + (t2 - t1) * coefs[0]
    # Linear interpolation by str
    t1 = val[4] + (val[5] - val[4]) * coefs[1]
    t2 = val[6] + (val[7] - val[6]) * coefs[1]
    # Bilinear interpolation by dex
    v2 = t1 + (t2 - t1) * coefs[0]
    # Trilinear interpolation by height
    return v1 + (v2 - v1) * coefs[2]


def create_geometry(part_data, coefs=[0, 0, 0]):
    vert_buf = []
    norm_buf = []
    tex_buf = []
    ind_buf = []
    for i in range(part_data[1]):
        for j in range(4):
            vert_buf.append([
                trilinear([part_data[13][i][0][k][j] for k in range(8)], coefs),
                trilinear([part_data[13][i][1][k][j] for k in range(8)], coefs),
                trilinear([part_data[13][i][2][k][j] for k in range(8)], coefs),
            ])
    for i in range(part_data[2]):
        for j in range(4):
            norm_buf.append([
                part_data[14][i][0][j],
                part_data[14][i][1][j],
                part_data[14][i][2][j],
            ])
    for i in range(part_data[3]):
        tex_buf.extend(part_data[15][i])
    for i in part_data[16]:
        ind_buf.append([
            part_data[17][i[0]][0] * 4 + part_data[17][i[0]][1],
            part_data[17][i[0]][2] * 4 + part_data[17][i[0]][3],
            part_data[17][i[0]][4],
            part_data[17][i[1]][0] * 4 + part_data[17][i[1]][1],
            part_data[17][i[1]][2] * 4 + part_data[17][i[1]][3],
            part_data[17][i[1]][4],
            part_data[17][i[2]][0] * 4 + part_data[17][i[2]][1],
            part_data[17][i[2]][2] * 4 + part_data[17][i[2]][3],
            part_data[17][i[2]][4],
        ])
    return vert_buf, norm_buf, tex_buf, ind_buf



def load_human_female(scene, coefs=[0, 0, 0]):
    # print('model_tree', model_tree)
    # print('parts_list', parts_list)

    u = UnitData()
    u.template = 'unhufa'
    u.parts = [
        'hp',
        'bd',
        'hd',
        'rh1',
        'rh2',
        'rh3',
        'lh1',
        'lh2',
        'lh3',
        'll1',
        'll2',
        'll3',
        'rl1',
        'rl2',
        'rl3',
        'hr.01',
        # 'hp.armor01',
        # 'bd.armor01',
        # 'hd.armor01',
        # 'rh1.armor01',
        # 'rh2.armor01',
        # 'rh3.armor01',
        # 'lh1.armor01',
        # 'lh2.armor01',
        # 'lh3.armor01',
        # 'll1.armor01',
        # 'll2.armor01',
        # 'll3.armor01',
        # 'rl1.armor01',
        # 'rl2.armor01',
        # 'rl3.armor01',
        # 'rh3.crbow01main',
        # 'crbow01part01',
        # 'crbow01tetiva01',
        # 'crbow01part02',
        # 'crbow01tetiva02',
        # 'rh3.arrow00',
        # 'basearrow00',
        # 'baserh3',
        # 'quiver',
        # 'arrows',
        # 'l_shell.armor01',
        # 'r_shell.armor01',
    ]

    model_tree = read_lnk_info('/Users/veselinpenev/tmp/unhufe/unhufe.lnk')
    parts_list = flat_tree(model_tree)

    # u.bones = {
    #     'hp': (0, 0, 0),
    #     'bd': (6.402842700481415e-09, -0.00930155348032713, 0.10833406448364258),
    #     'hd': (-2.444721758365631e-08, 0.002422838471829891, 0.31675976514816284),
    # }
    u.textures = {
        'hp': 'unhufeskin_00.png',
        'bd': 'unhufeskin_00.png',
        'hd': 'unhufeskin_00.png',
        'rh1': 'unhufeskin_00.png',
        'rh2': 'unhufeskin_00.png',
        'rh3': 'unhufeskin_00.png',
        'lh1': 'unhufeskin_00.png',
        'lh2': 'unhufeskin_00.png',
        'lh3': 'unhufeskin_00.png',
        'll1': 'unhufeskin_00.png',
        'll2': 'unhufeskin_00.png',
        'll3': 'unhufeskin_00.png',
        'rl1': 'unhufeskin_00.png',
        'rl2': 'unhufeskin_00.png',
        'rl3': 'unhufeskin_00.png',
        'hr.01': 'unhufeskin_00.png',
        'hp.armor01': 'unhufelg_01.ad.1.png',
        'bd.armor01': 'unhufesh_08.ad.0.png',
        'hd.armor01': 'unhufehl_01.ad.0.png',
        'rh1.armor01': 'unhufepl_01.ad.0.png',
        'rh2.armor01': 'unhufepl_01.ad.0.png',
        'rh3.armor01': 'unhufegl_08.ad.0.png',
        'lh1.armor01': 'unhufepl_01.ad.0.png',
        'lh2.armor01': 'unhufepl_01.ad.0.png',
        'lh3.armor01': 'unhufegl_08.ad.0.png',
        'll1.armor01': 'unhufelg_01.ad.0.png',
        'll2.armor01': 'unhufelg_01.ad.0.png',
        'll3.armor01': 'unhufelg_01.ad.0.png',
        'rl1.armor01': 'unhufelg_01.ad.0.png',
        'rl2.armor01': 'unhufelg_01.ad.0.png',
        'rl3.armor01': 'unhufebt_08.ad.0.png',
        # 'rh3.crbow01main',
        # 'rh3.arrow00',
        # 'crbow01part01',
        # 'crbow01tetiva01',
        # 'crbow01part02',
        # 'crbow01tetiva02',
        # 'basearrow00',
        # 'baserh3',
        # 'quiver',
        # 'arrows',
        # 'l_shell.armor01',
        # 'r_shell.armor01',
    }

    nodes_prepare = {}
    for part_name in parts_list:
        if part_name not in u.parts:
            continue
        bones_info = read_bon_info('/Users/veselinpenev/tmp/unhufe/' + part_name + '.bon')
        # part_pos = [
        #     trilinear([bones_info[i][0] for i in range(8)], coefs),
        #     trilinear([bones_info[i][1] for i in range(8)], coefs),
        #     trilinear([bones_info[i][2] for i in range(8)], coefs),
        # ]
        bone_pos = [
            bones_info[0][0],
            bones_info[0][1],
            bones_info[0][2],
        ]
        mesh_data = scene.load_fig_file(
            filename='/Users/veselinpenev/tmp/unhufe/' + part_name + '.fig',
            # bone_pos=bones_info[0],  # part_pos,
            bone_pos=bone_pos,
            texture_filename=u.textures[part_name],
        )
        nodes_prepare.update({part_name: [mesh_data, bone_pos]})
        # mesh_data.bone_coords = [bones_info[0][0], -bones_info[0][1], bones_info[0][2]]
        # mesh_data.animation_translate_coords
        # mesh_data.bone_coords = part_pos
        # mesh_data.animation_rotate_matrix = Matrix()

    def update_bone_pos(parent, child):
        if parent and parent in scene.objects and child in scene.objects:
            scene.objects[child].bone_pos_calculated = [
                scene.objects[parent].bone_pos_calculated[0] + scene.objects[child].bone_pos[0],
                scene.objects[parent].bone_pos_calculated[1] + scene.objects[child].bone_pos[1],
                scene.objects[parent].bone_pos_calculated[2] + scene.objects[child].bone_pos[2],
            ]
        # print(f'calculated bone pos for {child} from {parent}')

    scene.objects[model_tree[0]].bone_pos_calculated = [0, 0, 0]
    walk_tree(model_tree, None, update_bone_pos)


class MeshData(object):

    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.vertex_format = [
            (b'v_pos', 3, 'float'),
            (b'v_normal', 3, 'float'),
            (b'v_tc0', 2, 'float')]
        self.vertices = []
        self.indices = []
        self.material = kwargs.get('material', None)
        self.bone_pos = kwargs.get('bone_pos', [0, 0, 0])
        self.bone_pos_calculated = None
        self.bone_translate = None
        self.animation_translate = None
        self.animation_translate_coords = None
        self.animation_rotate = None
        self.animation_rotate_matrix = None

    def calculate_normals(self):
        for i in range(len(self.indices) / (3)):
            fi = i * 3
            v1i = self.indices[fi]
            v2i = self.indices[fi + 1]
            v3i = self.indices[fi + 2]
            vs = self.vertices
            p1 = [vs[v1i + c] for c in range(3)]
            p2 = [vs[v2i + c] for c in range(3)]
            p3 = [vs[v3i + c] for c in range(3)]
            u, v = [0, 0, 0], [0, 0, 0]
            for j in range(3):
                v[j] = p2[j] - p1[j]
                u[j] = p3[j] - p1[j]
            n = [0, 0, 0]
            n[0] = u[1] * v[2] - u[2] * v[1]
            n[1] = u[2] * v[0] - u[0] * v[2]
            n[2] = u[0] * v[1] - u[1] * v[0]
            for k in range(3):
                self.vertices[v1i + 3 + k] = n[k]
                self.vertices[v2i + 3 + k] = n[k]
                self.vertices[v3i + 3 + k] = n[k]


class UnitData(object):

    def __init__(self):
        self.template = None
        self.parts = []
        self.bones = {}
        self.animations = {}
        self.textures = {}


class SceneData(object):

    def __init__(self):
        self.objects = {}
        # buffers
        self._file = None
        self._object = None
        self._vertices = []
        self._normals = []
        self._texcoords = []
        self._faces = []

    def load_fig_file(self, filename, texture_filename=None, coefs=[0, 0, 0], bone_pos=[0, 0, 0]):
        name = os.path.basename(filename).replace('.fig', '')
        part_data = read_fig_info(filename)
        mesh = MeshData(
            name=name,
            material={'map_Kd': texture_filename} if texture_filename else None,
            bone_pos=bone_pos,
        )
        vert_buf = []
        norm_buf = []
        tex_buf = []
        for i in range(part_data[1]):
            for j in range(4):
                # vert_buf.append([
                #     trilinear([part_data[13][i][0][k][j] for k in range(8)], coefs),
                #     trilinear([part_data[13][i][1][k][j] for k in range(8)], coefs),
                #     trilinear([part_data[13][i][2][k][j] for k in range(8)], coefs),
                # ])
                vert_buf.append([
                    part_data[13][i][0][0][j],
                    part_data[13][i][1][0][j],
                    part_data[13][i][2][0][j],
                ])
        for i in range(part_data[2]):
            for j in range(4):
                norm_buf.append([
                    part_data[14][i][0][j],
                    part_data[14][i][1][j],
                    part_data[14][i][2][j],
                ])
        for i in range(part_data[3]):
            tex_buf.append(part_data[15][i])
        idx = 0
        d = part_data[17]
        for i in part_data[16]:
            for f in range(3):
                j = i[f]
                mesh.vertices.extend([
                    vert_buf[d[j][0] * 4 + d[j][1]][0],  #  + bone_pos[0] - part_data[9][0][0],
                    vert_buf[d[j][0] * 4 + d[j][1]][1],  #  + bone_pos[1] - part_data[9][0][1],
                    vert_buf[d[j][0] * 4 + d[j][1]][2],  #  + bone_pos[2] - part_data[9][0][2],
                    norm_buf[d[j][2] * 4 + d[j][3]][0],
                    norm_buf[d[j][2] * 4 + d[j][3]][1],
                    norm_buf[d[j][2] * 4 + d[j][3]][2],
                    tex_buf[d[j][4]][0],
                    tex_buf[d[j][4]][1],
                ])
            mesh.indices.extend([idx, idx + 1, idx + 2])
            idx += 3
        self.objects[name] = mesh
        print(f'loaded {name} with {idx} faces')
        return mesh

    def load_obj_file(self, filename, swapyz=False, shift=None, texture_filename=None):
        # buffers
        self._vertices = []
        self._normals = []
        self._texcoords = []
        self._faces = []
        self._file = os.path.basename(filename).replace('.obj', '')
        _dir = os.path.dirname(filename)
        self._object = None
        self._material = None
        self._mtl = None
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            if line.startswith('s'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'o':
                self._finish_object(texture_filename=texture_filename)
                self._object = values[1]
            elif values[0] == 'mtllib':
                self._mtl = self._load_mtl(os.path.join(_dir, values[1]))
            elif values[0] in ('usemtl', 'usemat'):
                self._material = values[1]
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                if shift:
                    v = v[0] + shift[0], v[1] + shift[1], v[2] + shift[2]
                self._vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self._normals.append(v)
            elif values[0] == 'vt':
                self._texcoords.append(list(map(float, values[1:3])))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(-1)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(-1)
                self._faces.append((face, norms, texcoords))
        self._finish_object(texture_filename=texture_filename)
        self._mtl = None
        self._material = None
        self._file = None
        self._dir = None

    def _load_mtl(self, filename):
        contents = {}
        mtl = None
        if os.path.isfile(filename):
            for line in open(filename, "r"):
                if not line.strip():
                    continue
                if line.startswith('#'):
                    continue
                key, _, value = line.partition(' ')
                if not value:
                    continue
                value = value.strip()
                if key == 'newmtl':
                    mtl = value
                    contents[value] = {}
                elif mtl is None:
                    raise ValueError("mtl file doesn't start with newmtl stmt")
                else:
                    contents[mtl][key] = value
        return contents

    def _finish_object(self, texture_filename=None):
        if self._object is None:
            return
        mat = None
        if texture_filename:
            mat = {'map_Kd': texture_filename}
        else:
            if self._mtl and self._material and self._mtl.get(self._material).get('map_Kd'):
                mat = self._mtl.get(self._material)
        if self._object:
            name = self._file + '#' + self._object
        else:
            name = self._file
        mesh = MeshData(name=name, material=mat)
        idx = 0
        for f in self._faces:
            verts = f[0]
            norms = f[1]
            tcs = f[2]
            for i in range(3):
                # get normal components
                n = (0.0, 0.0, 0.0)
                if norms[i] != -1:
                    n = self._normals[norms[i] - 1]
                # get texture coordinate components
                t = (0.0, 0.0)
                if tcs[i] != -1:
                    t = self._texcoords[tcs[i] - 1]
                # get vertex components
                v = self._vertices[verts[i] - 1]
                data = [v[0], v[1], v[2], n[0], n[1], n[2], t[0], t[1]]
                mesh.vertices.extend(data)
            tri = [idx, idx + 1, idx + 2]
            mesh.indices.extend(tri)
            idx += 3
        self.objects[name] = mesh
        # mesh.calculate_normals()
        self._vertices = []
        self._normals = []
        self._texcoords = []
        self._faces = []
        self._object = None
        self._file = None
