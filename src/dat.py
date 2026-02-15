import os
import json
import time
import numpy as np

from kivy.core.image import Image

import res
import mth


_Debug = True


_NextUnitID = 0
_NextMeshID = 0


class MeshData(object):

    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.unit_name = None
        self.unit_part_name = None
        self.onstage = False
        self.vertices = []
        self.indices = []
        self.material = kwargs.get('material', None)
        self.part_translate = None
        self.part_animate = None


class UnitPartAnimationData(object):

    def __init__(self):
        self.frames = 0
        self.rotation_frames_input = []
        self.translation_frames_input = []
        self.morphing_frames_input = []
        self.rotation_frames = []
        self.translation_frames = []
        self.morphing_frames = []

    def duplicate(self):
        d = UnitPartAnimationData()
        d.frames = self.frames
        d.rotation_frames_input = self.rotation_frames_input.copy()
        d.translation_frames_input = self.translation_frames_input.copy()
        d.morphing_frames_input = self.morphing_frames_input.copy()
        d.rotation_frames = self.rotation_frames.copy()
        d.translation_frames = self.translation_frames.copy()
        d.morphing_frames = self.morphing_frames.copy()
        return d


class UnitAnimationData(object):

    def __init__(self, template, animation):
        self.template = template
        self.animation = animation
        self.parts = {}


class UnitData(object):

    def __init__(self):
        self.name = None
        self.template = None
        self.meshes = {}
        self.parts = []
        self.parts_tree = {}
        self.parts_tree_ordered = []
        self.bones = {}
        self.parts_parents = {}
        self.textures = {}
        self.animations = {}
        self.animations_loaded = []
        self.animation_playing = None
        self.animation_frame = 0
        self.onstage = False

    def list_parents(self, part_name):
        if part_name not in self.parts_parents:
            return []
        parents = []
        current_part = part_name
        while current_part and current_part in self.parts_parents:
            next_parent = self.parts_parents[current_part]
            if next_parent:
                parents.insert(0, next_parent)
            current_part = next_parent
        return parents

    def walk_parts(self, visitor_before, visitor_after=None, tree=None):
        if tree is None:
            tree = self.parts_tree
        for part_name, other_parts in tree.items():
            if part_name in self.parts:
                visitor_before(self, part_name)
                self.walk_parts(visitor_before, visitor_after, other_parts)
                if visitor_after:
                    visitor_after(self, part_name)

    def walk_parts_ordered(self, visitor, ordered_tree=None, parent_part_name=None):
        if ordered_tree is None:
            ordered_tree = self.parts_tree_ordered
        this_part_name = ordered_tree[0]
        if this_part_name not in self.parts:
            return
        this_part_branches = ordered_tree[1]
        visitor(this_part_name, parent_part_name)
        if this_part_branches:
            for branch in this_part_branches:
                self.walk_parts_ordered(visitor, ordered_tree=branch, parent_part_name=this_part_name)

    def walk_parts_before_after(self, visitor_before, visitor_after, ordered_tree=None, parent_part_name=None):
        if ordered_tree is None:
            ordered_tree = self.parts_tree_ordered
        this_part_name = ordered_tree[0]
        if this_part_name not in self.parts:
            return
        this_part_branches = ordered_tree[1]
        visitor_before(this_part_name, parent_part_name)
        if this_part_branches:
            for branch in this_part_branches:
                self.walk_parts_before_after(visitor_before, visitor_after, ordered_tree=branch, parent_part_name=this_part_name)
        visitor_after(this_part_name, parent_part_name)

    def calculate_animations(self):

        def _part_visitor(part_name, parent_part_name):
            bone_t = [0, 0, 0]
            if part_name in self.bones:
                bone_t = self.bones[part_name]
            count = 0
            for anim_name in self.animations_loaded:
                a = self.animations[anim_name]
                if part_name not in a.parts:
                    a.parts[part_name] = a.parts[parent_part_name].duplicate()
                    continue
                part_a = a.parts[part_name]
                part_rotation_frames_input = part_a.rotation_frames_input
                part_translation_frames_input = part_a.translation_frames_input
                if parent_part_name:
                    parent_part_a = a.parts[parent_part_name]
                    part_rotation_frames_calculated = []
                    part_translation_frames_calculated = []
                    for i in range(part_a.frames):
                        parent_q = parent_part_a.rotation_frames[i]
                        parent_t = parent_part_a.translation_frames[i]                        
                        part_q = part_rotation_frames_input[i]
                        final_q = mth.quaternion_multiply(part_q, parent_q)
                        final_t = mth.vec3sum(mth.quaternion_by_vector(parent_q, bone_t), parent_t)
                        part_rotation_frames_calculated.append(final_q)
                        part_translation_frames_calculated.append(final_t)
                    part_a.rotation_frames = part_rotation_frames_calculated
                    part_a.translation_frames = part_translation_frames_calculated
                else:
                    part_rotation_frames_calculated = []
                    part_translation_frames_calculated = []
                    for i in range(part_a.frames):
                        part_q = part_rotation_frames_input[i]
                        part_t = part_translation_frames_input[i]
                        part_rotation_frames_calculated.append(part_q)
                        part_translation_frames_calculated.append(mth.vec3sum(bone_t, part_t))
                    part_a.rotation_frames = part_rotation_frames_calculated
                    part_a.translation_frames = part_translation_frames_calculated
                part_a.rotation_frames_input = []
                part_a.translation_frames_input = []
                count += 1
            if _Debug:
                print(f'    calculated {count} animations for [{part_name}]')

        self.walk_parts_ordered(_part_visitor)


class ModelData(object):

    def __init__(self, template, **kwargs):
        self.template = template
        self.links = {}
        self.figures = {}
        self.bones = {}
        self.animations = {}

    def unpack_figure_data(self, figures_res_file_path, destination_dir, selected_parts=[], selected_animations=[], save_json=False):
        destination_sub_dir = os.path.join(destination_dir, self.template)
        if not os.path.isdir(destination_sub_dir):
            os.makedirs(destination_sub_dir)
        lnk_count = 0
        fig_count = 0
        bon_count = 0
        anm_count = 0
        with open(figures_res_file_path, 'rb') as figures_file:
            res_filetree_dict = res.read_res_filetree(figures_file, return_dict=True)
            res_mod_element = res_filetree_dict.get(self.template + '.mod')
            if res_mod_element:
                mod_file_name = res.unpack_res_element(figures_file, res_mod_element, dest_file_name=os.path.join(destination_sub_dir, self.template + '.mod'))
                mod_filetree = res.unpack_mod_info(mod_file_name, destination_dir=destination_sub_dir)
                for mod_element in mod_filetree:
                    el = mod_element[0][:-4]
                    if mod_element[0].endswith('.fig'):
                        if not selected_parts or el in selected_parts:
                            fig_file_name = os.path.join(destination_sub_dir, mod_element[0])
                            self.figures[mod_element[0][:-4]] = res.read_fig_info(fig_file_name)
                            fig_count += 1
                    elif mod_element[0].endswith('.lnk'):
                        lnk_file_name = os.path.join(destination_sub_dir, mod_element[0])
                        lnk_list, lnk_tree, lnk_parents, _ = res.read_lnk_info(lnk_file_name)
                        lnk_count += 1
                        self.links[mod_element[0][:-4]] = {
                            'ordered': lnk_list,
                            'tree': lnk_tree,
                            'parents': lnk_parents,
                        }
            res_anm_element = res_filetree_dict.get(self.template + '.anm')
            if res_anm_element:
                anm_file_name = res.unpack_res_element(figures_file, res_anm_element, dest_file_name=os.path.join(destination_sub_dir, self.template + '.anm'))
                with open(anm_file_name, 'rb') as anm_file:
                    anm_filetree = res.read_res_filetree(anm_file)
                    for anm_element in anm_filetree:
                        if not os.path.isdir(os.path.join(destination_sub_dir, anm_element[0])):
                            os.makedirs(os.path.join(destination_sub_dir, anm_element[0]))
                        anm_element[0] += '.anm'
                    res.unpack_res(anm_file, anm_filetree, destination_dir=destination_sub_dir)
                    for anm_element in anm_filetree:
                        el = anm_element[0][:-4]
                        if not selected_animations or el in selected_animations:
                            one_anm_file_name = os.path.join(destination_sub_dir, anm_element[0])
                            with open(one_anm_file_name, 'rb') as one_anm_file:
                                one_anm_filetree = res.read_res_filetree(one_anm_file)
                                self.animations[anm_element[0][:-4]] = {}
                                for one_anm_element in one_anm_filetree:
                                    el_part = one_anm_element[0]
                                    if not selected_parts or el_part in selected_parts:
                                        one_anm_dest_file_name = os.path.join(destination_sub_dir, anm_element[0][:-4], one_anm_element[0] + '.anm')
                                        res.unpack_res_element(one_anm_file, one_anm_element, dest_file_name=one_anm_dest_file_name)
                                        self.animations[anm_element[0][:-4]][one_anm_element[0]] = res.read_anm_info(one_anm_dest_file_name)
                                        anm_count += 1
            res_bon_element = res_filetree_dict.get(self.template + '.bon')
            if res_bon_element:
                bon_file_name = res.unpack_res_element(figures_file, res_bon_element, dest_file_name=os.path.join(destination_sub_dir, self.template + '.bon'))
                with open(bon_file_name, 'rb') as bon_file:
                    bon_filetree = res.read_res_filetree(bon_file)
                    for bon_element in bon_filetree:
                        bon_element[0] += '.bon'
                    res.unpack_res(bon_file, bon_filetree, destination_dir=destination_sub_dir)
                    for bon_element in bon_filetree:
                        one_bon_file_name = os.path.join(destination_sub_dir, bon_element[0])
                        self.bones[bon_element[0][:-4]] = res.read_bon_info(one_bon_file_name)
                        bon_count += 1
        if _Debug:
            print(f'for model {{{self.template}}} unpacked {lnk_count} links, {fig_count} figures, {bon_count} bones and {anm_count} animations')
        if save_json:
            dest_json_file_path = os.path.join(destination_dir, self.template + '.json')
            open(dest_json_file_path, 'wt').write(json.dumps({
                'template': self.template,
                'figures': self.figures,
                'links': self.links,
                'animations': self.animations,
                'bones': self.bones,
            }, indent=2))
            if _Debug:
                print(f'for model {{{self.template}}} saved {lnk_count} links, {fig_count} figures, {bon_count} bones and {anm_count} animations to {dest_json_file_path}')


class LandData(object):

    def __init__(self):
        self.width = None
        self.height = None
        self.elevation_map_data = None
        self.tiles_texture_map = {}

    def load_elevation_file(self, elevation_file_name):
        im = Image(elevation_file_name, keep_data=True)
        self.width = im.width
        self.height = im.height
        _elevation = []
        # for h in range(self.height):
        #     _elevation.append(tuple(im.read_pixel(w, h)[0] for w in range(self.width)))
        for w in range(self.width):
            _elevation.append(tuple(im.read_pixel(w, h)[0] for h in range(self.height)))
        self.elevation_map_data = tuple(_elevation)
        return self.width, self.height

    def save_elevation_memmap(self, file_name_prefix, destination_dir):
        file_path = os.path.join(destination_dir, f'{file_name_prefix}.{self.width}.{self.height}.memmap')
        fp = np.memmap(file_path, dtype='float32', mode='w+', shape=(self.width, self.height))
        fp[:] = self.elevation_map_data
        fp.flush()
        return file_path

    def get_elevation(self, w, h):
        _w = w
        if w < 0:
            _w = -w 
        elif w >= self.width:
            _w = self.width - w
        _h = h
        if h < 0:
            _h = -h
        elif h >= self.height:
            _h = self.height - h
        return self.elevation_map_data[_w][_h]


class SceneData(object):

    def __init__(self, land):
        self.units = {}
        self.meshes = {}
        self.land = land
        self.models = {}

    def add_model(self, model):
        self.models[model.template] = model

    def create_mesh_from_fig_data(self, fig_data, prefix='', texture_filename=None, coefs=[0, 0, 0]):
        global _NextMeshID
        _NextMeshID += 1
        name = prefix + '_' + str(_NextMeshID)
        mesh = MeshData(
            name=name,
            material={'map_Kd': texture_filename} if texture_filename else None,
        )
        vert_buf = []
        norm_buf = []
        tex_buf = []
        for i in range(fig_data[1]):
            for j in range(4):
                vert_buf.append(mth.ei2xyz_list([
                    mth.trilinear([fig_data[13][i][0][k][j] for k in range(8)], coefs),
                    mth.trilinear([fig_data[13][i][1][k][j] for k in range(8)], coefs),
                    mth.trilinear([fig_data[13][i][2][k][j] for k in range(8)], coefs),
                ]))
        for i in range(fig_data[2]):
            for j in range(4):
                norm_buf.append(mth.ei2xyz_list([
                    fig_data[14][i][0][j],
                    fig_data[14][i][1][j],
                    fig_data[14][i][2][j],
                ]))
        for i in range(fig_data[3]):
            tex_buf.append(fig_data[15][i])
        idx = 0
        d = fig_data[17]
        for i in fig_data[16]:
            for f in range(3):
                j = i[f]
                mesh.vertices.extend([
                    vert_buf[d[j][0] * 4 + d[j][1]][0],
                    vert_buf[d[j][0] * 4 + d[j][1]][1],
                    vert_buf[d[j][0] * 4 + d[j][1]][2],
                    norm_buf[d[j][2] * 4 + d[j][3]][0],
                    norm_buf[d[j][2] * 4 + d[j][3]][1],
                    norm_buf[d[j][2] * 4 + d[j][3]][2],
                    tex_buf[d[j][4]][0],
                    tex_buf[d[j][4]][1],
                ])
            mesh.indices.extend([idx, idx + 1, idx + 2])
            idx += 3
        self.meshes[name] = mesh
        if _Debug:
            print(f'  prepared mesh <{name}> with {idx} faces')
        return mesh
    
    def create_unit_from_model_data(self, template, coefs=[0, 0, 0], selected_parts=[], excluded_parts=[], selected_animations=[], textures={'*': 'default.png'}):
        global _NextUnitID
        _NextUnitID += 1
        m = self.models[template]
        u = UnitData()
        u.template = template
        u.name = template + str(_NextUnitID)
        u.textures = textures
        u.parts_tree_ordered = m.links[template]['ordered']
        u.parts_tree = m.links[template]['tree']
        u.parts_parents = m.links[template]['parents']
        ordered_parts_list = res.flat_tree(u.parts_tree_ordered)
        u.animations_loaded = selected_animations
        if not u.animations_loaded:
            u.animations_loaded = list(m.animations.keys())
        related_meshes = {}
        if not selected_parts:
            selected_parts = ordered_parts_list
        for exclude in excluded_parts:
            if exclude in selected_parts:
                selected_parts.remove(exclude)
        if _Debug:
            print(f'about to prepare unit ({u.name}) with {len(selected_parts)} parts and {len(u.animations_loaded)} animations from model {{{template}}}')
        t1 = time.time()
        for part_name in selected_parts:
            u.parts.append(part_name)
            part_info = m.bones[part_name]
            u.bones[part_name] = mth.ei2xyz_list([
                mth.trilinear([part_info[i][0] for i in range(8)], coefs),
                mth.trilinear([part_info[i][1] for i in range(8)], coefs),
                mth.trilinear([part_info[i][2] for i in range(8)], coefs),
            ])
            mesh = self.create_mesh_from_fig_data(
                fig_data=m.figures[part_name],
                prefix=u.name + '_' + part_name,
                texture_filename=u.textures[part_name] if part_name in u.textures else u.textures['*'],
                coefs=coefs,
            )
            mesh.unit_name = u.name
            mesh.unit_part_name = part_name
            u.meshes[part_name] = mesh
            related_meshes[part_name] = mesh.name
            for anim_name in u.animations_loaded:
                if part_name not in m.animations[anim_name]:
                    continue
                animation_info = m.animations[anim_name][part_name]
                if anim_name not in u.animations:
                    u.animations[anim_name] = UnitAnimationData(template, anim_name)
                a = UnitPartAnimationData()
                a.rotation_frames_input = [mth.ei2quad_list(quad) for quad in animation_info[1]]
                a.translation_frames_input = [mth.ei2xyz_list(coord) for coord in animation_info[3]]
                morphing_frames = []
                if animation_info[4] != 0 and animation_info[5] != 0:
                    for value in animation_info[6]:
                        morphing_frames.append([])
                        for i in range(animation_info[5]):
                            morphing_frames[0].append(mth.ei2xyz_list(value[i]))
                    a.morphing_frames_input = morphing_frames
                a.frames = len(a.rotation_frames_input)
                u.animations[anim_name].parts[part_name] = a

        u.calculate_animations()
        self.units[u.name] = u
        t2 = time.time()
        if _Debug:
            print(f'unit ({u.name}) created in {t2 - t1} sec')
        return u
