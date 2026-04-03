import os
import sys
import json
import numpy as np

from kivy.core.image import Image
from kivy.cache import Cache
from kivy.resources import resource_find

import res
import mth


_Debug = True


class MeshData(object):

    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.object_name = None
        self.object_part_name = None
        self.coefs = [0, 0, 0]
        self.center = []
        self.min = []
        self.max = []
        self.radius = []
        self.vertices = []
        self.indices = []
        self.material = kwargs.get('material', None)


class MeshTransformData(object):

    def __init__(self):
        self.part_translate = None
        self.part_rotate = None


class ObjectPartAnimationData(object):

    def __init__(self):
        self.frames = 0
        self.rotation_frames_input = []
        self.translation_frames_input = []
        self.morphing_frames_input = []
        self.rotation_frames = []
        self.translation_frames = []
        self.morphing_frames = []

    def duplicate(self):
        d = ObjectPartAnimationData()
        d.frames = self.frames
        d.rotation_frames_input = self.rotation_frames_input.copy()
        d.translation_frames_input = self.translation_frames_input.copy()
        d.morphing_frames_input = self.morphing_frames_input.copy()
        d.rotation_frames = self.rotation_frames.copy()
        d.translation_frames = self.translation_frames.copy()
        d.morphing_frames = self.morphing_frames.copy()
        return d


class ObjectAnimationData(object):

    def __init__(self, template, animation):
        self.template = template
        self.animation = animation
        self.parts = {}


class ObjectData(object):

    def __init__(self, name, static=True):
        self.name = name
        self.static = static
        self.template = None
        self.meshes = {}
        self.parts = []
        self.parts_tree = {}
        self.parts_tree_ordered = []
        self.bones = {}
        self.parts_parents = {}
        self.textures = {}
        self.root_part_name = None
        self.root_mesh_name = None
        self.root_mesh_center = None
        self.animations = {}
        self.animations_loaded = []

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
            # if _Debug:
            #     print(f'    calculated {count} animations for [{part_name}]')

        self.walk_parts_ordered(_part_visitor)


class ModelData(object):

    def __init__(self, **kwargs):
        self.links = {}
        self.figures = {}
        self.bones = {}
        self.animations = {}

    def scan_figure_data(self, figures_res_file_path):
        items = {}
        with open(figures_res_file_path, 'rb') as figures_file:
            res_filetree_dict = res.read_res_filetree(figures_file, return_dict=True)
            for k in res_filetree_dict.keys():
                ext = k[-4:]
                if ext not in items:
                    items[ext] = []
                items[ext].append(k[:-4])
        return items

    def unpack_texture(self, texture_res_file_path, destination_dir, name):
        with open(texture_res_file_path, 'rb') as texture_file:
            res_filetree_dict = res.read_res_filetree(texture_file, return_dict=True)
            for k in res_filetree_dict.keys():
                if k.endswith('.mmp') and k[:-4].lower() == name.lower():
                    mmp_file_path = os.path.join(destination_dir, name.lower() + '.mmp')
                    png_file_path = os.path.join(destination_dir, name.lower() + '.png')
                    res.unpack_res_element(texture_file, res_filetree_dict[k], dest_file_name=mmp_file_path)
                    pil_img = res.read_mmp(mmp_file_path)
                    pil_img.save(png_file_path)
                    os.remove(mmp_file_path)
                    return png_file_path
        return None

    def unpack_figure_data(self, figures_res_file_path, destination_dir, template, selected_parts=[], selected_animations=[], save_json=False):
        destination_sub_dir = os.path.join(destination_dir, template)
        if not os.path.isdir(destination_sub_dir):
            os.makedirs(destination_sub_dir)
        lnk_count = 0
        fig_count = 0
        bon_count = 0
        anm_count = 0
        with open(figures_res_file_path, 'rb') as figures_file:
            res_filetree_dict = res.read_res_filetree(figures_file, return_dict=True)
            res_mod_element = res_filetree_dict.get(template + '.mod')
            if res_mod_element:
                mod_file_name = res.unpack_res_element(figures_file, res_mod_element, dest_file_name=os.path.join(destination_sub_dir, template + '.mod'))
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
            res_anm_element = res_filetree_dict.get(template + '.anm')
            if res_anm_element:
                anm_file_name = res.unpack_res_element(figures_file, res_anm_element, dest_file_name=os.path.join(destination_sub_dir, template + '.anm'))
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
            res_bon_element = res_filetree_dict.get(template + '.bon')
            if res_bon_element:
                bon_file_name = res.unpack_res_element(figures_file, res_bon_element, dest_file_name=os.path.join(destination_sub_dir, template + '.bon'))
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
            print(f'for model {{{template}}} unpacked {lnk_count} links, {fig_count} figures, {bon_count} bones and {anm_count} animations')
        if save_json:
            dest_json_file_path = os.path.join(destination_dir, template + '.json')
            open(dest_json_file_path, 'wt').write(json.dumps({
                'template': template,
                'figures': self.figures,
                'links': self.links,
                'animations': self.animations,
                'bones': self.bones,
            }, indent=2))
            if _Debug:
                print(f'for model {{{template}}} saved {lnk_count} links, {fig_count} figures, {bon_count} bones and {anm_count} animations to {dest_json_file_path}')


class LandData(object):

    def __init__(self):
        self.width = None
        self.height = None
        self.elevation_map_data = {}
        self.tiles_map_data = {}
        self.tiles_textures_dir_path = None
        self.plants_map_data = {}
        self.plants_variants = {}

    def load_tilemap_file(self, tilemap_file_name):
        im = Image(tilemap_file_name, keep_data=True)
        if self.width is not None and self.width != im.width:
            raise ValueError(f'land width mismatch: expected {self.width}, got {im.width}')
        if self.height is not None and self.height != im.height:
            raise ValueError(f'land height mismatch: expected {self.height}, got {im.height}')
        data = im.image._data[0]
        size = 3 if data.fmt in ('rgb', 'bgr') else 4
        for x in range(self.width):
            for y in range(self.height):
                index = y * data.width * size + x * size
                raw = bytearray(data.data[index:index + size])
                color = [int(c) for c in raw]
                bgr_flag = False
                if data.fmt == 'argb':
                    color.reverse()  # bgra
                    bgr_flag = True
                elif data.fmt == 'abgr':
                    color.reverse()  # rgba
                # conversion for BGR->RGB, BGRA->RGBA format
                if bgr_flag or data.fmt in ('bgr', 'bgra'):
                    color[0], color[2] = color[2], color[0]
                catalog_id = color[0] + color[1] * 256
                rotate = color[2] * 90
                self.tiles_map_data[(x, y)] = (catalog_id, rotate)
        return self.width, self.height

    def load_heightmap_file(self, heightmap_file_name, sea_level=0.0):
        im = Image(heightmap_file_name, keep_data=True)
        self.width = im.width
        self.height = im.height
        for x in range(self.width):
            for y in range(self.height):
                e = float(im.read_pixel(x, y)[0])
                if e < sea_level:
                    e = sea_level
                self.elevation_map_data[(x, y)] = e
        return self.width, self.height

    def load_cache_tiles_textures(self, textures_dir_path):
        self.tiles_textures_dir_path = textures_dir_path
        for file_name in os.listdir(self.tiles_textures_dir_path):
            if not file_name.endswith('.png'):
                continue
            file_path = os.path.join(self.tiles_textures_dir_path, file_name)
            file_path_source = resource_find(file_path)
            if file_path_source:
                _tex = Cache.get('kv.texture', file_path)
                if not _tex:
                    _tex = Image(file_path_source).texture
                    Cache.append('kv.texture', file_path, _tex)

    def load_plants_data(self, plants_data_file_name):
        plants_list = json.loads(open(plants_data_file_name, 'rt').read())
        for plant_coded in plants_list:
            variant, w, h, direction = plant_coded.split(' ')
            template, texture, c1, c2, c3 = variant.split(':')
            plant = {}
            plant['k'] = variant
            plant['m'] = template
            plant['t'] = texture
            plant['c'] = mth.quantize_coefs([float(c1), float(c2), float(c3)])
            if variant not in self.plants_variants:
                variant = dict(plant)
                variant['so'] = None
                self.plants_variants[plant['k']] = variant
            int_w = int(float(w))
            int_h = int(float(h))
            shift_w = float(w) - float(int_w)
            shift_h = float(h) - float(int_h)
            plant['w'] = int_w
            plant['h'] = int_h
            plant['sw'] = shift_w
            plant['sh'] = shift_h
            plant['d'] = direction
            if (int_w, int_h) not in self.plants_map_data:
                self.plants_map_data[(int_w, int_h)] = []
            self.plants_map_data[(int_w, int_h)].append(plant)

    def save_elevation_memmap(self, file_name_prefix, destination_dir):
        file_path = os.path.join(destination_dir, f'{file_name_prefix}.{self.width}.{self.height}.memmap')
        fp = np.memmap(file_path, dtype='float32', mode='w+', shape=(self.width, self.height))
        fp[:] = self.elevation_map_data
        fp.flush()
        return file_path

    def get_elevation(self, w, h):
        _w = w
        _h = h
        if w < 0:
            _w = w + self.width
        if w >= self.width:
            _w = w - self.width
        if h < 0:
            _h = h + self.height
        if h >= self.height:
            _h = h - self.height
        return self.elevation_map_data[(_w, _h)]

    def get_texture(self, w, h):
        _w = w
        _h = h
        if w < 0:
            _w = w + self.width
        if w >= self.width:
            _w = w - self.width
        if h < 0:
            _h = h + self.height
        if h >= self.height:
            _h = h - self.height
        catalog_id, rotate = self.tiles_map_data[(_w, _h)]
        texture_file_path = os.path.join(self.tiles_textures_dir_path, f'{catalog_id:05d}.png')
        return texture_file_path, rotate


def main():
    cmd = sys.argv[1]
    if cmd == 'list_models':
        md = ModelData()
        st = md.scan_figure_data(sys.argv[2])
        print('\n'.join(sorted(st['mod'])))
    elif cmd == 'list_figures':
        md = ModelData()
        st = md.scan_figure_data()
        print('\n'.join(sorted(st['fig'])))
    elif cmd == 'unpack_models':
        md = ModelData()
        st = md.scan_figure_data(sys.argv[2])
        lst = sorted(st['mod'])
        for m in lst:
            print(f'loading {m}')
            try:
                md.unpack_figure_data(sys.argv[2], destination_dir='models', template=m)
            except:
                pass
            

if __name__ == '__main__':
    main()
