"""
Data of the game loaded into memory: meshes, animations and models of the objects, the catalog of figures,
armors and weapons, and the land of the biome. Run as a script to list the content of the .res files.
"""

#------------------------------------------------------------------------------

_Debug = True

#------------------------------------------------------------------------------

import os
import sys
import json
import traceback
import pprint

import numpy as np
from PIL import Image as PILImage
from kivy.core.image import Image
from kivy.cache import Cache
from kivy.resources import resource_find

#------------------------------------------------------------------------------

from lib import const
from lib import res
from lib import dress
from lib import mth
from lib import plants
from lib import stores
from lib import terrain
from lib import zone

#------------------------------------------------------------------------------

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
        self.texture = kwargs.get('texture', {})

#------------------------------------------------------------------------------


class MeshTransformData(object):

    def __init__(self):
        self.part_translate = None
        self.part_rotate = None

#------------------------------------------------------------------------------


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

#------------------------------------------------------------------------------


class ObjectAnimationData(object):

    def __init__(self, template, animation):
        self.template = template
        self.animation = animation
        self.parts = {}

#------------------------------------------------------------------------------


class ObjectData(object):

    def __init__(self, name, static=True):
        self.name = name
        self.static = static
        self.template = None
        self.meshes = {}
        self.parts = []
        # self.parts_tree = {}
        self.parts_tree_ordered = []
        self.bones = {}
        # self.parts_parents = {}
        self.textures = {}
        self.root_part_name = None
        self.root_mesh_name = None
        self.root_mesh_center = None
        self.animations = {}
        self.animations_loaded = []
        # heights of the lowest and of the highest point of the figure above its root part,
        # see scen.Scene.figure_height_range()
        self.ground_offset = 0.0
        self.top_offset = 0.0

    # def list_parents(self, part_name):
    #     if part_name not in self.parts_parents:
    #         return []
    #     parents = []
    #     current_part = part_name
    #     while current_part and current_part in self.parts_parents:
    #         next_parent = self.parts_parents[current_part]
    #         if next_parent:
    #             parents.insert(0, next_parent)
    #         current_part = next_parent
    #     return parents

    # def walk_parts(self, visitor_before, visitor_after=None, tree=None):
    #     if tree is None:
    #         tree = self.parts_tree
    #     for part_name, other_parts in tree.items():
    #         if part_name in self.parts:
    #             visitor_before(self, part_name)
    #             self.walk_parts(visitor_before, visitor_after, other_parts)
    #             if visitor_after:
    #                 visitor_after(self, part_name)

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

    # def walk_parts_before_after(self, visitor_before, visitor_after, ordered_tree=None, parent_part_name=None):
    #     if ordered_tree is None:
    #         ordered_tree = self.parts_tree_ordered
    #     this_part_name = ordered_tree[0]
    #     if this_part_name not in self.parts:
    #         return
    #     this_part_branches = ordered_tree[1]
    #     visitor_before(this_part_name, parent_part_name)
    #     if this_part_branches:
    #         for branch in this_part_branches:
    #             self.walk_parts_before_after(visitor_before, visitor_after, ordered_tree=branch, parent_part_name=this_part_name)
    #     visitor_after(this_part_name, parent_part_name)

    def calculate_animations(self):
        anim_part_y_shift = {}

        def _part_visitor(part_name, parent_part_name):
            bone_t = [0, 0, 0]
            if part_name in self.bones:
                bone_t = self.bones[part_name]
            count = 0
            for anim_name in self.animations_loaded:
                if anim_name not in anim_part_y_shift:
                    anim_part_y_shift[anim_name] = 0
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

#------------------------------------------------------------------------------


class ModelData(object):

    def __init__(self, **kwargs):
        self.links = {}
        self.figures = {}
        self.bones = {}
        self.animations = {}
        # template -> coefs of the body variant the animations were recorded for
        self.animations_coefs = {}
        # (template, part names, coefs) -> height of the root part above the lowest point of the body
        self.root_heights = {}
        # template -> {part name: parent part name}
        self.parents = {}
        # part name -> True when the figure is the same in all 8 body variants
        self.constant_figures = {}

    def part_parents(self, template):
        if template not in self.parents:
            parents = {}
            res.walk_tree(self.links[template]['ordered'], None, lambda parent, name: parents.__setitem__(name, parent))
            self.parents[template] = parents
        return self.parents[template]

    def is_figure_constant(self, part_name):
        """
        Some parts, for example the head of a human, do not depend on the coefs: all 8 variants of the figure are the same.
        """
        if part_name not in self.constant_figures:
            constant = False
            fig_data = self.figures.get(part_name)
            if fig_data:
                constant = all(block[a][v] == block[a][0] for block in fig_data[13] for a in range(3) for v in range(8))
            self.constant_figures[part_name] = constant
        return self.constant_figures[part_name]

    def part_coefs(self, template, part_name, coefs):
        """
        Returns coefs to build the given part with. Parts put on a part which does not depend on the coefs
        (a helmet on the head) are built with the coefs of the body variant the constant part was made for,
        otherwise a helmet of a small body becomes smaller than the head.
        """
        parent_part_name = self.part_parents(template).get(part_name)
        if not parent_part_name or not self.is_figure_constant(parent_part_name):
            return coefs
        animations_coefs = self.find_animations_coefs(template)
        if animations_coefs is None:
            return coefs
        return animations_coefs

    def find_animations_coefs(self, template):
        """
        Animations keep shifts of the parts made for one of the 8 body variants mixed by the coefs.
        Finds that variant comparing lengths of the shifts with lengths of the bones of every variant.
        Returns coefs of the variant or None when the model has no animations.
        """
        if template in self.animations_coefs:
            return self.animations_coefs[template]
        parents = self.part_parents(template)
        votes = [0] * 8
        for animation in self.animations.values():
            errors = [0.0] * 8
            found = False
            for part_name, anm_info in animation.items():
                bone = self.bones.get(part_name)
                if not parents.get(part_name) or not bone or len(bone) != 8 or not anm_info[3]:
                    continue
                shift_len = mth.vec3len(anm_info[3][0])
                for v in range(8):
                    errors[v] += abs(shift_len - mth.vec3len(bone[v]))
                found = True
            if found:
                votes[errors.index(min(errors))] += 1
        coefs = None
        if max(votes) > 0:
            v = votes.index(max(votes))
            # same order of the variants as in mth.trilinear(): index = 4 * coefs[2] + 2 * coefs[0] + coefs[1]
            coefs = [float((v >> 1) & 1), float(v & 1), float(v >> 2)]
        self.animations_coefs[template] = coefs
        return coefs

    def root_height(self, template, part_names, coefs):
        """
        Height of the root part above the lowest vertex of the body in the bind pose, in the coordinates of the model.
        """
        key = (template, tuple(part_names), tuple(coefs))
        if key in self.root_heights:
            return self.root_heights[key]

        def _lowest(tree, position):
            """
            Returns the lowest vertex of the part and of all parts attached to it.
            """
            lowest = 0.0
            part_name = tree[0]
            if part_name not in part_names:
                return lowest
            if position is None:
                # the root part is placed by the animation, its own shift is not part of the height
                position = [0.0, 0.0, 0.0]
            else:
                bone = self.bones.get(part_name)
                if bone and len(bone) == 8:
                    position = [position[i] + mth.trilinear([bone[v][i] for v in range(8)], coefs) for i in range(3)]
            fig_data = self.figures.get(part_name)
            if fig_data:
                for block in fig_data[13]:
                    for j in range(4):
                        lowest = min(lowest, position[2] + mth.trilinear([block[2][k][j] for k in range(8)], coefs))
            for branch in tree[1]:
                lowest = min(lowest, _lowest(branch, position))
            return lowest

        lowest = _lowest(self.links[template]['ordered'], None)
        self.root_heights[key] = -lowest
        return -lowest

    def animations_root_scale(self, template, part_names, coefs):
        """
        Shifts of the root part in the animations (height of the hips, swinging of the body) are recorded for one body variant.
        Returns the ratio which fits them to the body built with the given coefs, so the legs stay on the ground.
        """
        animations_coefs = self.find_animations_coefs(template)
        if animations_coefs is None:
            return 1.0
        recorded_height = self.root_height(template, part_names, animations_coefs)
        if recorded_height <= 0.0:
            return 1.0
        return self.root_height(template, part_names, coefs) / recorded_height

    def load_figure_data(self, figure_dir_path, template):
        lnk_file_path = os.path.join(figure_dir_path, template + '.lnk')
        if not os.path.isfile(lnk_file_path):
            return None
        lnk_list, lnk_tree, lnk_parents, _ = res.read_lnk_info(lnk_file_path)
        self.links[template] = {
            'ordered': lnk_list,
            # 'tree': lnk_tree,
            # 'parents': lnk_parents,
        }
        this_model_links = res.flat_tree(lnk_list)
        anm_count = 0
        for file_name in os.listdir(figure_dir_path):
            file_name = file_name.lower()
            sub_path = os.path.join(figure_dir_path, file_name)
            if os.path.isfile(sub_path):
                if file_name.endswith('.fig'):
                    if file_name[:-4] not in this_model_links:
                        continue
                    fig_file_path = os.path.join(figure_dir_path, file_name)
                    # try:
                    fig_info = res.read_fig_info(fig_file_path)
                    # except Exception as exc:
                    #     if _Debug:
                    #         print(f'error reading figure file {fig_file_path}: {exc}')
                    #     continue
                    self.figures[file_name[:-4]] = fig_info
                elif file_name.endswith('.bon'):
                    if file_name[:-4] not in this_model_links:
                        continue
                    bon_file_path = os.path.join(figure_dir_path, file_name)
                    self.bones[file_name[:-4]] = res.read_bon_info(bon_file_path)
                    # if file_name[:-4] == 'rh3':
                    #     print('bon_file_path', bon_file_path, file_name[:-4])
                    #     self.bones['rh3.arrow00'] = self.bones['rh3'].copy()
            elif os.path.isdir(sub_path):
                if file_name not in self.animations:
                    self.animations[file_name] = {}
                for sub_file_name in os.listdir(sub_path):
                    sub_file_name = sub_file_name.lower()
                    if sub_file_name.endswith('.anm'):
                        anm_file_path = os.path.join(sub_path, sub_file_name)
                        self.animations[file_name][sub_file_name[:-4]] = res.read_anm_info(anm_file_path)
                        anm_count += 1
        if _Debug:
            print(f'      for model {{{template}}} loaded {len(self.links)} links, {len(self.figures)} figures, {len(self.bones)} bones and {anm_count} animations')

    def unpack_figure_data(self, figures_res_file_path, destination_dir_path, template, selected_parts=[], selected_animations=[], save_json=False):
        destination_sub_dir_path = os.path.join(destination_dir_path, template)
        if not os.path.isdir(destination_sub_dir_path):
            os.makedirs(destination_sub_dir_path)
        lnk_count = 0
        fig_count = 0
        bon_count = 0
        anm_count = 0
        with open(figures_res_file_path, 'rb') as figures_file:
            res_filetree_dict = res.read_res_filetree(figures_file, return_dict=True)
            lnk_file_name = template + '.lnk'
            mod_file_name = template + '.mod'
            anm_file_name = template + '.anm'
            lnk_element = res_filetree_dict.get(lnk_file_name)
            this_model_links = None
            if lnk_element:
                res.unpack_res_element(figures_file, lnk_element, dest_file_path=os.path.join(destination_sub_dir_path, lnk_file_name))
                lnk_file_path = os.path.join(destination_sub_dir_path, lnk_file_name)
                lnk_list, lnk_tree, lnk_parents, _ = res.read_lnk_info(lnk_file_path)
                lnk_count += 1
                self.links[lnk_file_name[:-4]] = {
                    'ordered': lnk_list,
                    # 'tree': lnk_tree,
                    # 'parents': lnk_parents,
                }
                this_model_links = res.flat_tree(lnk_list)
            res_mod_element = res_filetree_dict.get(mod_file_name)
            if res_mod_element:
                mod_file_path = res.unpack_res_element(figures_file, res_mod_element, dest_file_path=os.path.join(destination_sub_dir_path, mod_file_name))
                mod_filetree = res.unpack_mod_info(mod_file_path, destination_dir_path=destination_sub_dir_path)
                if this_model_links is None:
                    for mod_element in sorted(mod_filetree):
                        el = mod_element[0][:-4]
                        if mod_element[0].lower().endswith('.lnk'):
                            lnk_file_name = mod_element[0].lower()
                            lnk_file_path = os.path.join(destination_sub_dir_path, lnk_file_name)
                            lnk_list, lnk_tree, lnk_parents, _ = res.read_lnk_info(lnk_file_path)
                            lnk_count += 1
                            self.links[lnk_file_name[:-4]] = {
                                'ordered': lnk_list,
                                # 'tree': lnk_tree,
                                # 'parents': lnk_parents,
                            }
                            if template == lnk_file_name[:-4]:
                                this_model_links = res.flat_tree(lnk_list)
                for mod_element in sorted(mod_filetree):
                    if mod_element[0].lower().endswith('.fig'):
                        fig_element = mod_element[0][:-4].lower()
                        if fig_element not in this_model_links:
                            continue
                        if not selected_parts or fig_element in selected_parts:
                            fig_file_path = os.path.join(destination_sub_dir_path, mod_element[0])
                            fig_info = res.read_fig_info(fig_file_path)
                            self.figures[mod_element[0][:-4]] = fig_info
                            fig_count += 1
            else:
                if this_model_links is not None:
                    for part_name in this_model_links:
                        part_name = part_name.lower()
                        fig_element = part_name + '.fig'
                        bon_element = part_name + '.bon'
                        if not selected_parts or part_name in selected_parts:
                            res_fig_element = res_filetree_dict.get(template + fig_element)
                            if res_fig_element:
                                fig_file_path = res.unpack_res_element(figures_file, res_fig_element, dest_file_path=os.path.join(destination_sub_dir_path, fig_element))
                                fig_info = res.read_fig_info(fig_file_path)
                                self.figures[part_name] = fig_info
                                fig_count += 1
                            res_bon_element = res_filetree_dict.get(template + bon_element)
                            if res_bon_element:
                                bon_file_path = res.unpack_res_element(figures_file, res_bon_element, dest_file_path=os.path.join(destination_sub_dir_path, bon_element))
                                bon_info = res.read_bon_info(bon_file_path)
                                self.bones[part_name] = bon_info
                                bon_count += 1
            res_anm_element = res_filetree_dict.get(anm_file_name)
            if res_anm_element:
                anm_file_path = res.unpack_res_element(figures_file, res_anm_element, dest_file_path=os.path.join(destination_sub_dir_path, anm_file_name))
                with open(anm_file_path, 'rb') as anm_file:
                    anm_filetree = res.read_res_filetree(anm_file)
                    for anm_element in anm_filetree:
                        if not os.path.isdir(os.path.join(destination_sub_dir_path, anm_element[0])):
                            os.makedirs(os.path.join(destination_sub_dir_path, anm_element[0]))
                        anm_element[0] += '.anm'
                    res.unpack_res(anm_file, anm_filetree, destination_dir_path=destination_sub_dir_path)
                    for anm_element in anm_filetree:
                        el = anm_element[0][:-4]
                        if not selected_animations or el in selected_animations:
                            one_anm_file_path = os.path.join(destination_sub_dir_path, anm_element[0])
                            with open(one_anm_file_path, 'rb') as one_anm_file:
                                one_anm_filetree = res.read_res_filetree(one_anm_file)
                                self.animations[anm_element[0][:-4]] = {}
                                for one_anm_element in one_anm_filetree:
                                    el_part = one_anm_element[0]
                                    if this_model_links and el_part not in this_model_links:
                                        continue
                                    if not selected_parts or el_part in selected_parts:
                                        one_anm_dest_file_path = os.path.join(destination_sub_dir_path, anm_element[0][:-4], one_anm_element[0] + '.anm')
                                        res.unpack_res_element(one_anm_file, one_anm_element, dest_file_path=one_anm_dest_file_path)
                                        self.animations[anm_element[0][:-4]][one_anm_element[0]] = res.read_anm_info(one_anm_dest_file_path)
                                        anm_count += 1
            res_bon_element = res_filetree_dict.get(template + '.bon')
            if res_bon_element:
                bon_file_path = res.unpack_res_element(figures_file, res_bon_element, dest_file_path=os.path.join(destination_sub_dir_path, template + '.bon'))
                with open(bon_file_path, 'rb') as bon_file:
                    bon_filetree = res.read_res_filetree(bon_file)
                    for bon_element in bon_filetree:
                        bon_element[0] += '.bon'
                    res.unpack_res(bon_file, bon_filetree, destination_dir_path=destination_sub_dir_path)
                    for bon_element in bon_filetree:
                        if this_model_links and bon_element[0][:-4] not in this_model_links:
                            continue
                        one_bon_file_path = os.path.join(destination_sub_dir_path, bon_element[0])
                        self.bones[bon_element[0][:-4]] = res.read_bon_info(one_bon_file_path)
                        bon_count += 1
        if _Debug:
            print(f'      for model {{{template}}} unpacked {lnk_count} links, {fig_count} figures, {bon_count} bones and {anm_count} animations')
        if save_json:
            dest_json_file_path = os.path.join(destination_dir_path, template + '.json')
            open(dest_json_file_path, 'wt').write(json.dumps({
                'template': template,
                'figures': self.figures,
                'links': self.links,
                'animations': self.animations,
                'bones': self.bones,
            }, indent=2))
            if _Debug:
                print(f'      for model {{{template}}} saved {lnk_count} links, {fig_count} figures, {bon_count} bones and {anm_count} animations to {dest_json_file_path}')

#------------------------------------------------------------------------------


class CatalogData(object):

    def __init__(self):
        # store id -> figures.json, animations.json, armors.json, weapons.json and materials.json of the store,
        # a unit is built of the data of the store of its template, see lib.stores
        self.figures = {}
        self.animations = {}
        self.armors = {}
        self.weapons = {}
        self.materials = {}
        # catalog/land.json: type of the land or types around a segment -> ids of the tiles, see lib.terrain
        self.land = {}
        self.land_tiles = {}

    def load_land_tiles(self, land_tiles_file_path):
        land_tiles_list = json.loads(open(land_tiles_file_path, 'rt').read())
        for tile_info in land_tiles_list:
            catalog_id, mozaic_id, mozaic_pos = [int(i) for i in tile_info.split(' ')]
            self.land_tiles[catalog_id] = (mozaic_id, mozaic_pos)

    def load_land(self, land_file_path):
        self.land = json.loads(open(land_file_path, 'rt').read())

    def load_figures(self, figures_file_path):
        self.figures = stores.read(figures_file_path)

    def load_animations(self, animations_file_path):
        self.animations = stores.read(animations_file_path)

    def load_armors(self, armors_file_path):
        self.armors = stores.read(armors_file_path)

    def load_weapons(self, weapons_file_path):
        self.weapons = stores.read(weapons_file_path)

    def load_materials(self, materials_file_path):
        self.materials = stores.read(materials_file_path)

    def store_animations(self, store_id=None):
        # animations.json of that store, of the default store when store_id is None
        return stores.of(self.animations, store_id)

    def build_template_data(self, model_name, skin=0, hair=None, wears=[], weapon=None, texture=None, store_id=None):
        figures = stores.of(self.figures, store_id)
        armors = stores.of(self.armors, store_id)
        weapons = stores.of(self.weapons, store_id)
        materials = stores.of(self.materials, store_id)
        animations = []
        textures = {}
        is_human = model_name in ['unhuma', 'unhufe']
        is_orc = model_name in ['unorfe', 'unorma']
        if is_human or is_orc:
            textures['*'] = f'{model_name}skin_{skin:02}:0'
        else:
            textures['*'] = f'{texture or "default0"}:0'
        if not isinstance(wears, list):
            wears = [wears, ]
        parts_ordered_tree = figures[model_name].copy()
        parts_list_flat = res.flat_tree(parts_ordered_tree)
        parts = []
        for part_name in parts_list_flat:
            if part_name.count('.') == 1:
                continue
            ignore_prefix_found = False
            for ignore_prefix in ['r_shell', 'l_shell', 'bwpart', 'bwtetiva', 'baserh', 'basearrow', 'basepike', 'baseaxe',
                                  'baseclub', 'crbow', 'basesword', 'basedagger', 'basesword', 'quiver', 'arrows']:
                if part_name.startswith(ignore_prefix):
                    ignore_prefix_found = True
                    break
            if ignore_prefix_found:
                continue
            if part_name not in parts:
                parts.append(part_name)
        if not parts:
            parts = parts_list_flat.copy()
        has_helm = False
        # has_plate = False
        # has_pants = False
        # has_shirt = False
        # has_boots = False
        for wear in wears:
            armor_name, material_name = wear.split('.')
            material_name = material_name.strip()
            armor = armors[armor_name]
            armor_type = armor['type']
            if armor_type == 'helm':
                has_helm = True
            # elif armor_type == 'plate':
            #     has_plate = True
            # elif armor_type == 'pants':
            #     has_pants = True
            # elif armor_type == 'shirt':
            #     has_shirt = True
            # elif armor_type == 'boots':
            #     has_boots = True
            armor_id = int(armor['texture_type_1'])
            material = materials[material_name]
            body_parts = dress.armor_body_parts(armor_type, armor_id)
            for body_part in body_parts:
                body_part_texture = dress.armor_texture_name(model_name, armor_type, armor, material)
                if body_part.count('.armor'):
                    if armor_type in ['helm', ]:
                        body_part_texture = f'{body_part_texture}:1'
                    elif armor_type in ['boots', 'gloves', ]:
                        body_part_texture = f'{body_part_texture}:0'
                    elif armor_type in ['plate', 'pants']:
                        body_part_texture = f'{body_part_texture}:0'
                    else:
                        body_part_texture = f'{body_part_texture}:0'
                elif body_part in parts:
                    body_part_texture = f'{body_part_texture}:0'
                else:
                    body_part_texture = f'{body_part_texture}:0'
                textures[body_part] = body_part_texture
            for body_part in body_parts:
                if body_part not in parts:
                    parts.append(body_part)
        if is_human:
            if not has_helm and hair is not None and hair >= 0:
                parts.append(f'hr.{hair:02}')
        weapon_type = None
        if weapon:
            weapon_type, weapon_parts, weapon_textures = dress.weapon_parts(model_name, weapon, weapons, materials)
            parts.extend(weapon_parts)
            textures.update(weapon_textures)
        anim = self.store_animations(store_id)[model_name]
        action_types = {}
        for action in anim['actions']:
            action_name = action['action_name']
            animation_length = action['animation_length']
            weapons = action['weapons'].split(',')
            if not action['weapons'] or weapons == ['all', ]:
                animations.append(action_name)
            else:
                if weapon_type and weapon_type in weapons:
                    animations.append(action_name)
            action_type = action['action_type'].split(':')[0]
            if action['animation_stage'] == 'cycle':
                if action_type not in action_types:
                    action_types[action_type] = []
                action_types[action_type].append({
                    'name': action_name,
                    'frames': animation_length,
                })
        return dict(
            model_name=model_name,
            parts=parts,
            textures=textures,
            action_types=action_types,
            animations=animations,
            min_height=anim['minimal_height'],
            avg_height=anim['average_height'],
            max_height=anim['maximal_height'],
        )

#------------------------------------------------------------------------------


class LandData(object):

    def __init__(self, catalog):
        self.width = None
        self.height = None
        self.catalog = catalog
        self.tiles_textures_dir_path = None
        # elevation of the land at the corners of the segments, the land is drawn with it
        self.elevation_map_data = {}
        # elevation where units walk, higher than the land where a bridge goes over it
        self.walk_elevation_map_data = {}
        self.tiles_map_data = {}
        self.tiles_files = {}
        self.plants_map_data = {}
        self.plants_variants = {}
        self.buildings_map_data = {}
        self.capitals = {}
        self.towers = {}
        # plants and buildings as they are in plants.bin and buildings.json, the zonemap is built with them
        self.plants_data = None
        self.buildings_data = None
        # lib.zone.ZoneMap: where units can walk, the movement is predicted with it the same way the server moves them
        self.zonemap = None

    def load_zonemap_file(self, zonemap_file_path):
        # the heightmap, the plants and the buildings are loaded before: the heightmap gives the size of the map,
        # trees and buildings are added to the closed land of the file the same way the server does it
        with open(zonemap_file_path, 'rb') as f:
            self.zonemap = zone.ZoneMap.build(f.read(), self.width, self.height, self.plants_data, self.buildings_data)

    def load_terrain_file(self, terrain_file_path):
        # tiles of the segments are built from the types of the land at their corners, see lib.terrain
        with open(terrain_file_path, 'rb') as f:
            types, width, height = terrain.decode(f.read())
        if self.width is not None and self.width != width:
            raise ValueError(f'land width mismatch: expected {self.width}, got {width}')
        if self.height is not None and self.height != height:
            raise ValueError(f'land height mismatch: expected {self.height}, got {height}')
        tiles = terrain.build_tiles(types, width, height, self.catalog.land)
        step = 1.0 / 8.0
        corr = 1.0 / ( 64.0 * 8.0 )
        for x in range(self.width):
            for y in range(self.height):
                catalog_id, rotate = tiles[(x, y)]
                mozaic_id, mozaic_pos = self.catalog.land_tiles[catalog_id]
                tex_cell_x = ( mozaic_pos % 8 ) * step
                tex_cell_y = ( mozaic_pos // 8 ) * step
                if rotate == 270:
                    tex_coord00 = (tex_cell_x + 0.0 / 8.0 + corr, tex_cell_y + 1.0 / 8.0 - corr)
                    tex_coord01 = (tex_cell_x + 1.0 / 8.0 - corr, tex_cell_y + 1.0 / 8.0 - corr)
                    tex_coord10 = (tex_cell_x + 0.0 / 8.0 + corr, tex_cell_y + 0.0 / 8.0 + corr)
                    tex_coord11 = (tex_cell_x + 1.0 / 8.0 - corr, tex_cell_y + 0.0 / 8.0 + corr)
                elif rotate == 0:
                    tex_coord00 = (tex_cell_x + 0.0 / 8.0 + corr, tex_cell_y + 0.0 / 8.0 + corr)
                    tex_coord01 = (tex_cell_x + 0.0 / 8.0 + corr, tex_cell_y + 1.0 / 8.0 - corr)
                    tex_coord10 = (tex_cell_x + 1.0 / 8.0 - corr, tex_cell_y + 0.0 / 8.0 + corr)
                    tex_coord11 = (tex_cell_x + 1.0 / 8.0 - corr, tex_cell_y + 1.0 / 8.0 - corr)
                elif rotate == 90:
                    tex_coord00 = (tex_cell_x + 1.0 / 8.0 - corr, tex_cell_y + 0.0 / 8.0 + corr)
                    tex_coord01 = (tex_cell_x + 0.0 / 8.0 + corr, tex_cell_y + 0.0 / 8.0 + corr)
                    tex_coord10 = (tex_cell_x + 1.0 / 8.0 - corr, tex_cell_y + 1.0 / 8.0 - corr)
                    tex_coord11 = (tex_cell_x + 0.0 / 8.0 + corr, tex_cell_y + 1.0 / 8.0 - corr)
                elif rotate == 180:
                    tex_coord00 = (tex_cell_x + 1.0 / 8.0 - corr, tex_cell_y + 1.0 / 8.0 - corr)
                    tex_coord01 = (tex_cell_x + 1.0 / 8.0 - corr, tex_cell_y + 0.0 / 8.0 + corr)
                    tex_coord10 = (tex_cell_x + 0.0 / 8.0 + corr, tex_cell_y + 1.0 / 8.0 - corr)
                    tex_coord11 = (tex_cell_x + 0.0 / 8.0 + corr, tex_cell_y + 0.0 / 8.0 + corr)
                self.tiles_map_data[(x, y)] = (mozaic_id, tex_coord00, tex_coord01, tex_coord10, tex_coord11)
        return self.width, self.height

    def elevation_unpack(self, h):
        """
        h is from 0 to 100
        result is from -water_level*underwater_factor to 100^height_exponent
        """
        if h > const.INPUT_WATER_LEVEL:
            return pow(h - 18, const.ELEVATION_UNPACK_EXPONENT)
        if h <= 0:
            return -1 * (const.INPUT_WATER_LEVEL - 1) * const.ELEVATION_UNPACK_UNDERWATER_FACTOR
        return (float(h - const.INPUT_WATER_LEVEL) / float(h)) * float(const.ELEVATION_UNPACK_UNDERWATER_FACTOR)

    def load_heightmap_file(self, heightmap_file_path):
        e_min_unpacked = self.elevation_unpack(1)
        e_max_unpacked = self.elevation_unpack(100)
        unpacked_delta = e_max_unpacked - e_min_unpacked
        # the red channel is where units walk, read the same way as on the server, see
        # server.simulation.Simulation.load_heightmap(), the green one is the land, see server.land.build_walk_heights()
        im = PILImage.open(heightmap_file_path).convert('RGB')
        self.width, self.height = im.size
        pixels = im.load()
        scaled = {}
        for w in range(self.width):
            for h in range(self.height):
                walk_value, land_value, _ = pixels[w, h]
                for value in (walk_value, land_value):
                    if value not in scaled:
                        scaled[value] = float(self.elevation_unpack(float(value)) - e_min_unpacked) / unpacked_delta
                self.walk_elevation_map_data[(w, h)] = scaled[walk_value]
                self.elevation_map_data[(w, h)] = scaled[land_value]
        return self.width, self.height

    # def load_heightmap_file(self, heightmap_file_name):
    #     im = Image(heightmap_file_name, keep_data=True)
    #     self.width = im.width
    #     self.height = im.height
    #     for w in range(self.width):
    #         for h in range(self.height):
    #             e = float(im.read_pixel(w, h)[0]) * 255.0
    #             self.elevation_map_data[(w, h)] = e
    #     return self.width, self.height

    def load_cache_tiles_textures(self, textures_dir_path):
        self.tiles_textures_dir_path = textures_dir_path
        count = 0
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
                    count += 1
                # textures stay in the kivy cache when the next biome is loaded, but this land needs all of them
                self.tiles_files[int(file_name[:-4])] = file_path
        if _Debug:
            print(f'  cached {count} textures at {self.tiles_textures_dir_path} for land tiles')

    def load_plants_data(self, plants_data_file_path):
        with open(plants_data_file_path, 'rb') as f:
            plants_data = plants.decode(f.read())
        self.plants_data = plants_data
        for plant_key in plants_data.keys():
            template, texture, parts = plant_key.split('#')
            plants_list = plants_data[plant_key]
            for plant_coded in plants_list:
                coefs, w, h, direction, plant_name, plant_template, _ = plants.split(plant_coded)
                w = float(w)
                h = float(h)
                c1, c2, c3 = coefs.split(':')
                coefs_q = mth.quantize_coefs([float(c1), float(c2), float(c3)])
                coefs_str = ':'.join([str(c) for c in coefs_q])
                plant_variant_key = f'{template}#{texture}#{parts}#{coefs_str}'
                plant = {}
                plant['k'] = plant_variant_key
                plant['m'] = template
                plant['t'] = texture
                plant['c'] = coefs_q
                if plant_variant_key not in self.plants_variants:
                    variant = dict(plant)
                    variant['so'] = None
                    self.plants_variants[plant_variant_key] = variant
                int_w = int(float(w))
                int_h = int(float(h))
                shift_w = float(w) - float(int_w)
                shift_h = float(h) - float(int_h)
                plant['w'] = int_w
                plant['h'] = int_h
                plant['sw'] = shift_w
                plant['sh'] = shift_h
                plant['d'] = float(direction)
                if (int_w, int_h) not in self.plants_map_data:
                    self.plants_map_data[(int_w, int_h)] = []
                self.plants_map_data[(int_w, int_h)].append(plant)

    def load_buildings_data(self, buildings_data_file_path):
        buildings_data = json.loads(open(buildings_data_file_path, 'rt').read())
        self.buildings_data = buildings_data
        for building_info in buildings_data:
            x = int(building_info['x'])
            y = int(building_info['y'])
            if (x, y) not in self.buildings_map_data:
                self.buildings_map_data[(x, y)] = []
            self.buildings_map_data[(x, y)].append(building_info)
            if building_info['k'] == 'capital':
                self.capitals[building_info['i']] = building_info
            elif building_info['k'] == 'tower':
                self.towers[building_info['i']] = building_info

    def save_elevation_memmap(self, file_name_prefix, destination_dir_path):
        file_path = os.path.join(destination_dir_path, f'{file_name_prefix}.{self.width}.{self.height}.memmap')
        fp = np.memmap(file_path, dtype='float32', mode='w+', shape=(self.width, self.height))
        fp[:] = self.elevation_map_data
        fp.flush()
        return file_path

    def get_elevation(self, w, h):
        # positions are not limited by the size of the map, it is repeated around itself
        _w = w % self.width
        _h = h % self.height
        return self.elevation_map_data[(_w, _h)]

    def get_walk_elevation(self, w, h):
        """
        Elevation where units walk at the corner of a segment: on a bridge it is higher than the land under it.
        """
        _w = w % self.width
        _h = h % self.height
        return self.walk_elevation_map_data[(_w, _h)]

    def get_texture(self, w, h):
        # positions are not limited by the size of the map, it is repeated around itself
        _w = w % self.width
        _h = h % self.height
        mozaic_id, tex_coord00, tex_coord01, tex_coord10, tex_coord11 = self.tiles_map_data[(_w, _h)]
        return self.tiles_files[mozaic_id], tex_coord00, tex_coord01, tex_coord10, tex_coord11

#------------------------------------------------------------------------------


def main():
    cmd = sys.argv[1]
    if cmd.startswith('list_') and len(sys.argv) > 2:
        st = res.scan_figures_data(sys.argv[2])
        print('\n'.join(sorted(st[sys.argv[1].replace('list_', '')])))
    elif cmd.startswith('show_') and len(sys.argv) > 3:
        st = res.scan_figures_data(sys.argv[2])
        print('\n'.join(sorted(st[sys.argv[1].replace('show_', '')])))
    elif cmd == 'list' and len(sys.argv) == 3:
        st = res.scan_figures_data(sys.argv[2])
        print('\n'.join(sorted(st.keys())))
    elif cmd == 'list_models':
        st = res.scan_figures_data(sys.argv[2])
        print('\n'.join(sorted(st['mod'])))
    elif cmd == 'scan_textures':
        ls = res.scan_textures_data(sys.argv[2])
        print('\n'.join(sorted(ls)))
    elif cmd == 'list_figures':
        st = res.scan_figures_data(sys.argv[2])
        print('\n'.join(sorted(st['fig'])))
    elif cmd == 'unpack_models':
        md = ModelData()
        st = res.scan_figures_data(sys.argv[2])
        lst = sorted(st['mod'])
        for m in lst:
            print(f'loading {m}')
            try:
                md.unpack_figure_data(sys.argv[2], destination_dir_path='models', template=m)
            except Exception as exc:
                print(m, traceback.format_exc())
            

if __name__ == '__main__':
    main()
