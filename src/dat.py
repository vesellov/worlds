import os
import sys
import json
import traceback
import pprint
import numpy as np

from kivy.core.image import Image
from kivy.cache import Cache
from kivy.resources import resource_find

import const
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
        self.texture = kwargs.get('texture', {})


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

    def scan_figures_data(self, figures_res_file_path):
        items = {}
        with open(figures_res_file_path, 'rb') as figures_file:
            res_filetree_dict = res.read_res_filetree(figures_file, return_dict=True)
            for k in res_filetree_dict.keys():
                ext = k[-4:].lower().replace('.', '')
                if ext not in items:
                    items[ext] = []
                items[ext].append(k[:-4])
        return items

    def unpack_texture(self, texture_res_file_path, destination_dir, name):
        if not os.path.isdir(destination_dir):
            os.makedirs(destination_dir)
        with open(texture_res_file_path, 'rb') as texture_file:
            res_filetree_dict = res.read_res_filetree(texture_file, return_dict=True)
            for k in res_filetree_dict.keys():
                if k.lower().endswith('.mmp') and k[:-4].lower() == name.lower():
                    mmp_file_path = os.path.join(destination_dir, name.lower() + '.mmp')
                    png_file_path = os.path.join(destination_dir, name.lower() + '.png')
                    res.unpack_res_element(texture_file, res_filetree_dict[k], dest_file_name=mmp_file_path)
                    pil_img = res.read_mmp(mmp_file_path)
                    pil_img.save(png_file_path)
                    os.remove(mmp_file_path)
                    if _Debug:
                        print(f'unpacked texture {name} to {png_file_path}')
                    return png_file_path
        return None

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
        anm_count = 0
        for file_name in os.listdir(figure_dir_path):
            file_name = file_name.lower()
            sub_path = os.path.join(figure_dir_path, file_name)
            if os.path.isfile(sub_path):
                if file_name.endswith('.fig'):
                    fig_file_path = os.path.join(figure_dir_path, file_name)
                    # try:
                    fig_info = res.read_fig_info(fig_file_path)
                    # except Exception as exc:
                    #     if _Debug:
                    #         print(f'error reading figure file {fig_file_path}: {exc}')
                    #     continue
                    self.figures[file_name[:-4]] = fig_info
                    continue
                if file_name.endswith('.bon'):
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
            print(f'for model {{{template}}} loaded {len(self.links)} links, {len(self.figures)} figures, {len(self.bones)} bones and {anm_count} animations')

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
            # if template == 'unhufe':
            #     pprint.pprint(res_filetree_dict)
            if res_mod_element:
                mod_file_name = res.unpack_res_element(figures_file, res_mod_element, dest_file_name=os.path.join(destination_sub_dir, template + '.mod'))
                mod_filetree = res.unpack_mod_info(mod_file_name, destination_dir=destination_sub_dir)
                # print('mod_filetree', mod_file_name, mod_filetree)
                for mod_element in sorted(mod_filetree):
                    el = mod_element[0][:-4]
                    if mod_element[0].lower().endswith('.fig'):
                        if not selected_parts or el in selected_parts:
                            fig_file_name = os.path.join(destination_sub_dir, mod_element[0])
                            # try:
                            fig_info = res.read_fig_info(fig_file_name)
                            # except Exception as exc:
                            #     if _Debug:
                            #         print(f'error reading figure file {fig_file_name}: {exc}')
                            #     continue
                            # print('fig_file_name', fig_file_name, mod_element[0][:-4])
                            self.figures[mod_element[0][:-4]] = fig_info
                            fig_count += 1
                    elif mod_element[0].lower().endswith('.lnk'):
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
                    # if template == 'unhufe':
                        # print('!!!!!!!!!!!!!', anm_file_name)
                    #     pprint.pprint(anm_filetree)
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
                                # print('one_anm_filetree', one_anm_file, one_anm_filetree)
                                self.animations[anm_element[0][:-4]] = {}
                                for one_anm_element in one_anm_filetree:
                                    el_part = one_anm_element[0]
                                    if not selected_parts or el_part in selected_parts:
                                        one_anm_dest_file_name = os.path.join(destination_sub_dir, anm_element[0][:-4], one_anm_element[0] + '.anm')
                                        res.unpack_res_element(one_anm_file, one_anm_element, dest_file_name=one_anm_dest_file_name)
                                        self.animations[anm_element[0][:-4]][one_anm_element[0]] = res.read_anm_info(one_anm_dest_file_name)
                                        anm_count += 1
            res_bon_element = res_filetree_dict.get(template + '.bon')
            # if template == 'unhufe':
            #     pprint.pprint(res_bon_element)
            if res_bon_element:
                bon_file_name = res.unpack_res_element(figures_file, res_bon_element, dest_file_name=os.path.join(destination_sub_dir, template + '.bon'))
                with open(bon_file_name, 'rb') as bon_file:
                    bon_filetree = res.read_res_filetree(bon_file)
                    # if template == 'unhufe':
                        # print('!!!!!!!!!!!!!', bon_file_name)
                    #     pprint.pprint(bon_filetree)
                    for bon_element in bon_filetree:
                        bon_element[0] += '.bon'
                    res.unpack_res(bon_file, bon_filetree, destination_dir=destination_sub_dir)
                    for bon_element in bon_filetree:
                        one_bon_file_name = os.path.join(destination_sub_dir, bon_element[0])
                        self.bones[bon_element[0][:-4]] = res.read_bon_info(one_bon_file_name)
                        bon_count += 1
                        # if bon_element[0][:-4] == 'rh3':
                        #     print('bon_filetree', bon_element[0])
                        #     self.bones['rh3.arrow00'] = self.bones['rh3'].copy()
                        #     bon_count += 1
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


class CatalogData(object):

    def __init__(self):
        self.figures = {}
        self.animations = {}
        self.armors = {}
        self.weapons = {}
        self.materials = {}

    def load_figures(self, figures_file_name):
        self.figures = json.loads(open(figures_file_name, 'rt').read())

    def load_animations(self, animations_file_name):
        self.animations = json.loads(open(animations_file_name, 'rt').read())

    def load_armors(self, armors_file_name):
        self.armors = json.loads(open(armors_file_name, 'rt').read())

    def load_weapons(self, weapons_file_name):
        self.weapons = json.loads(open(weapons_file_name, 'rt').read())

    def load_materials(self, materials_file_name):
        self.materials = json.loads(open(materials_file_name, 'rt').read())

    def build_template_data(self, model_name, skin=0, hair=None, wears=[], weapon=None, texture=None):
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
        parts_ordered_tree = self.figures[model_name].copy()
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
            armor = self.armors[armor_name]
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
            armor_code = dict(
                plate='pl', 
                gloves='gl',
                leggings='lg',
                boots='bt',
                shirt='sh',
                helm='hl',
                pants='pt',
            ).get(armor_type)
            texture_type_1 = armor['texture_type_1']
            texture_type_2 = armor['texture_type_2']
            armor_id = int(texture_type_1)
            material = self.materials[material_name]
            material_code = material['code']
            body_parts = dict(
                plate='bd.a,rh1.a,rh2.a,lh1.a,lh2.a',
                gloves='lh3,rh3',
                leggings='hp.a,rl1.a,rl2.a,rl3.a,ll1.a,ll2.a,ll3.a',
                boots='ll3,rl3',
                shirt='bd,rh1,rh2,rh3,lh1,lh2,lh3',
                helm='hd.a',
                pants='hp,rl1,rl2,ll1,ll2',
            ).get(armor_type).split(',')
            if armor_type == 'pants' and armor_id in [1, 2, 3, 6]:
                body_parts.append('l_shell.a')
                body_parts.append('r_shell.a')
            for body_part in body_parts:
                body_part = body_part.replace('.a', f'.armor{armor_id:02}' if armor_id else '')
                body_part_texture = f"{model_name}{armor_code}_{texture_type_1:02}.{material_code}.{texture_type_2}"
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
                armor_id = texture_type_1
                body_part = body_part.replace('.a', f'.armor{armor_id:02}' if armor_id else '')
                if body_part not in parts:
                    parts.append(body_part)
        if is_human:
            if not has_helm and hair is not None and hair >= 0:
                parts.append(f'hr.{hair:02}')
        weapon_type = None
        if weapon:
            weapon_name, material_name = weapon.split('.')
            if material_name.count(' ['):
                material_name = material_name.split(' ')[0].strip()
            weapon_info = self.weapons[weapon_name]
            weapon_type = weapon_info['type']
            material = self.materials[material_name]
            material_code = material['code']
            texture_type_1 = weapon_info['texture_type_1']
            texture_type_2 = weapon_info['texture_type_2']
            weapon_id = int(texture_type_1)
            weapon_code = dict(
                hammer='hm',
                dagger='dg',
                spear='sp',
                crossbow='cb',
                sword='sw',
                axe='ax',
                bow='bw',
            ).get(weapon_type)
            body_parts = ''
            weapon_texture = f"{model_name}{weapon_code}_{texture_type_1:02}.{material_code}.{texture_type_2}:1"
            if weapon_type == 'bow':
                if weapon_id == 0:
                    parts.append('lh3.bwpartb00')
                    parts.append('bwparta00')
                    parts.append('bwtetivaa00')
                    parts.append('bwtetivab00')
                    textures['lh3.bwpartb00'] = weapon_texture
                    textures['bwparta00'] = weapon_texture
                    textures['bwtetivaa00'] = weapon_texture
                    textures['bwtetivab00'] = weapon_texture
                else:
                    parts.append('lh3.bwpartb00.hidden')
                    parts.append(f'lh3.bwpartb{weapon_id:02}')
                    textures[f'lh3.bwpartb{weapon_id:02}'] = weapon_texture
                    parts.append('bwparta00.hidden')
                    parts.append(f'bwparta{weapon_id:02}')
                    textures[f'bwparta{weapon_id:02}'] = weapon_texture
                    parts.append('bwtetivaa00.hidden')
                    parts.append(f'bwtetivaa{weapon_id:02}')
                    textures[f'bwtetivaa{weapon_id:02}'] = weapon_texture
                    parts.append('bwtetivab00.hidden')
                    parts.append(f'bwtetivab{weapon_id:02}')
                    textures[f'bwtetivab{weapon_id:02}'] = weapon_texture
                parts.append('rh3.arrow00')
                textures['rh3.arrow00'] = weapon_texture
                # parts.append('basearrow00')
                # textures['basearrow00'] = weapon_texture
                parts.append('quiver')
                textures['quiver'] = weapon_texture
                parts.append('arrows')
                textures['arrows'] = weapon_texture
            elif weapon_type == 'crossbow':
                if weapon_id == 1:
                    parts.append('rh3.crbow01main')
                    parts.append('crbow01part01')
                    parts.append('crbow01tetiva01')
                    parts.append('crbow01part02')
                    parts.append('crbow01tetiva02')
                    textures['rh3.crbow01main'] = weapon_texture
                    textures['crbow01part01'] = weapon_texture
                    textures['crbow01tetiva01'] = weapon_texture
                    textures['crbow01part02'] = weapon_texture
                    textures['crbow01tetiva02'] = weapon_texture
                else:
                    parts.append('rh3.crbow01main.hidden')
                    parts.append(f'rh3.crbow{weapon_id:02}main')
                    textures[f'rh3.crbow{weapon_id:02}main'] = weapon_texture
                    parts.append('crbow01part01.hidden')
                    parts.append(f'crbow{weapon_id:02}part01')
                    textures[f'crbow{weapon_id:02}part01'] = weapon_texture
                    parts.append('crbow01tetiva01.hidden')
                    parts.append(f'crbow{weapon_id:02}tetiva01')
                    textures[f'crbow{weapon_id:02}tetiva01'] = weapon_texture
                    parts.append('crbow01part02.hidden')
                    parts.append(f'crbow{weapon_id:02}part02')
                    textures[f'crbow{weapon_id:02}part02'] = weapon_texture
                    parts.append('crbow01tetiva02.hidden')
                    parts.append(f'crbow{weapon_id:02}tetiva02')
                    textures[f'crbow{weapon_id:02}tetiva02'] = weapon_texture
            elif weapon_type == 'spear':
                if weapon_id == 0:
                    parts.append('rh3.pike00')
                    parts.append('basepike00')
                    textures['rh3.pike00'] = weapon_texture
                    textures['basepike00'] = weapon_texture
                else:
                    parts.append('rh3.pike00.hidden')
                    parts.append(f'rh3.pike{weapon_id:02}')
                    parts.append(f'basepike{weapon_id:02}')
                    textures[f'rh3.pike{weapon_id:02}'] = weapon_texture
                    textures[f'basepike{weapon_id:02}'] = weapon_texture
            elif weapon_type == 'sword':
                if weapon_id == 0:
                    parts.append('rh3.sword00')
                    parts.append('basesword00')
                    textures['rh3.sword00'] = weapon_texture
                    textures['basesword00'] = weapon_texture
                else:
                    weapon_id = weapon_id % 5
                    parts.append('rh3.sword00.hidden')
                    parts.append(f'rh3.sword{weapon_id:02}')
                    parts.append(f'basesword{weapon_id:02}')
                    textures[f'rh3.sword{weapon_id:02}'] = weapon_texture
                    textures[f'basesword{weapon_id:02}'] = weapon_texture
            elif weapon_type == 'dagger':
                if weapon_id == 0:
                    parts.append('rh3.sword00')
                    parts.append('basesword00')
                    textures['rh3.sword00'] = weapon_texture
                    textures['basesword00'] = weapon_texture
                else:
                    parts.append('rh3.sword00.hidden')
                    parts.append(f'rh3.dagger{weapon_id:02}')
                    parts.append(f'basedagger{weapon_id:02}')
                    textures[f'rh3.dagger{weapon_id:02}'] = weapon_texture
                    textures[f'basedagger{weapon_id:02}'] = weapon_texture
            elif weapon_type == 'axe':
                if weapon_id == 0:
                    parts.append('rh3.axe00')
                    parts.append('baseaxe00')
                    textures['rh3.axe00'] = weapon_texture
                    textures['baseaxe00'] = weapon_texture
                else:
                    parts.append('rh3.axe00.hidden')
                    parts.append(f'rh3.axe{weapon_id:02}')
                    parts.append(f'baseaxe{weapon_id:02}')
                    textures[f'rh3.axe{weapon_id:02}'] = weapon_texture
                    textures[f'baseaxe{weapon_id:02}'] = weapon_texture
            elif weapon_type == 'hammer':
                if weapon_id == 0:
                    parts.append('rh3.axe00')
                    parts.append('baseaxe00')
                    textures['rh3.axe00'] = weapon_texture
                    textures['baseaxe00'] = weapon_texture
                else:
                    parts.append('rh3.axe00.hidden')
                    parts.append(f'rh3.club{weapon_id:02}')
                    parts.append(f'baseclub{weapon_id:02}')
                    textures[f'rh3.club{weapon_id:02}'] = weapon_texture
                    textures[f'baseclub{weapon_id:02}'] = weapon_texture
        anim = self.animations[model_name]
        action_types = {}
        for action in anim['actions']:
            action_name = action['action_name']
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
                action_types[action_type].append(action_name)
        return dict(
            model_name=model_name,
            parts=parts,
            textures=textures,
            animations=animations,
            action_types=action_types,
        )


class LandData(object):

    def __init__(self):
        self.width = None
        self.height = None
        self.tiles_textures_dir_path = None
        self.elevation_map_data = {}
        self.tiles_map_data = {}
        self.tiles_files = {}
        self.plants_map_data = {}
        self.plants_variants = {}
        self.buildings_map_data = {}
        self.capitals = {}
        self.towers = {}

    def load_tilemap_file(self, tilemap_file_name):
        tiles_list = json.loads(open('assets/tiles.json', 'rt').read())
        tiles_registry = {}
        for tile_info in tiles_list:
            catalog_id, mozaic_id, mozaic_pos = [int(i) for i in tile_info.split(' ')]
            tiles_registry[catalog_id] = (mozaic_id, mozaic_pos)
        im = Image(tilemap_file_name, keep_data=True)
        if self.width is not None and self.width != im.width:
            raise ValueError(f'land width mismatch: expected {self.width}, got {im.width}')
        if self.height is not None and self.height != im.height:
            raise ValueError(f'land height mismatch: expected {self.height}, got {im.height}')
        data = im.image._data[0]
        size = 3 if data.fmt in ('rgb', 'bgr') else 4
        step = 1.0 / 8.0
        corr = 1.0 / ( 64.0 * 8.0 )
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
                mozaic_id, mozaic_pos = tiles_registry[catalog_id]
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

    def load_heightmap_file(self, heightmap_file_name):
        e_min_unpacked = self.elevation_unpack(1)
        e_max_unpacked = self.elevation_unpack(100)
        unpacked_delta = e_max_unpacked - e_min_unpacked
        im = Image(heightmap_file_name, keep_data=True)
        self.width = im.width
        self.height = im.height
        for w in range(self.width):
            for h in range(self.height):
                e = float(im.read_pixel(w, h)[0]) * 255.0
                e_unpacked = self.elevation_unpack(e)
                e_scaled = (float(e_unpacked - e_min_unpacked) / unpacked_delta)
                self.elevation_map_data[(w, h)] = e_scaled
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
                    self.tiles_files[int(file_name[:-4])] = file_path
        if _Debug:
            print(f'  cached {count} textures at {self.tiles_textures_dir_path} for land tiles')

    def load_plants_data(self, plants_data_file_name):
        plants_data = json.loads(open(plants_data_file_name, 'rt').read())
        for plant_key in plants_data.keys():
            template, texture, parts = plant_key.split('#')
            plants_list = plants_data[plant_key]
            for plant_coded in plants_list:
                coefs, w, h, direction, plant_name, plant_template = plant_coded.split(' ')
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

    def load_buildings_data(self, buildings_data_file_name):
        buildings_data = json.loads(open(buildings_data_file_name, 'rt').read())
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
        mozaic_id, tex_coord00, tex_coord01, tex_coord10, tex_coord11 = self.tiles_map_data[(_w, _h)]
        return self.tiles_files[mozaic_id], tex_coord00, tex_coord01, tex_coord10, tex_coord11


def main():
    cmd = sys.argv[1]
    if cmd.startswith('list_') and len(sys.argv) > 2:
        md = ModelData()
        st = md.scan_figures_data(sys.argv[2])
        print('\n'.join(sorted(st[sys.argv[1].replace('list_', '')])))
    elif cmd.startswith('show_') and len(sys.argv) > 3:
        md = ModelData()
        st = md.scan_figures_data(sys.argv[2])
        print('\n'.join(sorted(st[sys.argv[1].replace('show_', '')])))
    elif cmd == 'list' and len(sys.argv) == 3:
        md = ModelData()
        st = md.scan_figures_data(sys.argv[2])
        print('\n'.join(sorted(st.keys())))
    elif cmd == 'list_models':
        md = ModelData()
        st = md.scan_figures_data(sys.argv[2])
        print('\n'.join(sorted(st['mod'])))
    elif cmd == 'list_figures':
        md = ModelData()
        st = md.scan_figures_data(sys.argv[2])
        print('\n'.join(sorted(st['fig'])))
    elif cmd == 'unpack_models':
        md = ModelData()
        st = md.scan_figures_data(sys.argv[2])
        lst = sorted(st['mod'])
        for m in lst:
            print(f'loading {m}')
            try:
                md.unpack_figure_data(sys.argv[2], destination_dir='models', template=m)
            except Exception as exc:
                print(m, traceback.format_exc())
            

if __name__ == '__main__':
    main()
