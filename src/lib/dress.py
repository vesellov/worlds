"""
Look of the figures: which parts and textures the armors and the weapons give them.
Shared by the client, which builds the models, and the server, which chooses random looks of the neutral units
the clients ask for, see Looks and server.simulation.Simulation.add_neutral_unit().
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import json
import random

#------------------------------------------------------------------------------

from lib import models
from lib import res
from lib import stores

#------------------------------------------------------------------------------

# armor type -> code of the armor in the names of its textures: <model><code>_<texture_type_1>.<material code>.<texture_type_2>
ARMOR_TYPE_CODES = {
    'plate': 'pl',
    'gloves': 'gl',
    'leggings': 'lg',
    'boots': 'bt',
    'shirt': 'sh',
    'helm': 'hl',
    'pants': 'pt',
}

# armor type -> parts of the figure it covers, ".a" is replaced with ".armor<texture_type_1>" when the armor has its own parts
ARMOR_BODY_PARTS = {
    'plate': ('bd.a', 'rh1.a', 'rh2.a', 'lh1.a', 'lh2.a'),
    'gloves': ('lh3', 'rh3'),
    'leggings': ('hp.a', 'rl1.a', 'rl2.a', 'rl3.a', 'll1.a', 'll2.a', 'll3.a'),
    'boots': ('ll3', 'rl3'),
    'shirt': ('bd', 'rh1', 'rh2', 'rh3', 'lh1', 'lh2', 'lh3'),
    'helm': ('hd.a', ),
    'pants': ('hp', 'rl1', 'rl2', 'll1', 'll2'),
}

# pants of these texture_type_1 also have the shells on the hips
ARMOR_PANTS_WITH_SHELLS = (1, 2, 3, 6)

# weapon type -> code of the weapon in the names of its textures: <model><code>_<texture_type_1>.<material code>.<texture_type_2>
WEAPON_TYPE_CODES = {
    'hammer': 'hm',
    'dagger': 'dg',
    'spear': 'sp',
    'crossbow': 'cb',
    'sword': 'sw',
    'axe': 'ax',
    'bow': 'bw',
}

# armor types in the order they are put on and how often a unit wears each of them
ARMOR_CHANCES = (
    ('shirt', 0.9),
    ('pants', 0.9),
    ('boots', 0.8),
    ('gloves', 0.7),
    ('leggings', 0.6),
    ('plate', 0.6),
    ('helm', 0.6),
)

# how often a unit which wears armors also has a weapon
WEAPON_CHANCE = 0.8

# kinds of the samples in figures_samples.json of the figures which wear armors
ARMORED_KINDS = ('human', 'orc')

# catalog/textures_store.json says which .res file of the store holds the texture, see client.loader.texture_location()
TEXTURES_STORE_FILE_NAME = 'textures_store.json'

#------------------------------------------------------------------------------


def armor_body_parts(armor_type, armor_id):
    """
    Parts of the figure which the armor covers, see ARMOR_BODY_PARTS.
    """
    body_parts = list(ARMOR_BODY_PARTS[armor_type])
    if armor_type == 'pants' and armor_id in ARMOR_PANTS_WITH_SHELLS:
        body_parts.append('l_shell.a')
        body_parts.append('r_shell.a')
    return [body_part.replace('.a', f'.armor{armor_id:02}' if armor_id else '') for body_part in body_parts]


def armor_texture_name(model_name, armor_type, armor, material):
    """
    Name of the texture of the armor made of the material for that model, without the layer number.
    """
    return f"{model_name}{ARMOR_TYPE_CODES[armor_type]}_{armor['texture_type_1']:02}.{material['code']}.{armor['texture_type_2']}"


def weapon_parts(model_name, weapon, weapons, materials):
    """
    Parts of the figure the weapon "<weapon name>.<material name>" adds and their textures, weapons and materials
    are the catalog. Parts with ".hidden" at the end are the parts of the figure the weapon hides.
    Returns (weapon type, parts, textures).
    """
    parts = []
    textures = {}
    weapon_name, material_name = weapon.split('.')
    if material_name.count(' ['):
        material_name = material_name.split(' ')[0].strip()
    weapon_info = weapons[weapon_name]
    weapon_type = weapon_info['type']
    material = materials[material_name]
    material_code = material['code']
    texture_type_1 = weapon_info['texture_type_1']
    texture_type_2 = weapon_info['texture_type_2']
    weapon_id = int(texture_type_1)
    weapon_code = WEAPON_TYPE_CODES.get(weapon_type)
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
    return weapon_type, parts, textures


def texture_name(texture):
    # textures of the templates can have the layer number at the end: unhumaskin_41:0
    return texture.split(':')[0].lower()

#------------------------------------------------------------------------------


class Looks(object):
    """
    Chooses random figures and their look from the catalog of the game.
    """

    def __init__(self, catalog_path, store_id=models.DEFAULT_STORE_ID):
        self.store_id = store_id
        # the catalog keeps the data of every store, the looks are made of the data of one store, see lib.stores
        self.figures = self.read_store(catalog_path, 'figures.json')
        self.animations = self.read_store(catalog_path, 'animations.json')
        self.armors = self.read_store(catalog_path, 'armors.json')
        self.weapons = self.read_store(catalog_path, 'weapons.json')
        self.materials = self.read_store(catalog_path, 'materials.json')
        # the samples are made of the maps of "Lost in Astral", the figures of all stores have the same constitution
        self.samples = self.read(catalog_path, 'figures_samples.json')
        self.store_textures = self.read(catalog_path, TEXTURES_STORE_FILE_NAME).get(str(store_id), {})
        self.models = self.showcase_models()
        if _Debug:
            print(f'{len(self.models)} figures can be shown from store {store_id}')

    def read(self, catalog_path, file_name):
        with open(os.path.join(catalog_path, file_name), 'rt') as f:
            return json.loads(f.read())

    def read_store(self, catalog_path, file_name):
        return stores.of(stores.read(os.path.join(catalog_path, file_name)), self.store_id)

    def has_texture(self, texture):
        return texture_name(texture) in self.store_textures

    def figure_parts(self, model_name):
        return set(res.flat_tree(self.figures[model_name]))

    def figure_textures(self, model_name):
        """
        Textures for a figure without armors: of its samples in figures_samples.json or, when it has no samples,
        of the store which are named as the figure in animations.json. Only textures which the store has.
        """
        textures = []
        for sample in self.samples.get(model_name, []):
            # samples with the list of parts are pieces of the figure, for example a golem falling apart
            if sample.get('p'):
                continue
            texture = sample['t'].lower()
            if self.has_texture(texture) and texture not in textures:
                textures.append(texture)
        if textures:
            return textures
        name = self.animations[model_name]['name'].lower().rstrip('0123456789')
        return sorted(t for t in self.store_textures if t.startswith(name))

    def figure_coefs(self, model_name):
        """
        Coefficients of the figure from one of its samples, figures_samples.json has a few variants for every sample.
        Samples of the "unknown" kind are taken only when the figure has no other samples.
        """
        figure_samples = self.samples.get(model_name, [])
        known = [sample for sample in figure_samples if sample.get('k') != 'unknown']
        variants = []
        for sample in (known or figure_samples):
            for coefs in (sample.get('c') or '').split(' '):
                if coefs:
                    variants.append([float(c) for c in coefs.split(':')])
        if not variants:
            return [0.5, 0.5, 0.5]
        return random.choice(variants)

    def is_armored(self, model_name):
        return any(sample.get('k') in ARMORED_KINDS for sample in self.samples.get(model_name, []))

    def showcase_models(self):
        """
        Figures which can be shown: animated, with the idle animation and with textures in the store.
        """
        result = []
        for model_name in sorted(self.animations.keys()):
            if model_name not in self.figures:
                continue
            action_types = set(a['action_type'].split(':')[0] for a in self.animations[model_name]['actions'] if a['animation_stage'] == 'cycle')
            if 'idle' not in action_types:
                continue
            if not self.is_armored(model_name) and not self.figure_textures(model_name):
                continue
            result.append(model_name)
        return result

    def random_wears(self, model_name):
        """
        Random set of armors, every one of them is of a random cut and material: the material must suit the armor
        and the store must have the texture of that armor for the figure, the figure must have the parts of the armor.
        """
        figure_parts = self.figure_parts(model_name)
        wears = []
        for armor_type, chance in ARMOR_CHANCES:
            if random.random() > chance:
                continue
            variants = []
            for armor_name, armor in self.armors.items():
                if armor['type'] != armor_type:
                    continue
                body_parts = armor_body_parts(armor_type, int(armor['texture_type_1']))
                if any(p not in figure_parts for p in body_parts if '.armor' in p):
                    continue
                for material_name, material in self.materials.items():
                    if material['type'] != armor['material_type']:
                        continue
                    if not self.has_texture(armor_texture_name(model_name, armor_type, armor, material)):
                        continue
                    variants.append(f'{armor_name}.{material_name}')
            if variants:
                wears.append(random.choice(variants))
        return wears

    def random_weapon(self, model_name):
        """
        Random weapon of a random material, or None: the figure must have all parts of the weapon
        and the store must have its texture.
        """
        if random.random() > WEAPON_CHANCE:
            return None
        figure_parts = self.figure_parts(model_name)
        variants = []
        for weapon_name, weapon_info in self.weapons.items():
            for material_name, material in self.materials.items():
                if material['type'] != weapon_info['material_type']:
                    continue
                weapon = f'{weapon_name}.{material_name}'
                _, parts, textures = weapon_parts(model_name, weapon, self.weapons, self.materials)
                if any(p.replace('.hidden', '') not in figure_parts for p in parts):
                    continue
                if not all(self.has_texture(t) for t in textures.values()):
                    continue
                variants.append(weapon)
        if not variants:
            return None
        return random.choice(variants)

    def random_skin(self, model_name):
        skins = sorted(t for t in self.store_textures if t.startswith(f'{model_name}skin_'))
        if not skins:
            return 0
        return int(random.choice(skins).split('_')[-1])

    def random_hair(self, model_name):
        hairs = sorted(p for p in self.figure_parts(model_name) if p.startswith('hr.'))
        if not hairs:
            return None
        return int(random.choice(hairs).split('.')[-1])

    def random_template(self, model_name=None):
        """
        Keyword arguments of models.Template of a figure with a random look, of a random type when model_name is None.
        """
        if model_name is None:
            model_name = random.choice(self.models)
        template = dict(model_name=model_name, store_id=self.store_id, coefs=self.figure_coefs(model_name))
        if self.is_armored(model_name):
            template['skin'] = self.random_skin(model_name)
            template['hair'] = self.random_hair(model_name)
            template['wears'] = self.random_wears(model_name)
            template['weapon'] = self.random_weapon(model_name)
        else:
            template['texture'] = random.choice(self.figure_textures(model_name))
        if _Debug:
            print(f'random look: {template}')
        return template
