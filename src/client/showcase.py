"""
Showcase of the figures of the units: a unit runs in a circle around the place where the hero stood,
every next() shows the next type of the figures with a random look.
Units of the showcase exist only on this client, the game server does not know about them.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import random

#------------------------------------------------------------------------------

from lib import models
from lib import res
from client import dat
from client import hero
from client import loader

#------------------------------------------------------------------------------

# radius of the circle in segments
CIRCLE_RADIUS = 1.0

# segments per second, as fast as the hero runs: models.Unit.max_speed per tick
RUN_SPEED = 0.06 * models.TICK_RATE

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

# kinds of the samples in figures_samples.json of the figures which wear armors
ARMORED_KINDS = ('human', 'orc')

# animations played while the unit runs in the circle, the first one the figure has
RUN_ACTION_TYPES = ('run', 'walk', 'idle')

#------------------------------------------------------------------------------


def figure_textures(model_name, catalog, samples, store_id):
    """
    Textures for a figure without armors: of its samples in figures_samples.json or, when it has no samples,
    of the store which are named as the figure in animations.json. Only textures which the store has.
    """
    textures = []
    for sample in samples.get(model_name, []):
        # samples with the list of parts are pieces of the figure, for example a golem falling apart
        if sample.get('p'):
            continue
        texture = sample['t'].lower()
        if loader.texture_location(texture, store_id) and texture not in textures:
            textures.append(texture)
    if textures:
        return textures
    name = catalog.animations[model_name]['name'].lower().rstrip('0123456789')
    return sorted(t for t in loader.store_index(store_id) if t.startswith(name))


def figure_coefs(model_name, samples):
    """
    Coefficients of the figure from one of its samples, figures_samples.json has a few variants for every sample.
    Samples of the "unknown" kind are taken only when the figure has no other samples.
    """
    figure_samples = samples.get(model_name, [])
    known = [sample for sample in figure_samples if sample.get('k') != 'unknown']
    variants = []
    for sample in (known or figure_samples):
        for coefs in (sample.get('c') or '').split(' '):
            if coefs:
                variants.append([float(c) for c in coefs.split(':')])
    if not variants:
        return [0.5, 0.5, 0.5]
    return random.choice(variants)


def is_armored(model_name, samples):
    return any(sample.get('k') in ARMORED_KINDS for sample in samples.get(model_name, []))


def random_wears(model_name, catalog, store_id):
    """
    Random set of armors, every one of them is of a random cut and material: the material must suit the armor
    and the store must have the texture of that armor for the figure, the figure must have the parts of the armor.
    """
    figure_parts = set(res.flat_tree(catalog.figures[model_name]))
    wears = []
    for armor_type, chance in ARMOR_CHANCES:
        if random.random() > chance:
            continue
        variants = []
        for armor_name, armor in catalog.armors.items():
            if armor['type'] != armor_type:
                continue
            body_parts = dat.armor_body_parts(armor_type, int(armor['texture_type_1']))
            if any(p not in figure_parts for p in body_parts if '.armor' in p):
                continue
            for material_name, material in catalog.materials.items():
                if material['type'] != armor['material_type']:
                    continue
                if not loader.texture_location(dat.armor_texture_name(model_name, armor_type, armor, material), store_id):
                    continue
                variants.append(f'{armor_name}.{material_name}')
        if variants:
            wears.append(random.choice(variants))
    return wears


def random_skin(model_name, store_id):
    skins = sorted(t for t in loader.store_index(store_id) if t.startswith(f'{model_name}skin_'))
    if not skins:
        return 0
    return int(random.choice(skins).split('_')[-1])


def random_hair(model_name, catalog):
    hairs = sorted(p for p in res.flat_tree(catalog.figures[model_name]) if p.startswith('hr.'))
    if not hairs:
        return None
    return int(random.choice(hairs).split('.')[-1])


def showcase_models(catalog, samples, store_id):
    """
    Figures which can be shown: animated, with the idle animation and with textures in the store.
    """
    result = []
    for model_name in sorted(catalog.animations.keys()):
        if model_name not in catalog.figures:
            continue
        action_types = set(a['action_type'].split(':')[0] for a in catalog.animations[model_name]['actions'] if a['animation_stage'] == 'cycle')
        if 'idle' not in action_types:
            continue
        if not is_armored(model_name, samples) and not figure_textures(model_name, catalog, samples, store_id):
            continue
        result.append(model_name)
    return result


def random_template(model_name, catalog, samples, store_id):
    """
    Keyword arguments of hero.Hero and the coefficients of a figure of that type with a random look.
    """
    template = dict(model_name=model_name, store_id=store_id)
    if is_armored(model_name, samples):
        template['skin'] = random_skin(model_name, store_id)
        template['hair'] = random_hair(model_name, catalog)
        template['wears'] = random_wears(model_name, catalog, store_id)
    else:
        template['texture'] = random.choice(figure_textures(model_name, catalog, samples, store_id))
    return template, figure_coefs(model_name, samples)

#------------------------------------------------------------------------------


class Showcase(object):
    """
    One unit which runs in a circle around the place where the hero was when next() was called.
    """

    def __init__(self, scene, catalog, samples, store_id=models.DEFAULT_STORE_ID):
        self.scene = scene
        self.catalog = catalog
        # figures_samples.json
        self.samples = samples
        self.store_id = store_id
        self.models = showcase_models(catalog, samples, store_id)
        # index in self.models of the figure shown now
        self.index = None
        # hero.Hero of the shown unit, it is set before the files of the figure are downloaded
        self.view = None
        self.center = None
        self.angle = 0.0

    def next(self):
        """
        Removes the shown unit and shows the next type of the figures, the first time the type of the hero.
        """
        hero_unit = self.scene.hero.get_unit() if self.scene.hero else None
        if not hero_unit or not self.models:
            return
        self.remove()
        if self.index is None:
            hero_model_name = self.scene.hero.model_name
            self.index = self.models.index(hero_model_name) if hero_model_name in self.models else 0
        else:
            self.index = (self.index + 1) % len(self.models)
        model_name = self.models[self.index]
        self.center = (hero_unit.w + hero_unit.shift_w, hero_unit.h + hero_unit.shift_h)
        # starts on the other side of the circle from the camera, behind the hero is not seen
        self.angle = math.radians(hero_unit.direction)
        template, coefs = random_template(model_name, self.catalog, self.samples, self.store_id)
        view = hero.Hero(scene=self.scene, **template)
        self.view = view
        if _Debug:
            print(f'showcase: figure {self.index + 1} of {len(self.models)} {template} coefs={coefs}')
        loader.StoreLoader(view.required_files(), on_done=lambda failed: self.on_files_ready(view, coefs, failed)).start()

    def on_files_ready(self, view, coefs, failed):
        if view is not self.view or not self.scene:
            # the button was pressed again while the files were downloading
            return
        if failed:
            print(f'ERROR, showcase figure {view.model_name} is not shown, files not downloaded: {failed}')
            self.view = None
            return
        x, y, direction = self.position()
        w = int(math.floor(x))
        h = int(math.floor(y))
        unit = view.create_unit(w, h, coefs=coefs, shift_w=x - w, shift_h=y - h, direction=direction)
        for action_type in RUN_ACTION_TYPES:
            if action_type in unit.action_types:
                view.set_animation(unit.action_types[action_type][0]['name'])
                break

    def position(self):
        """
        (x, y, direction) of the unit on the circle, it looks along the circle.
        """
        x = self.center[0] + CIRCLE_RADIUS * math.cos(self.angle)
        y = self.center[1] + CIRCLE_RADIUS * math.sin(self.angle)
        return x, y, math.degrees(self.angle) + 90.0

    def update(self, delta):
        """
        Called every frame: moves the unit along the circle.
        """
        u = self.view.get_unit() if self.view else None
        if not u:
            return
        self.angle = (self.angle + RUN_SPEED / CIRCLE_RADIUS * delta) % (2.0 * math.pi)
        x, y, direction = self.position()
        w = int(math.floor(x))
        h = int(math.floor(y))
        u.set_position(self.scene, w, h, x - w, y - h, direction)

    def remove(self):
        view = self.view
        self.view = None
        if view and view.unit_name in self.scene.units:
            self.scene.remove_unit_from_stage(container=self.scene.container_animated_objects, unit_name=view.unit_name)
