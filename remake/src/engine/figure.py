"""
A figure of a unit: which parts of the model are shown, their skeleton for the body proportions
and the clips of the model by the names of the actions.

A model (.mod) holds every variant of every part: "hd" is the bare head, "hd.armor05" is a helmet on it,
"rh3.sword02" is a sword in the right hand. A unit shows the base parts and replaces some of them
by the parts of its armor and weapon.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import animation
from engine import equipment as equipment_module

#------------------------------------------------------------------------------

# parts which belong to the weapons and to the quiver, they are hidden unless the unit holds such a weapon
EQUIPMENT_PREFIXES = (
    'r_shell', 'l_shell', 'bwpart', 'bwtetiva', 'baserh', 'basearrow', 'basepike', 'baseaxe', 'baseclub',
    'crbow', 'basesword', 'basedagger', 'quiver', 'arrows',
)

HUMANOID_MODELS = ('unhuma', 'unhufe', 'unorma', 'unorfe')

# the texture number of a part (fig header +36): 2 - the second texture of the object (OBJSECTXTR of the .mob),
# like the labels on the stones of the menu columns
SECOND_TEXTURE_NUMBER = 2
TEXTURE_SECOND = 'second'

#------------------------------------------------------------------------------


class FigureModel(object):

    def __init__(self, game_data, model_name, coefs=(0.0, 0.0, 0.0), height_coef=None, extra_parts=(), equipment=None,
                 parts=None):
        self.game_data = game_data
        self.model_name = model_name
        self.coefs = tuple(coefs)
        self.database = game_data.animation_database(model_name)
        self.parts_tree = game_data.parts_tree(model_name)
        self.equipment = equipment
        self.hidden_parts = set()
        if equipment is not None:
            # part name -> equipment.TEXTURE_BODY or TEXTURE_GEAR
            self.part_textures, self.hidden_parts = equipment.parts(self.parts_tree)
            self.part_names = [name for name in self.parts_tree.order if name in self.part_textures]
        elif parts is not None:
            # CInterfaceModel::ShowPart: only these parts are shown, their parents are posed
            self.part_names = [name for name in self.parts_tree.order if name in parts]
            self.part_textures = {}
        else:
            self.part_names = default_parts(self.parts_tree, model_name)
            self.part_textures = {}
        self.part_names += [p for p in extra_parts if p in self.parts_tree.parents and p not in self.part_names]
        height_coef = self.coefs[2] if height_coef is None else height_coef
        self.scale = self.database.size_scale(height_coef) if self.database else 1.0
        self.skeleton = animation.Skeleton(self.parts_tree, game_data.bones(model_name), self.part_names,
                                           coefs=self.coefs, scale=self.scale)
        self._clips = {}

    def __repr__(self):
        return 'FigureModel(%s, %d parts)' % (self.model_name, len(self.part_names))

    def actions(self):
        return self.database.actions if self.database else []

    def clip(self, action_or_name):
        """
        animation.Clip for an action of the .adb or for a clip name, None when the model has no such clip.
        """
        if isinstance(action_or_name, str):
            action = self.database.by_name(action_or_name) if self.database else None
            name = action_or_name
        else:
            action = action_or_name
            name = action.name
        if name not in self._clips:
            tracks = self.game_data.clip(self.model_name, name)
            self._clips[name] = animation.Clip(action, tracks) if tracks is not None else None
        return self._clips[name]

    def figure(self, part_name):
        return self.game_data.part_figure(self.model_name, part_name)

    def texture_group(self, part_name):
        """
        equipment.TEXTURE_BODY, TEXTURE_GEAR or TEXTURE_SECOND: which texture of the object the part uses.
        """
        if part_name in self.part_textures:
            return self.part_textures[part_name]
        if self.figure(part_name).texture_number == SECOND_TEXTURE_NUMBER:
            return TEXTURE_SECOND
        return equipment_module.TEXTURE_BODY

    def composed_textures(self):
        """
        equipment.TEXTURE_BODY / TEXTURE_GEAR -> (name, RGBA) of the textures the equipment builds
        from the layers of redress.res, empty without equipment.
        """
        if self.equipment is None:
            return {}
        result = {}
        for group, layers, size in (
            (equipment_module.TEXTURE_BODY, self.equipment.body_layers(), equipment_module.BODY_TEXTURE_SIZE),
            (equipment_module.TEXTURE_GEAR, self.equipment.gear_layers(), equipment_module.GEAR_TEXTURE_SIZE),
        ):
            if not layers:
                continue
            key = '+'.join(layers)
            rgba = self.game_data.cached(('composed', key), lambda: equipment_module.compose(
                [self.game_data.texture(name) for name in layers], size))
            result[group] = (key, rgba)
        return result

    def default_texture_name(self, skin=0):
        """
        The skin of the humans and orcs, for other figures the texture the maps use with this model most often.
        """
        skin_name = '%sskin_%02d' % (self.model_name, skin)
        if self.game_data.texture(skin_name) is not None:
            return skin_name
        for texture_name in self.game_data.model_textures().get(self.model_name, []):
            if texture_name != 'default0' and self.game_data.texture(texture_name) is not None:
                return texture_name
        return None

#------------------------------------------------------------------------------


def default_parts(parts_tree, model_name):
    parts = []
    for name in parts_tree.order:
        if '.' in name:
            continue
        if name.startswith(EQUIPMENT_PREFIXES):
            continue
        parts.append(name)
    if model_name in ('unhuma', 'unhufe') and 'hr.00' in parts_tree.parents:
        parts.append('hr.00')
    if not parts:
        parts = list(parts_tree.order)
    return parts
