"""
The armor and the weapon of a unit: which parts of the model are shown and which textures they use,
as the client part of a unit in game.exe does it.

Parts (535cb0 and CUnitClientSpecific vf26 at 5381b0):
- the body is 15 parts and the hair "hr.NN" of the prototype (no hair for -1);
- the armor of the slots 0, 1, 2 (helm, plate, leggins) changes the parts of a row of ARMOR_PARTS:
  "<part>.armorNN" replaces "<part>", but the helmet "hd.armorNN" is put over the head which stays,
  and a helmet hides the hair because no model has the parts "hr.NN.armorNN";
- the weapon adds its parts (6a0fb0), the arrow in the hand of an archer is hidden until it is shot (vf25);
- "base*" parts are only points on the skeleton (535770 reads their positions), "l_shell" and "r_shell"
  are never named by game.exe, these parts are not shown.

Textures (536670): a unit has two textures built from layers of redress.res drawn over each other (692fd0):
- the body "mo%dp" 256x256: the skin "<model>skin_NN", then the shirt, the pants, the boots, the gloves,
  then the plate and the leggins, the one with the smaller "layer" of its shape first;
- the gear "mo%ds" 128x128: the helmet, then the weapon in the hands.
The helmet parts and the weapon parts use the gear texture, all other parts use the body texture.
The name of a layer is "<model>" + "%s_%.2d.%s.%d" % (kind of the shape, number, code of the material, variant).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import numpy as np

#------------------------------------------------------------------------------

from eilib import adb
from eilib import idb

#------------------------------------------------------------------------------

# game.exe 0x73ec24: 3 rows of 7 names of 4 bytes, the row of an armor slot is chosen in vf26
ARMOR_PARTS = (
    ('bd', 'rh1', 'rh2', 'rh3', 'lh1', 'lh2', 'lh3'),
    ('hp', 'rl1', 'rl2', 'rl3', 'll1', 'll2', 'll3'),
    ('hr', 'hd'),
)

# armor slot -> row of ARMOR_PARTS
ARMOR_SLOT_ROWS = {idb.ARMOR_HELM: 2, idb.ARMOR_PLATE: 0, idb.ARMOR_LEGGINS: 1}

# the layers of the body texture under the plate and the leggins, in the order of 536670
UNDER_LAYER_SLOTS = (idb.ARMOR_SHIRT, idb.ARMOR_PANTS, idb.ARMOR_BOOTS, idb.ARMOR_GLOVES)

ARROW_IN_HAND = 'rh3.arrow00'

# 5341a0: the weapon bits of an action (+0x14 of a record of the .adb) for ShowWeapon
ACTION_WEAPON_BITS = 0x7fff

# the weapon types of items.idb are in the order of the weapon bits of the .adb
WEAPON_ANIMATIONS = adb.WEAPONS

BODY_TEXTURE_SIZE = 256
GEAR_TEXTURE_SIZE = 128

TEXTURE_BODY = 'body'
TEXTURE_GEAR = 'gear'

#------------------------------------------------------------------------------


class Item(object):

    def __init__(self, name, shape, material):
        self.name = name
        self.shape = shape
        self.material = material
        # +0x24 and +0x20 (6a5d20): the durability of the shape times that of the material, an item starts whole
        factor = getattr(material, 'durability', 1.0) if material is not None else 1.0
        self.max_durability = float(getattr(shape, 'durability', 0.0) or 0.0) * factor
        self.durability = self.max_durability
        # +0x48: the things in a stack share the wear (5253e0), 1 for a weapon or an armor
        self.count = 1

    def __repr__(self):
        return 'Item(%r)' % self.name

    @property
    def condition(self):
        """
        The share 0..1 of the durability which the item still has, 1 for an item without durability.
        """
        if self.max_durability <= 0.0:
            return 1.0
        return min(max(self.durability / self.max_durability, 0.0), 1.0)

    @property
    def is_broken(self):
        """
        523300 adds to the armor only the items whose rounded durability is above 0.
        """
        return self.max_durability > 0.0 and int(round(self.durability)) < 1

    @property
    def is_weapon(self):
        return isinstance(self.shape, idb.WeaponShape)

    @property
    def slot(self):
        return self.shape.type

    def _factor(self, name):
        return getattr(self.material, name) if self.material is not None else 1.0

    @property
    def weight(self):
        """
        6a5d20: the weight of the shape times the weight factor of the material.
        """
        return self.shape.weight * self._factor('weight_factor')

    @property
    def damage(self):
        """
        (min, spread) of a weapon: those of the shape times the damage factor of the material.
        """
        factor = self._factor('damage_factor')
        return self.shape.damage_min * factor, self.shape.damage_spread * factor

    @property
    def damage_types(self):
        return tuple(self.shape.damage_types)

    def absorption(self, limb=False):
        """
        The absorption of an armor against the 7 types of the damage: for every type the value of the shape
        times the factor of the material times the scale of the shape (6a5d20); limb gives the second set
        (the hit locations of the armor after the first one).
        """
        scale, values, _ = self.shape.limb_absorption if limb else self.shape.absorption
        factors = self.material.absorption_factors if self.material is not None else (1.0, ) * len(values)
        return tuple(v * f * scale for v, f in zip(values, factors))

    def texture_name(self, model_name):
        kinds = idb.WEAPON_KINDS if self.is_weapon else idb.ARMOR_KINDS
        if not 0 <= self.shape.type < len(kinds):
            return None
        code = self.material.code if self.material else ''
        return '%s%s_%02d.%s.%d' % (model_name, kinds[self.shape.type], self.shape.number, code, self.shape.variant)


class Equipment(object):

    def __init__(self, model_name, armors=(), weapon=None, skin=0, hair=-1):
        self.model_name = model_name
        # slot -> Item, a later item of the same slot replaces the earlier one like CUnit puts it on
        self.armors = {}
        for item in armors:
            self.armors[item.slot] = item
        self.weapon = weapon
        self.skin = skin
        self.hair = hair

    def __repr__(self):
        return 'Equipment(%s, %s, %s)' % (self.model_name, sorted(self.armors), self.weapon)

    def items(self):
        """
        The worn items: the armor by its slot, then the weapon.
        """
        items = [item for _, item in sorted(self.armors.items())]
        return items + ([self.weapon] if self.weapon is not None else [])

    def keep_wear(self, worn_items):
        """
        The items take the durability of the items of the same names of worn_items (name -> Item, the items
        which the unit wore before its equipment was built again). Returns the new name -> Item.
        """
        result = {}
        for item in self.items():
            old = worn_items.get(item.name)
            if old is not None and old is not item:
                item.durability = min(old.durability, item.max_durability)
            result[item.name] = item
        return result

    @property
    def weapon_name(self):
        """
        The name of the weapon bit of the .adb (SWORD, AXE...) for the choice of the animations.
        """
        if self.weapon is None:
            return None
        weapon_type = self.weapon.shape.type
        return WEAPON_ANIMATIONS[weapon_type] if 0 <= weapon_type < len(WEAPON_ANIMATIONS) else None

    def hair_part(self):
        return 'hr.%02d' % self.hair if self.hair >= 0 else None

    def parts(self, parts_tree):
        """
        part name -> TEXTURE_BODY or TEXTURE_GEAR for every shown part, plus the set of the parts
        which are there but hidden (the arrow in the hand).
        """
        known = parts_tree.parents
        shown = {}
        for row in ARMOR_PARTS:
            for entry in row:
                name = self.hair_part() if entry == 'hr' else entry
                if name and name in known:
                    shown[name] = TEXTURE_BODY
        for slot, row_index in sorted(ARMOR_SLOT_ROWS.items()):
            item = self.armors.get(slot)
            if item is None:
                continue
            texture = TEXTURE_GEAR if slot == idb.ARMOR_HELM else TEXTURE_BODY
            for entry in ARMOR_PARTS[row_index]:
                base_name = self.hair_part() if entry == 'hr' else entry
                if base_name is None:
                    continue
                armor_name = '%s.armor%02d' % (base_name, item.shape.number)
                if armor_name in known:
                    shown[armor_name] = texture
                    if entry != 'hd':
                        shown.pop(base_name, None)
                elif entry == 'hr':
                    shown.pop(base_name, None)
        hidden = set()
        for name in weapon_parts(self.weapon):
            if name in known:
                shown[name] = TEXTURE_GEAR
        if self.weapon is not None and self.weapon.shape.type == idb.WEAPON_BOW and ARROW_IN_HAND in shown:
            hidden.add(ARROW_IN_HAND)
        return shown, hidden

    def body_layers(self):
        layers = [self.model_name + 'skin_%02d' % max(self.skin, 0)]
        for slot in UNDER_LAYER_SLOTS:
            if slot in self.armors:
                layers.append(self.armors[slot].texture_name(self.model_name))
        plate = self.armors.get(idb.ARMOR_PLATE)
        leggins = self.armors.get(idb.ARMOR_LEGGINS)
        top = [item for item in (plate, leggins) if item is not None]
        if len(top) == 2 and leggins.shape.layer < plate.shape.layer:
            top = [leggins, plate]
        layers.extend(item.texture_name(self.model_name) for item in top)
        return [name for name in layers if name]

    def gear_layers(self):
        layers = []
        helm = self.armors.get(idb.ARMOR_HELM)
        if helm is not None:
            layers.append(helm.texture_name(self.model_name))
        if self.weapon is not None:
            layers.append(self.weapon.texture_name(self.model_name))
        return [name for name in layers if name]

#------------------------------------------------------------------------------


def make_item(item_database, item_name):
    """
    CItemManager::CreateItem (6acfc0): the shape and the material of items.idb, then BuildItem (6a5d20).
    """
    shape, material = item_database.item(item_name)
    if shape is None:
        if _Debug:
            print('unknown item %r' % item_name)
        return None
    return Item(item_name, shape, material)


def from_unit(game_data, model_name, prototype_name=None, armor_names=(), weapon_names=()):
    """
    Equipment of a unit of a map: the skin and the hair come from its prototype in units.udb,
    the items from the lists of the .mob file (or of the prototype when the map gives no weapon).
    The first weapon is in the hands.
    """
    items = game_data.items()
    prototype = game_data.units().prototype(prototype_name) if prototype_name else None
    skin = prototype.skin if prototype else 0
    hair = prototype.hair if prototype else default_hair(model_name)
    if not weapon_names and prototype is not None:
        weapon_names = prototype.weapons
    armors = [a for a in (make_item(items, name) for name in armor_names) if a is not None and not a.is_weapon]
    weapons = [w for w in (make_item(items, name) for name in weapon_names) if w is not None and w.is_weapon]
    return Equipment(model_name, armors=armors, weapon=weapons[0] if weapons else None, skin=skin, hair=hair)


def default_hair(model_name):
    return 0 if model_name == 'unhuma' else (1 if model_name == 'unhufe' else -1)


def weapon_parts(weapon):
    """
    The parts of a weapon in the hands, game.exe 6a0fb0.
    """
    if weapon is None:
        return []
    number = '%02d' % weapon.shape.number
    kind = weapon.shape.type
    if kind == idb.WEAPON_SWORD:
        return ['rh3.sword' + number]
    if kind == idb.WEAPON_AXE:
        return ['rh3.axe' + number]
    if kind == idb.WEAPON_DAGGER:
        return ['rh3.dagger' + number]
    if kind == idb.WEAPON_PIKE:
        return ['rh3.pike' + number]
    if kind == idb.WEAPON_CLUB:
        return ['rh3.club' + number]
    if kind == idb.WEAPON_BOW:
        return ['lh3.bwpartb' + number, 'bwparta' + number, 'bwtetivaa' + number, 'bwtetivab' + number,
                'quiver', 'arrows', ARROW_IN_HAND]
    if kind == idb.WEAPON_CROSSBOW:
        prefix = 'crbow' + number
        return ['rh3.%smain' % prefix, prefix + 'part01', prefix + 'part02', prefix + 'tetiva01', prefix + 'tetiva02',
                'quiver', 'arrows']
    return []


def weapon_hidden_parts(unit):
    """
    CUnitClientSpecific::ShowWeapon 537cb0 (from vf06 5341a0 (partly) on every tick): the parts of the weapon item
    in the hands are hidden while the action played has none of the weapon bits (the bits 0x7fff of its
    packed field), an empty set otherwise.
    """
    figure = getattr(unit, 'figure', None)
    equipment = getattr(figure, 'equipment', None) if figure is not None else None
    if equipment is None or equipment.weapon is None:
        return frozenset()
    player = getattr(unit, 'player', None)
    clip = getattr(player, 'clip', None) if player is not None else None
    action = getattr(clip, 'action', None) if clip is not None else None
    if action is None or action.packed & ACTION_WEAPON_BITS:
        return frozenset()
    return frozenset(weapon_parts(equipment.weapon))


def arrow_in_hand_visible(unit):
    """
    538c20: the arrow in the hand of a bowman is shown between the two frames of the action of the shot
    (adb +0x28 and +0x2c) when the bit 1 of its probability (+0x18) is set, and hidden otherwise.
    """
    figure = getattr(unit, 'figure', None)
    if figure is None or ARROW_IN_HAND not in figure.hidden_parts:
        return True
    equipment = getattr(figure, 'equipment', None)
    if equipment is None or equipment.weapon is None:
        return False
    if equipment.weapon.shape.type != idb.WEAPON_BOW:
        return False
    player = getattr(unit, 'player', None)
    clip = getattr(player, 'clip', None) if player is not None else None
    action = getattr(clip, 'action', None) if clip is not None else None
    if action is None or not (action.probability & 1):
        return False
    shown, hidden = action.show_hide_frames
    time = float(getattr(player, 'time', 0.0))
    if hidden <= shown:
        return time >= shown
    return shown <= time < hidden


def compose(layers, size):
    """
    One RGBA image of the layers drawn over each other in their order, the empty places are transparent
    (CTextureManager::ComposeLayers 693140).
    layers are (height, width, 4) uint8 arrays, a layer of another size is skipped like game.exe 692fd0 does.
    """
    result = np.zeros((size, size, 4), dtype=np.float32)
    for rgba in layers:
        if rgba is None or rgba.shape[0] != size or rgba.shape[1] != size:
            continue
        layer = rgba.astype(np.float32)
        alpha = layer[..., 3:4] / 255.0
        result[..., 0:3] = layer[..., 0:3] * alpha + result[..., 0:3] * (1.0 - alpha)
        result[..., 3:4] = np.maximum(result[..., 3:4], layer[..., 3:4])
    return result.astype(np.uint8)
