"""
The wounds drawn on a unit (game.exe CUnitClientSpecific::vf31 537360 -> 5370f0, 536ea0, 51c3d0 (partly)).

A unit has the 6 parts of the body of its type (the head, the torso, the two arms, the two legs). 536ea0 gives
every part a level by its health h of its maximum m:
- 0 for a whole part (h / m == 1), a part of the type with no health, or h / m >= 0.75;
- 1 below 0.75, 2 at 0.33 and below, 3 at h <= 1 (h / m <= 1 / m), a lost part too.
A humanoid (the class 0x32 of units.udb) keeps all six: "hd", "bd", "lh", "rh", "ll", "rl"; the helmet, the plate
(the torso and the arms) and the leggins whose shape has the byte +0xb4 (the tag 0x17 of items.idb) at 0 hide
the wounds of their parts. Other units have one layer for both arms "h" and one for both legs "l": the pair
takes the higher level of its two parts (the layer is then named twice).

5370f0 composes the layers "<model><part>w<level>.mmp" of the parts with a level into the texture "mo%dw"
(128 x 128 for a humanoid, 64 x 64 for others), the empty places transparent; 51c3d0 draws it as the second
pass of the figure with the same coordinates, blended by its alpha (5, 6: D3DBLEND_SRCALPHA, INVSRCALPHA).
vf31 runs after a blow and a healing; with no wounds 51c920 drops the texture.
The remake draws the composed wounds over the body texture, stretched to its size.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import numpy as np

#------------------------------------------------------------------------------

from eilib import idb
from engine import equipment

#------------------------------------------------------------------------------

PARTS_COUNT = 6

# 78f974 and 78f990: the names of the layers of the parts
HUMANOID_PART_NAMES = ('hd', 'bd', 'lh', 'rh', 'll', 'rl')
OTHER_PART_NAMES = ('hd', 'bd', 'h', 'h', 'l', 'l')

# 5370f0: "%s%sw%d.mmp" of the model, the part and the level
LAYER_NAME = '%s%sw%d'

HUMANOID_SIZE = 128
OTHER_SIZE = 64

# 536ea0: 73c524, 73d5a4
LIGHT_LIMIT = 0.75
HEAVY_LIMIT = 0.33

# 536ea0: an armor whose shape does not show the wounds clears the levels of its parts
ARMOR_COVERS = (
    (idb.ARMOR_LEGGINS, (4, 5)),
    (idb.ARMOR_PLATE, (1, 2, 3)),
    (idb.ARMOR_HELM, (0, )),
)

#------------------------------------------------------------------------------


def part_level(health, max_health):
    """
    536ea0: the level 0..3 of the wounds of a part.
    """
    if max_health <= 0.0:
        return 0
    ratio = health / max_health
    if ratio == 1.0:
        return 0
    if ratio <= 1.0 / max_health:
        return 3
    if ratio <= HEAVY_LIMIT:
        return 2
    if ratio < LIGHT_LIMIT:
        return 1
    return 0


def levels(parts, humanoid, armors=None):
    """
    The 6 levels of the parts of the body (engine.combat.BodyPart), armors are {slot: engine.equipment.Item}.
    """
    result = [0] * PARTS_COUNT
    for index, part in enumerate(parts[:PARTS_COUNT]):
        result[index] = part_level(part.health, part.max_health)
    if not humanoid:
        result[4] = result[5] = max(result[4], result[5])
        result[2] = result[3] = max(result[2], result[3])
        return tuple(result)
    for slot, covered in ARMOR_COVERS:
        item = (armors or {}).get(slot)
        if item is not None and not getattr(item.shape, 'shows_wounds', True):
            for index in covered:
                result[index] = 0
    return tuple(result)


def layer_names(model_name, wound_levels, humanoid):
    """
    5370f0: the names of the layers of the wounds, in the order of the parts.
    """
    part_names = HUMANOID_PART_NAMES if humanoid else OTHER_PART_NAMES
    return [LAYER_NAME % (model_name, part_names[index], level)
            for index, level in enumerate(wound_levels) if level]


def stretch(rgba, size):
    """
    The image stretched to size x size with the linear filter of the second pass.
    """
    height, width = rgba.shape[:2]
    if height == size and width == size:
        return rgba
    source = rgba.astype(np.float32)
    ys = np.clip((np.arange(size) + 0.5) * height / size - 0.5, 0.0, height - 1.0)
    xs = np.clip((np.arange(size) + 0.5) * width / size - 0.5, 0.0, width - 1.0)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    fy = (ys - y0)[:, None, None]
    fx = (xs - x0)[None, :, None]
    top = source[y0][:, x0] * (1.0 - fx) + source[y0][:, x1] * fx
    bottom = source[y1][:, x0] * (1.0 - fx) + source[y1][:, x1] * fx
    return np.clip(top * (1.0 - fy) + bottom * fy + 0.5, 0, 255).astype(np.uint8)


def wounded(base, layers, humanoid):
    """
    The body texture base with the wounds over it: the layers composed like 5370f0 does and blended
    by their alpha like the second pass of 51c3d0 (partly).
    """
    size = HUMANOID_SIZE if humanoid else OTHER_SIZE
    wounds = stretch(equipment.compose(layers, size), base.shape[0])
    if wounds.shape[:2] != base.shape[:2]:
        return base
    alpha = wounds[..., 3:4].astype(np.float32) / 255.0
    result = base.astype(np.float32)
    result[..., 0:3] = wounds[..., 0:3] * alpha + result[..., 0:3] * (1.0 - alpha)
    return result.astype(np.uint8)
