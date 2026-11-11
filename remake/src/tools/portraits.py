"""
Draws the portraits of the characters of a scenario JSON (see tools.scenario) into PNG files without OpenGL:
the model of the unit of a character with its armor, weapon and textures, as the client dresses it,
lit from the front, cropped to the head and the shoulders for the humanoids and to the whole figure for others.

    EI_DATA_PATH=<data folder> PYTHONPATH=src venv/bin/python -m tools.portraits ei1.json -o portraits/

Every character with a unit gets <id>.png (the id with other signs than letters and digits made '_') (the portrait) and <id>_full.png (the whole figure).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import re
import sys
import math
import json
import argparse

import numpy as np
from PIL import Image

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import quat
from engine import figure
from engine import equipment

#------------------------------------------------------------------------------

LIGHT = np.array([0.35, -0.8, 0.5])

BACKGROUND = (0, 0, 0, 0)

# the part of the height of a humanoid (from the top) shown in its portrait
PORTRAIT_PART = 0.36

# the actions tried for the pose, the first one the model has wins
POSE_ACTIONS = ('ubriefing01', 'uidle01', 'uidle', 'cidle01', 'idle01', 'idle', 'ustand', 'stand')

SUPERSAMPLE = 2

#------------------------------------------------------------------------------


def pose_clip(model):
    names = [a.name for a in model.actions()]
    for wanted in POSE_ACTIONS:
        for name in names:
            if name.lower() == wanted:
                return model.clip(name)
    for name in names:
        if 'idle' in name.lower() or 'stand' in name.lower():
            return model.clip(name)
    return None


def build_model(game_data, unit):
    """
    The figure model of the unit dressed as the client does it, and its textures: group -> RGBA.
    """
    model_name = unit['model']
    coefs = tuple(unit.get('complection') or (0.5, 0.5, 0.5))
    gear = None
    if model_name in figure.HUMANOID_MODELS:
        items = game_data.items()
        armor_names, weapon_names = [], []
        for item_name in list(unit.get('armors') or []) + list(unit.get('weapons') or []):
            item = equipment.make_item(items, item_name)
            if item is not None:
                (weapon_names if item.is_weapon else armor_names).append(item_name)
        gear = equipment.from_unit(game_data, model_name, unit.get('prototype'), armor_names, weapon_names)
    model = figure.FigureModel(game_data, model_name, coefs=coefs, equipment=gear)
    textures = {group: rgba for group, (_key, rgba) in model.composed_textures().items()}
    texture_name = unit.get('texture') if unit.get('texture') not in (None, '', 'default0') else None
    if texture_name or equipment.TEXTURE_BODY not in textures:
        texture_name = texture_name or model.default_texture_name()
        rgba = game_data.texture(texture_name) if texture_name else None
        if rgba is not None:
            textures[equipment.TEXTURE_BODY] = rgba
    second_name = unit.get('second_texture')
    if second_name and second_name != 'default0':
        rgba = game_data.texture(second_name)
        if rgba is not None:
            textures[figure.TEXTURE_SECOND] = rgba
    return model, textures


def posed_triangles(model, textures):
    """
    Arrays of the triangles of the posed model: positions (n, 3, 3), normals (n, 3), uvs (n, 3, 2)
    and the index of the texture of each triangle into the returned list of textures.
    """
    clip = pose_clip(model)
    pose = model.skeleton.pose(clip.tracks if clip else {}, 0.0)
    texture_list = []
    texture_index = {}
    positions, normals, uvs, kinds = [], [], [], []
    for part_name in model.skeleton.visible:
        if part_name not in pose:
            continue
        position, rotation = pose[part_name]
        vertices, indexes = model.figure(part_name).mesh_arrays(model.coefs)
        if not len(indexes):
            continue
        r = quat.to_matrix(rotation)
        world = vertices[:, 0:3] @ r.T + position
        normal = vertices[:, 3:6] @ r.T
        triangles = indexes.reshape(-1, 3).astype(np.int64)
        group = model.texture_group(part_name)
        rgba = textures.get(group)
        if rgba is None:
            rgba = textures.get(equipment.TEXTURE_BODY)
        key = id(rgba)
        if key not in texture_index:
            texture_index[key] = len(texture_list)
            texture_list.append(rgba)
        positions.append(world[triangles])
        normals.append(normal[triangles].mean(axis=1))
        uvs.append(vertices[:, 6:8][triangles])
        kinds.append(np.full(len(triangles), texture_index[key]))
    if not positions:
        return None
    return np.concatenate(positions), np.concatenate(normals), np.concatenate(uvs), np.concatenate(kinds), texture_list


def sample(rgba, u, v):
    if rgba is None:
        return np.full(u.shape + (4, ), 160, dtype=np.float32)
    height, width = rgba.shape[0:2]
    x = np.mod(np.floor(u * width).astype(np.int64), width)
    y = np.mod(np.floor(v * height).astype(np.int64), height)
    return rgba[y, x].astype(np.float32)


def rasterize(triangles, yaw_degrees, size, crop_top=1.0):
    """
    RGBA image of the triangles seen from the side of yaw_degrees, z up; crop_top keeps that part of the
    height of the figure from its top.
    """
    positions, normals, uvs, kinds, texture_list = triangles
    yaw = math.radians(yaw_degrees)
    ca, sa = math.cos(yaw), math.sin(yaw)
    flat = positions.reshape(-1, 3)
    sx = flat[:, 0] * ca - flat[:, 1] * sa
    depth = flat[:, 0] * sa + flat[:, 1] * ca
    sy = flat[:, 2]
    top, bottom = sy.max(), sy.min()
    low = top - (top - bottom) * crop_top
    keep = sy >= low - (top - bottom) * 0.05
    span_x = (sx[keep].min(), sx[keep].max())
    center_x = (span_x[0] + span_x[1]) / 2.0
    height = top - low
    extent = max(height, span_x[1] - span_x[0]) * 1.08
    if crop_top < 1.0:
        # the portrait is framed by the head, a long weapon does not widen it
        head = sy >= top - (top - bottom) * 0.12
        center_x = (sx[head].min() + sx[head].max()) / 2.0
        extent = height * 1.08
    scale = size * SUPERSAMPLE / extent
    full = size * SUPERSAMPLE
    px = (sx - center_x) * scale + full / 2.0
    py = (top + extent * 0.04 - sy) * scale
    px = px.reshape(-1, 3)
    py = py.reshape(-1, 3)
    pz = depth.reshape(-1, 3)
    color = np.zeros((full, full, 4), dtype=np.float32)
    zbuffer = np.full((full, full), np.inf, dtype=np.float32)
    light = LIGHT / np.linalg.norm(LIGHT)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    shade = 0.45 + 0.65 * np.clip((normals / lengths[:, None]) @ light, 0.0, 1.0)
    for i in range(len(px)):
        x0, x1, x2 = px[i]
        y0, y1, y2 = py[i]
        minx = max(int(math.floor(min(x0, x1, x2))), 0)
        maxx = min(int(math.ceil(max(x0, x1, x2))), full - 1)
        miny = max(int(math.floor(min(y0, y1, y2))), 0)
        maxy = min(int(math.ceil(max(y0, y1, y2))), full - 1)
        if minx > maxx or miny > maxy:
            continue
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(minx, maxx + 1) + 0.5, np.arange(miny, maxy + 1) + 0.5)
        w0 = ((x1 - gx) * (y2 - gy) - (x2 - gx) * (y1 - gy)) / area
        w1 = ((x2 - gx) * (y0 - gy) - (x0 - gx) * (y2 - gy)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * pz[i, 0] + w1 * pz[i, 1] + w2 * pz[i, 2]
        region = zbuffer[miny:maxy + 1, minx:maxx + 1]
        visible = inside & (z < region)
        if not visible.any():
            continue
        u = w0 * uvs[i, 0, 0] + w1 * uvs[i, 1, 0] + w2 * uvs[i, 2, 0]
        v = w0 * uvs[i, 0, 1] + w1 * uvs[i, 1, 1] + w2 * uvs[i, 2, 1]
        texel = sample(texture_list[kinds[i]], u[visible], v[visible])
        # the transparent texels of the equipment layers are holes of the mesh, as the alpha test of the client
        solid = texel[:, 3] > 40
        if not solid.any():
            continue
        ys, xs = np.nonzero(visible)
        ys, xs = ys[solid], xs[solid]
        texel = texel[solid]
        region[ys, xs] = z[visible][solid]
        target = color[miny:maxy + 1, minx:maxx + 1]
        target[ys, xs, 0:3] = np.clip(texel[:, 0:3] * shade[i], 0, 255)
        target[ys, xs, 3] = 255
    image = Image.fromarray(color.astype(np.uint8), 'RGBA')
    return image.resize((size, size), Image.LANCZOS)


def front_yaw(triangles):
    """
    The figures look along +y in the files, the camera of the portraits looks at them from there,
    a bit to the side.
    """
    return 20.0


def draw_character(game_data, unit, portrait_file_path, full_file_path, size):
    try:
        model, textures = build_model(game_data, unit)
        triangles = posed_triangles(model, textures)
    except KeyError:
        # a weapon whose parts the model lacks (the arrow in the hand of some archers): drawn without it
        unit = dict(unit, weapons=[], prototype='')
        model, textures = build_model(game_data, unit)
        triangles = posed_triangles(model, textures)
    if triangles is None:
        return False
    yaw = front_yaw(triangles)
    humanoid = unit['model'] in figure.HUMANOID_MODELS
    rasterize(triangles, yaw, size, crop_top=PORTRAIT_PART if humanoid else 1.0).save(portrait_file_path)
    rasterize(triangles, yaw, size).save(full_file_path)
    return True


def file_base_name(character_id):
    return re.sub(r'[^0-9a-z_-]+', '_', character_id.lower()).strip('_')

#------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
    p.add_argument('scenario_file_path', metavar='SCENARIO_JSON')
    p.add_argument('-o', dest='output_dir_path', metavar='OUTPUT_DIR', required=True)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--only', default='', help='comma separated ids of the characters')
    args = p.parse_args()
    with open(args.scenario_file_path, encoding='utf-8') as f:
        scenario = json.load(f)
    game_data = gamedata.GameData()
    os.makedirs(args.output_dir_path, exist_ok=True)
    only = set(x for x in args.only.split(',') if x)
    done = 0
    for character in scenario['characters']:
        unit = character.get('unit')
        if not unit or not unit.get('model') or (only and character['id'] not in only):
            continue
        if not game_data.has_model(unit['model']):
            if _Debug:
                print('%s: no model %s' % (character['id'], unit['model']))
            continue
        base_name = file_base_name(character['id'])
        portrait_file_path = os.path.join(args.output_dir_path, base_name + '.png')
        full_file_path = os.path.join(args.output_dir_path, base_name + '_full.png')
        try:
            if draw_character(game_data, unit, portrait_file_path, full_file_path, args.size):
                done += 1
        except Exception as exc:
            if _Debug:
                print('%s: %r' % (character['id'], exc))
    if _Debug:
        print('%d portraits' % done)
    return 0


if __name__ == '__main__':
    sys.exit(main())
