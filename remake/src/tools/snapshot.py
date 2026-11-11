"""
Draws a posed figure or a map into a PNG file without OpenGL: a quick check of the readers and of the animation
on a machine without a display.

    PYTHONPATH=src venv/bin/python -m tools.snapshot figure unhuma uattack04 --frame 10 -o out.png
    PYTHONPATH=src venv/bin/python -m tools.snapshot map bz10k -o map.png
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import argparse

import numpy as np
from PIL import Image, ImageDraw

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import quat
from engine import figure

#------------------------------------------------------------------------------

LIGHT = np.array([0.4, -0.6, 0.7])

#------------------------------------------------------------------------------


def figure_triangles(model, frame, action_name):
    clip = model.clip(action_name) if action_name else None
    tracks = clip.tracks if clip else {}
    pose = model.skeleton.pose(tracks, frame)
    triangles = []
    for part_name in model.skeleton.visible:
        position, rotation = pose[part_name]
        vertices, indexes = model.figure(part_name).mesh_arrays(model.coefs)
        r = quat.to_matrix(rotation)
        world = vertices[:, 0:3] @ r.T + position
        normals = vertices[:, 3:6] @ r.T
        for tri in indexes.reshape(-1, 3):
            triangles.append((world[tri], normals[tri].mean(axis=0)))
    return triangles


def draw_triangles(triangles, size, yaw_degrees, file_path):
    yaw = math.radians(yaw_degrees)
    # camera looks along +y turned by yaw, z is up
    ca, sa = math.cos(yaw), math.sin(yaw)
    points = np.array([t[0] for t in triangles])
    flat = points.reshape(-1, 3)
    sx = flat[:, 0] * ca - flat[:, 1] * sa
    depth = flat[:, 0] * sa + flat[:, 1] * ca
    sy = flat[:, 2]
    lo = np.array([sx.min(), sy.min()])
    hi = np.array([sx.max(), sy.max()])
    k = (size - 40) / max(hi - lo)
    image = Image.new('RGB', (size, size), (40, 44, 52))
    draw = ImageDraw.Draw(image)
    screen = np.stack([20 + (sx - lo[0]) * k, size - 20 - (sy - lo[1]) * k], axis=1).reshape(-1, 3, 2)
    depth = depth.reshape(-1, 3).mean(axis=1)
    for i in np.argsort(-depth):
        n = triangles[i][1]
        length = np.linalg.norm(n)
        light = 0.25 + 0.75 * max(0.0, float(np.dot(n / length, LIGHT / np.linalg.norm(LIGHT)))) if length else 0.5
        c = int(60 + 180 * light)
        draw.polygon([tuple(p) for p in screen[i]], fill=(c, c, int(c * 0.9)))
    image.save(file_path)


def draw_map(game_data, map_name, file_path, scale=0.25):
    h = game_data.map_header(map_name)
    textures = [game_data.texture('%s%03d' % (map_name, i)) for i in range(h.textures_count)]
    tile = h.tile_size
    per_row = h.tiles_per_row
    cell = int(tile * scale)
    image = Image.new('RGB', (h.sectors_x * 16 * cell, h.sectors_y * 16 * cell))
    for sy in range(h.sectors_y):
        for sx in range(h.sectors_x):
            sector = game_data.map_sector(map_name, sx, sy)
            if sector is None:
                continue
            for ty in range(16):
                for tx in range(16):
                    texture_index, tile_index, rotation = sector.land_tiles[ty, tx]
                    if texture_index >= len(textures) or textures[texture_index] is None:
                        continue
                    # the tiles of a terrain texture are counted from its last row, see engine.terrain.tile_uv()
                    col, row = tile_index % per_row, per_row - 1 - tile_index // per_row
                    piece = textures[texture_index][row * tile:(row + 1) * tile, col * tile:(col + 1) * tile]
                    piece = np.rot90(piece, k=-int(rotation))
                    patch = Image.fromarray(piece[:, :, :3]).resize((cell, cell))
                    # y of the map goes up, rows of the image go down
                    image.paste(patch, ((sx * 16 + tx) * cell, image.height - (sy * 16 + ty + 1) * cell))
    image.save(file_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=['figure', 'map'])
    p.add_argument('name')
    p.add_argument('action', nargs='?', default='')
    p.add_argument('--frame', type=float, default=0.0)
    p.add_argument('--yaw', type=float, default=0.0)
    p.add_argument('--size', type=int, default=600)
    p.add_argument('--coefs', default='0.5,0.5,0.5')
    p.add_argument('-o', dest='output_file_path', metavar='OUTPUT', default='snapshot.png')
    args = p.parse_args()
    g = gamedata.GameData()
    if args.kind == 'figure':
        coefs = tuple(float(c) for c in args.coefs.split(','))
        model = figure.FigureModel(g, args.name, coefs=coefs)
        draw_triangles(figure_triangles(model, args.frame, args.action), args.size, args.yaw, args.output_file_path)
    else:
        draw_map(g, args.name, args.output_file_path)


if __name__ == '__main__':
    main()
