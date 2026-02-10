import os
import sys
import json
import logging
import hashlib
import shutil
import pprint

import numpy as np
import imagehash
from PIL import Image, ImageChops
from PIL.Image import Transpose
from skimage.metrics import structural_similarity  # @UnresolvedImport


logging.getLogger("PIL").propagate = False

TILE_SIZE = 64
CORNER_SIZE = 8
hash_size = 8


index_by_hash = {}
index_by_ahash = {}
index_by_dhash = {}
index_by_phash = {}
index_by_whash = {}
tile_types = {}
tile_average_colors = {}
tile_types_images = {}
tile_types_corners = {}
tile_types_corners_average_color = {}
tile_types_corners_hash = {}
tile_types_sides = {}
tile_types_variants = {}
DIFF_CACHE = {}
HASH_DIFF_CACHE = {}


def image_tile(source_image, w, h):
    tile_image = source_image.crop((w * 64, h * 64, (w + 1) * 64, (h + 1) * 64))
    return tile_image, hashlib.md5(tile_image.tobytes()).hexdigest()


def image_phash(file_path, to_str=True):
    img = Image.open(file_path)
    if to_str:
        return str(imagehash.phash(img, hash_size=hash_size))
    return imagehash.phash(img, hash_size=hash_size)


def image_dhash(file_path, to_str=True):
    img = Image.open(file_path)
    if to_str:
        return str(imagehash.dhash(img, hash_size=hash_size))
    return imagehash.dhash(img, hash_size=hash_size)


def image_whash(file_path, to_str=True):
    img = Image.open(file_path)
    if to_str:
        return str(imagehash.whash(img, hash_size=hash_size))
    return imagehash.whash(img, hash_size=hash_size)


def image_ahash(file_path, to_str=True):
    img = Image.open(file_path)
    if to_str:
        return str(imagehash.average_hash(img, hash_size=hash_size))
    return imagehash.average_hash(img, hash_size=hash_size)


def image_hash(img, to_str=False, hash_type='phash'):
    if hash_type == 'ahash':
        h = imagehash.average_hash(img, hash_size=hash_size)
    elif hash_type == 'phash':
        h = imagehash.phash(img, hash_size=hash_size)
    elif hash_type == 'dhash':
        h = imagehash.dhash(img, hash_size=hash_size)
    elif hash_type == 'whash':
        h = imagehash.whash(img, hash_size=hash_size)
    if to_str:
        return str(h)
    return h


def average_color(image_path):
    img = Image.open(image_path).convert('RGBA')
    arr = np.array(img)
    mask = arr[..., 3] > 0
    if not mask.any():
        return (0, 0, 0)
    rgb = arr[..., :3][mask]
    avg = rgb.mean(axis=0)
    return (int(avg[0]), int(avg[1]), int(avg[2]))


def average_color_in_memory(image1):
    img = image1.convert('RGBA')
    arr = np.array(img)
    mask = arr[..., 3] > 0
    if not mask.any():
        return (0, 0, 0)
    rgb = arr[..., :3][mask]
    avg = rgb.mean(axis=0)
    return (int(avg[0]), int(avg[1]), int(avg[2]))


def color_distance(c1, c2):
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])


def add_to_index(tile_hash, map_name, map_image_variant, w, h):
    global index_by_hash
    if tile_hash in index_by_hash:
        return
    index_by_hash[tile_hash] = (map_name, map_image_variant, w, h)


def rename_files():
    global index_by_ahash
    global index_by_phash
    dest_dir = sys.argv[2]
    index_by_ahash = json.loads(open('tiles_ahash.json', 'r').read())
    for map_name in index_by_ahash.keys():
        sorted_hashes = index_by_ahash[map_name]['hashes']
        for pos in range(len(sorted_hashes)):
            ahash = sorted_hashes[pos]
            entries = index_by_ahash[map_name]['files'][ahash]
            for entry in entries:
                map_name, map_image_variant, w, h = entry
                src_file_path = os.path.abspath(os.path.join(dest_dir, map_name, 'ahash', f'{ahash}_{map_image_variant}_{w}_{h}.png'))
                dest_file_path = os.path.abspath(os.path.join(dest_dir, map_name, 'ahash', f'{pos:04d}_{map_image_variant}_{w}_{h}.png'))
                os.rename(src_file_path, dest_file_path)
    index_by_phash = json.loads(open('tiles_phash.json', 'r').read())
    for map_name in index_by_phash.keys():
        sorted_hashes = index_by_phash[map_name]['hashes']
        for pos in range(len(sorted_hashes)):
            phash = sorted_hashes[pos]
            entries = index_by_phash[map_name]['files'][phash]
            for entry in entries:
                map_name, map_image_variant, w, h = entry
                src_file_path = os.path.abspath(os.path.join(dest_dir, map_name, 'phash', f'{phash}_{map_image_variant}_{w}_{h}.png'))
                dest_file_path = os.path.abspath(os.path.join(dest_dir, map_name, 'phash', f'{pos:04d}_{map_image_variant}_{w}_{h}.png'))
                os.rename(src_file_path, dest_file_path)


def images_diff_score_in_memory(image1, image2, win_size=None, min_diff=None, round_digits=None):
    img1 = image1.convert("RGB")
    img1_array = np.array(img1)
    img2 = image2.convert("RGB")
    img2_array = np.array(img2)
    score = structural_similarity(img1_array, img2_array, multichannel=True, data_range=255, win_size=win_size, channel_axis=-1)
    # score = structural_similarity(img1_array, img2_array, multichannel=True, data_range=255, win_size=win_size, channel_axis=-1)
    v = float(score)
    if round_digits is not None:
        v = round(v, round_digits)
    if min_diff is not None:
        return max(v, min_diff)
    return v


def images_diff_score(img1_path, img2_path):
    global DIFF_CACHE
    cache_key1 = (os.path.basename(img1_path).replace('.png', ''), os.path.basename(img2_path).replace('.png', ''))
    cache_key2 = (os.path.basename(img2_path).replace('.png', ''), os.path.basename(img1_path).replace('.png', ''))
    if cache_key1 in DIFF_CACHE:
        return DIFF_CACHE[cache_key1]
    if cache_key2 in DIFF_CACHE:
        return DIFF_CACHE[cache_key1]
    img1 = Image.open(img1_path).convert("RGB")
    img1_array = np.array(img1)
    img2 = Image.open(img2_path).convert("RGB")
    img2_array = np.array(img2)
    score = structural_similarity(img1_array, img2_array, multichannel=True, data_range=255, win_size=3, channel_axis=-1)
    # score = ssim(img1_array, img2_array, data_range=255, win_size=3, channel_axis=2)
    DIFF_CACHE[cache_key1] = float(score)
    DIFF_CACHE[cache_key2] = float(score)
    return round(float(score), 3)


def images_diff_score_grayscale(path1, path2):
    global DIFF_CACHE
    cache_key1 = (os.path.basename(path1).replace('.png', ''), os.path.basename(path2).replace('.png', ''))
    cache_key2 = (os.path.basename(path2).replace('.png', ''), os.path.basename(path1).replace('.png', ''))
    if cache_key1 in DIFF_CACHE:
        return DIFF_CACHE[cache_key1]
    if cache_key2 in DIFF_CACHE:
        return DIFF_CACHE[cache_key1]
    img1 = Image.open(path1).convert('L')
    img2 = Image.open(path2).convert('L')
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    score, diff = structural_similarity(img1_array, img2_array, full=True, data_range=255)
    DIFF_CACHE[cache_key1] = float(score)
    DIFF_CACHE[cache_key2] = float(score)
    return float(score)


def images_hash_diff(path1, path2, hash_type='ahash'):
    global HASH_DIFF_CACHE
    cache_key1 = (os.path.basename(path1).replace('.png', ''), os.path.basename(path2).replace('.png', ''))
    cache_key2 = (os.path.basename(path2).replace('.png', ''), os.path.basename(path1).replace('.png', ''))
    if cache_key1 in HASH_DIFF_CACHE:
        return HASH_DIFF_CACHE[cache_key1]
    if cache_key2 in HASH_DIFF_CACHE:
        return HASH_DIFF_CACHE[cache_key1]
    if hash_type == 'ahash':
        hsh1 = image_ahash(path1, to_str=False)
        hsh2 = image_ahash(path2, to_str=False)
    elif hash_type == 'phash':
        hsh1 = image_phash(path1, to_str=False)
        hsh2 = image_phash(path2, to_str=False)
    elif hash_type == 'whash':
        hsh1 = image_whash(path1, to_str=False)
        hsh2 = image_whash(path2, to_str=False)
    elif hash_type == 'dhash':
        hsh1 = image_dhash(path1, to_str=False)
        hsh2 = image_dhash(path2, to_str=False)
    diff = abs(hsh1 - hsh2)
    HASH_DIFF_CACHE[cache_key1] = diff
    HASH_DIFF_CACHE[cache_key2] = diff
    return diff


def get_rotated_image(image, rotation):
    if rotation == 0:
        return image.copy()
    elif rotation == 90:
        return image.transpose(Transpose.ROTATE_90)
    elif rotation == 180:
        return image.transpose(Transpose.ROTATE_180)
    elif rotation == 270:
        return image.transpose(Transpose.ROTATE_270)
    elif rotation == -1:
        return image.transpose(Transpose.FLIP_LEFT_RIGHT)
    elif rotation == -2:
        return image.transpose(Transpose.FLIP_TOP_BOTTOM)


def get_corner_image(image, corner, size=CORNER_SIZE):
    if corner == 'top_left':
        box = (0, 0, size, size)
    elif corner == 'top_right':
        box = (TILE_SIZE - size, 0, TILE_SIZE, size)
    elif corner == 'bottom_right':
        box = (TILE_SIZE - size, TILE_SIZE - size, TILE_SIZE, TILE_SIZE)
    elif corner == 'bottom_left':
        box = (0, TILE_SIZE - size, size, TILE_SIZE)
    corner_image = image.crop(box)
    return corner_image


def get_half_image(image, half):
    if half == 'top':
        box = (0, 0, TILE_SIZE, int(TILE_SIZE / 2.0))
    elif half == 'bottom':
        box = (0, int(TILE_SIZE / 2.0), TILE_SIZE, TILE_SIZE)
    elif half == 'left':
        box = (0, 0, int(TILE_SIZE / 2.0), TILE_SIZE)
    elif half == 'right':
        box = (int(TILE_SIZE / 2.0), 0, TILE_SIZE, TILE_SIZE)
    half_image = image.crop(box)
    return half_image


def get_corner_image_bk(image, corner):
    corner_x = 0 if corner in ['top_left', 'bottom_left'] else TILE_SIZE - CORNER_SIZE
    corner_y = 0 if corner in ['top_left', 'top_right'] else TILE_SIZE - CORNER_SIZE
    corner_crop = (corner_x, corner_y, corner_x + CORNER_SIZE, corner_y + CORNER_SIZE)
    corner_image = image.crop(corner_crop)
    return corner_image


def get_side_image_simple(image, side, size=1, final_size=7):
    if side == 'top':
        box = (0, 0, TILE_SIZE, size)
        sz = (TILE_SIZE, final_size)
    elif side == 'bottom':
        box = (0, TILE_SIZE - size, TILE_SIZE, TILE_SIZE)
        sz = (TILE_SIZE, final_size)
    elif side == 'left':
        box = (0, 0, size, TILE_SIZE)
        sz = (final_size, TILE_SIZE)
    elif side == 'right':
        box = (TILE_SIZE - size, 0, TILE_SIZE, TILE_SIZE)
        sz = (final_size, TILE_SIZE)
    side_image = image.crop(box)
    im = Image.new("RGB", sz, "white")
    if side in ['top', 'bottom']:
        for i in range(final_size):
            im.paste(side_image, (0, i))
    else:
        for i in range(final_size):
            im.paste(side_image, (i, 0))
    return im


def get_side_image_deep(image, side, size=7):
    if side == 'top':
        box = (0, 0, TILE_SIZE, size)
    elif side == 'bottom':
        box = (0, TILE_SIZE - size, TILE_SIZE, TILE_SIZE)
    elif side == 'left':
        box = (0, 0, size, TILE_SIZE)
    elif side == 'right':
        box = (TILE_SIZE - size, 0, TILE_SIZE, TILE_SIZE)
    side_image = image.crop(box)
    return side_image


def get_two_side_images(image, side, size=1, final_size=7):
    TILE_SIZE_HALF = int(TILE_SIZE / 2.0)
    if side == 'top':
        box1 = (0, 0, TILE_SIZE_HALF, size)
        box2 = (TILE_SIZE_HALF, 0, TILE_SIZE, size)
        sz = (TILE_SIZE_HALF, final_size)
    elif side == 'bottom':
        box1 = (0, TILE_SIZE - size, TILE_SIZE_HALF, TILE_SIZE)
        box2 = (TILE_SIZE_HALF, TILE_SIZE - size, TILE_SIZE, TILE_SIZE)
        sz = (TILE_SIZE_HALF, final_size)
    elif side == 'left':
        box1 = (0, 0, size, TILE_SIZE_HALF)
        box2 = (0, TILE_SIZE_HALF, size, TILE_SIZE)
        sz = (final_size, TILE_SIZE_HALF)
    elif side == 'right':
        box1 = (TILE_SIZE - size, 0, TILE_SIZE, TILE_SIZE_HALF)
        box2 = (TILE_SIZE - size, TILE_SIZE_HALF, TILE_SIZE, TILE_SIZE)
        sz = (final_size, TILE_SIZE_HALF)
    side_image_1 = image.crop(box1)
    side_image_2 = image.crop(box2)
    im1 = Image.new("RGB", sz, "white")
    im2 = Image.new("RGB", sz, "white")
    if side in ['top', 'bottom']:
        for i in range(final_size):
            im1.paste(side_image_1, (0, i))
            im2.paste(side_image_2, (0, i))
    else:
        for i in range(final_size):
            im1.paste(side_image_1, (i, 0))
            im2.paste(side_image_2, (i, 0))
    return im1, im2


def get_two_side_images_deep(image, side, size=7):
    TILE_SIZE_HALF = int(TILE_SIZE / 2.0)
    if side == 'top':
        box1 = (0, 0, TILE_SIZE_HALF, size)
        box2 = (TILE_SIZE_HALF, 0, TILE_SIZE, size)
    elif side == 'bottom':
        box1 = (0, TILE_SIZE - size, TILE_SIZE_HALF, TILE_SIZE)
        box2 = (TILE_SIZE_HALF, TILE_SIZE - size, TILE_SIZE, TILE_SIZE)
    elif side == 'left':
        box1 = (0, 0, size, TILE_SIZE_HALF)
        box2 = (0, TILE_SIZE_HALF, size, TILE_SIZE)
    elif side == 'right':
        box1 = (TILE_SIZE - size, 0, TILE_SIZE, TILE_SIZE_HALF)
        box2 = (TILE_SIZE - size, TILE_SIZE_HALF, TILE_SIZE, TILE_SIZE)
    side_image_1 = image.crop(box1)
    side_image_2 = image.crop(box2)
    return side_image_1, side_image_2


def build_similarity_matrix():
    dest_dir = sys.argv[2]
    matrix = {}
    for map_name in os.listdir(dest_dir):
        map_dir = os.path.join(dest_dir, map_name)
        if not os.path.isdir(map_dir):
            continue
        if map_name not in matrix:
            matrix[map_name] = {}
        similarity_matrix = {}
        biggest_score = None
        most_similar_tiles = None
        lowest_score = None
        most_different_tiles = None
        for tile_file_name in os.listdir(map_dir):
            if not tile_file_name.endswith('.png'):
                continue
            tile_name = tile_file_name.replace('.png','')
            tile_file_path = os.path.join(map_dir, tile_file_name)
            for another_tile_file_name in os.listdir(map_dir):
                if not another_tile_file_name.endswith('.png'):
                    continue
                if another_tile_file_name == tile_file_name:
                    continue
                another_tile_name = another_tile_file_name.replace('.png','')
                k1 = f'{tile_name}.{another_tile_name}'
                k2 = f'{tile_name}.{another_tile_name}'
                if k1 in similarity_matrix:
                    continue
                if k2 in similarity_matrix:
                    continue
                another_tile_file_path = os.path.join(map_dir, another_tile_file_name)
                score = images_diff_score(tile_file_path, another_tile_file_path)
                similarity_matrix[k1] = score
                similarity_matrix[k2] = score
                if biggest_score is None or score > biggest_score:
                    biggest_score = score
                    most_similar_tiles = (tile_name, another_tile_name)
                if lowest_score is None or score < lowest_score:
                    lowest_score = score
                    most_different_tiles = (tile_name, another_tile_name)
        matrix[map_name] = similarity_matrix
        print(f'For map {map_name}, most similar tiles: {most_similar_tiles[0]} and {most_similar_tiles[1]} with score {biggest_score}')
        print(f'For map {map_name}, most different tiles: {most_different_tiles[0]} and {most_different_tiles[1]} with score {lowest_score}')
    return matrix


def build_hash_diff_matrix(hash_type='ahash'):
    dest_dir = sys.argv[2]
    matrix = {}
    for map_name in os.listdir(dest_dir):
        map_dir = os.path.join(dest_dir, map_name)
        if not os.path.isdir(map_dir):
            continue
        if map_name not in matrix:
            matrix[map_name] = {}
        hash_diff_matrix = {}
        smallest_diff = None
        similar_tiles = None
        biggest_diff = None
        dissimilar_tiles = None
        for tile_file_name in os.listdir(map_dir):
            if not tile_file_name.endswith('.png'):
                continue
            tile_name = tile_file_name.replace('.png','')
            tile_file_path = os.path.join(map_dir, tile_file_name)
            for another_tile_file_name in os.listdir(map_dir):
                if not another_tile_file_name.endswith('.png'):
                    continue
                if another_tile_file_name == tile_file_name:
                    continue
                another_tile_name = another_tile_file_name.replace('.png','')
                k1 = f'{tile_name}.{another_tile_name}'
                k2 = f'{tile_name}.{another_tile_name}'
                if k1 in hash_diff_matrix:
                    continue
                if k2 in hash_diff_matrix:
                    continue
                another_tile_file_path = os.path.join(map_dir, another_tile_file_name)
                diff = images_hash_diff(tile_file_path, another_tile_file_path, hash_type=hash_type)
                hash_diff_matrix[k1] = diff
                hash_diff_matrix[k2] = diff
                if smallest_diff is None or diff < smallest_diff:
                    smallest_diff = diff
                    similar_tiles = (tile_name, another_tile_name)
                if biggest_diff is None or diff > biggest_diff:
                    biggest_diff = diff
                    dissimilar_tiles = (tile_name, another_tile_name)
        matrix[map_name] = hash_diff_matrix
        print(f'For map {map_name} ({hash_type}), most similar tiles: {similar_tiles[0]} and {similar_tiles[1]} with diff {smallest_diff}')
        print(f'For map {map_name} ({hash_type}), most different tiles: {dissimilar_tiles[0]} and {dissimilar_tiles[1]} with diff {biggest_diff}')
    return matrix


def find_similar_images():
    dest_dir = sys.argv[2]
    score_threshold = float(sys.argv[3])
    for map_name in os.listdir(dest_dir):
        map_dir = os.path.join(dest_dir, map_name)
        if not os.path.isdir(map_dir):
            continue
        next_group_id = 0
        groups = []
        graph = {}
        if not os.path.isdir(map_dir):
            continue

        # read existing groups
        if os.path.exists(os.path.join(dest_dir, map_name+'_groups')):
            for group_id in os.listdir(os.path.join(dest_dir, map_name+'_groups')):
                if not os.path.isdir(os.path.join(dest_dir, map_name+'_groups', group_id)):
                    continue
                groups.append(int(group_id))
                for tile_file_name in os.listdir(os.path.join(dest_dir, map_name+'_groups', group_id)):
                    if not tile_file_name.endswith('.png'):
                        continue
                    tile_name = tile_file_name.replace('.png','')
                    graph[tile_name] = int(group_id)

        # first try to fit ungrouped tiles into existing groups
        for tile_file_name in sorted(os.listdir(map_dir)):
            if not tile_file_name.endswith('.png'):
                continue
            tile_name = tile_file_name.replace('.png','')
            if tile_name in graph:
                continue
            tile_file_path = os.path.join(map_dir, tile_file_name)

            # compare with existing groups
            groups_scores = {}
            for group_id in groups:
                group_dir_path = os.path.join(dest_dir, map_name+'_groups', str(group_id))
                max_score_in_group = 0.0
                group_entries_found = 0
                for group_tile_file_name in os.listdir(group_dir_path):
                    if not group_tile_file_name.endswith('.png'):
                        continue
                    group_tile_file_path = os.path.join(group_dir_path, group_tile_file_name)
                    score = images_diff_score(tile_file_path, group_tile_file_path)
                    if score > max_score_in_group:
                        max_score_in_group = score
                        group_entries_found += 1
                if group_entries_found > 0:
                    groups_scores[group_id] = max_score_in_group

            # find best matching group
            best_group_id = None
            max_group_score = 0.0
            for group_id in groups_scores.keys():
                group_score = groups_scores[group_id]
                if group_score > max_group_score:
                    max_group_score = group_score
                    best_group_id = group_id
            if best_group_id and max_group_score >= score_threshold:
                graph[tile_name] = best_group_id
                print(f'{map_name}: {tile_name}.png grouped in existing group {best_group_id} with score {max_group_score}')

        next_group_id = max(groups) if groups else 0
        # then compare ungrouped tiles with each other
        for tile_file_name in sorted(os.listdir(map_dir)):
            if not tile_file_name.endswith('.png'):
                continue
            tile_name = tile_file_name.replace('.png','')
            if tile_name in graph:
                continue
            tile_file_path = os.path.join(map_dir, tile_file_name)
            max_score = 0.0
            similar_tile_file_name = None
            for another_tile_file_name in os.listdir(map_dir):
                if not another_tile_file_name.endswith('.png'):
                    continue
                if another_tile_file_name == tile_file_name:
                    continue
                another_tile_file_path = os.path.join(map_dir, another_tile_file_name)
                score = images_diff_score(tile_file_path, another_tile_file_path)
                if score > max_score:
                    max_score = score
                    similar_tile_file_name = another_tile_file_name
            if similar_tile_file_name:
                similar_tile_name = similar_tile_file_name.replace('.png', '')
                if max_score >= score_threshold:
                    if tile_name not in graph:
                        if similar_tile_name not in graph:
                            next_group_id += 1
                            groups.append(next_group_id)
                            graph[tile_name] = next_group_id
                            graph[similar_tile_name] = next_group_id
                        else:
                            similar_tile_group = graph[similar_tile_name]
                            graph[tile_name] = similar_tile_group
                    else:
                        if similar_tile_name not in graph:
                            tile_group = graph[tile_name]
                            graph[similar_tile_name] = tile_group
                    print(f'{map_name}: {tile_name}.png ~ {similar_tile_name}.png : {max_score}, grouped in {graph[tile_name]}')

        # move files into group folders
        counter = 1
        for tile_name in graph.keys():
            group_id = graph[tile_name]
            group_dir_path = os.path.join(dest_dir, map_name+'_groups', str(group_id))
            if not os.path.exists(group_dir_path):
                os.makedirs(group_dir_path)
            if os.path.isfile(os.path.join(dest_dir, map_name, f'{tile_name}.png')):
                if not os.path.exists(os.path.join(group_dir_path, f'{tile_name}.png')):
                    os.rename(os.path.join(dest_dir, map_name, f'{tile_name}.png'), os.path.join(group_dir_path, f'{tile_name}.png'))
                    counter += 1
                    print(f'For {map_name} moved {tile_name}.png to group {group_id}')


def read_tile_types(corner_size=None):
    global tile_types
    global tile_average_colors
    global tile_types_images
    global tile_types_corners
    global tile_types_corners_average_color
    global tile_types_corners_hash
    global tile_types_variants
    if corner_size is None:
        corner_size = CORNER_SIZE
    for tile_file_name in os.listdir('samples'):
        if not tile_file_name.endswith('.png'):
            continue
        tile_type = tile_file_name.replace('.png','')
        tile_variant = tile_type[-1]
        tile_type = tile_type[:-1]
        tile_file_path = os.path.join('samples', tile_file_name)
        tile_types[tile_type] = tile_file_path
        if tile_type not in tile_types_variants:
            tile_types_variants[tile_type] = []
        tile_types_variants[tile_type].append(tile_variant)
        if tile_type not in tile_average_colors:
            tile_average_colors[tile_type] = {}
        tile_average_colors[tile_type][tile_variant] = average_color(tile_file_path)
        # print((tile_type, tile_average_colors[tile_type][tile_variant]))
        if tile_type not in tile_types_images:
            tile_types_images[tile_type] = {}
        tile_types_images[tile_type][tile_variant] = {}
        if tile_type not in tile_types_corners:
            tile_types_corners[tile_type] = {}
        tile_types_corners[tile_type][tile_variant] = {}
        if tile_type not in tile_types_corners_average_color:
            tile_types_corners_average_color[tile_type] = {}
        tile_types_corners_average_color[tile_type][tile_variant] = {}
        if tile_type not in tile_types_corners_hash:
            tile_types_corners_hash[tile_type] = {}
        tile_types_corners_hash[tile_type][tile_variant] = {}
        for rotation in [0, 90, 180, 270, -1, -2]:
            if rotation not in tile_types_images[tile_type][tile_variant]:
                tile_types_images[tile_type][tile_variant][rotation] = {}
                tile_types_corners[tile_type][tile_variant][rotation] = {}
                tile_types_corners_average_color[tile_type][tile_variant][rotation] = {}
                tile_types_corners_hash[tile_type][tile_variant][rotation] = {}
            tile_image = Image.open(tile_file_path)
            tile_rotated_image = get_rotated_image(tile_image, rotation)
            tile_types_images[tile_type][tile_variant][rotation] = tile_rotated_image
            for corner in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
                corner_image = get_corner_image(tile_rotated_image, corner, size=corner_size)
                tile_types_corners_average_color[tile_type][tile_variant][rotation][corner] = average_color_in_memory(corner_image)
                tile_types_corners_hash[tile_type][tile_variant][rotation][corner] = image_hash(corner_image)
                tile_types_corners[tile_type][tile_variant][rotation][corner] = corner_image
            for side in ['top', 'bottom', 'left', 'right']:
                # side_image = get_side_image_deep(tile_rotated_image, side)
                side_image = get_side_image_simple(tile_rotated_image, side)
                if tile_type not in tile_types_sides:
                    tile_types_sides[tile_type] = {}
                if tile_variant not in tile_types_sides[tile_type]:
                    tile_types_sides[tile_type][tile_variant] = {}
                if rotation not in tile_types_sides[tile_type][tile_variant]:
                    tile_types_sides[tile_type][tile_variant][rotation] = {}
                tile_types_sides[tile_type][tile_variant][rotation][side] = side_image


def move_tiles_by_type_1(tiles_dir, groups_dir, min_score, remove_original=False):
    global tile_types
    global tile_average_colors
    global tile_types_images
    global tile_types_corners
    global tile_types_corners_average_color
    global tile_types_corners_hash
    global tile_types_variants
    diff_color_factor = 200.0
    corner_size = CORNER_SIZE
    detected_tile_types = set()
    for tile_file_name in sorted(os.listdir(tiles_dir)):
        if not tile_file_name.endswith('.png'):
            continue
        tile_id = tile_file_name.replace('.png','')
        tile_file_path = os.path.join(tiles_dir, tile_file_name)
        tile_image = Image.open(tile_file_path)
        tile_average_color = average_color(tile_file_path)
        tile_corners_best = {}
        for tile_corner in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
            tile_corner_image = get_corner_image(tile_image, tile_corner, size=corner_size)
            best_score = None
            best_sample = None
            best_sample_corner = None
            best_sample_rotation = None
            for tile_rotation in [0, 90, 180, 270, -2, -1]:
                tile_corner_image_rotated = get_rotated_image(tile_corner_image, tile_rotation)
                for sample in tile_types.keys():
                    for tile_variant in tile_types_variants[sample]:
                        for sample_corner in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
                            for sample_rotation in [0, 90, 180, 270, -2, -1]:
                                sample_corner_image = tile_types_corners[sample][tile_variant][sample_rotation][sample_corner]
                                diff_color = color_distance(tile_types_corners_average_color[sample][tile_variant][sample_rotation][sample_corner], tile_average_color)
                                # diff_score = images_diff_score_in_memory(tile_corner_image_rotated, sample_corner_image)
                                score = images_diff_score_in_memory(tile_corner_image_rotated, sample_corner_image)
                                diff_score = round(score * (1.0 - (diff_color / diff_color_factor)), 4)
                                if best_score is None or diff_score > best_score:
                                    best_score = diff_score
                                    best_sample = sample
                                    best_sample_corner = sample_corner
                                    best_sample_rotation = sample_rotation
            tile_corners_best[tile_corner] = (best_sample, best_score, best_sample_corner, best_sample_rotation)
        tl = tile_corners_best['top_left'][0]
        tr = tile_corners_best['top_right'][0]
        bl = tile_corners_best['bottom_left'][0]
        br = tile_corners_best['bottom_right'][0]
        tl_score = round(tile_corners_best['top_left'][1], 3)
        tr_score = round(tile_corners_best['top_right'][1], 3)
        bl_score = round(tile_corners_best['bottom_left'][1], 3)
        br_score = round(tile_corners_best['bottom_right'][1], 3)
        if tl_score >= min_score and tr_score >= min_score and bl_score >= min_score and br_score >= min_score:
            if tl == tr == bl == br:
                detected_tile_types.add(tl)
                tile_type_dir_path = os.path.join(groups_dir, tl)
                if not os.path.exists(tile_type_dir_path):
                    os.makedirs(tile_type_dir_path)
                tile_dest_file_path = os.path.join(tile_type_dir_path, f'{tile_id}.png')
                if remove_original:
                    os.rename(tile_file_path, tile_dest_file_path)
                else:
                    tile_image.save(tile_dest_file_path)
                print(f'{tile_id}    {tl}    tl:{tl}({tl_score}) tr:{tr}({tr_score}) br:{br}({br_score}) bl:{bl}({bl_score})            IS {tl}')
                continue
        print(f'        {tile_id}    tl:{tl}({tl_score}) tr:{tr}({tr_score}) br:{br}({br_score}) bl:{bl}({bl_score}) ')
    return detected_tile_types


def move_tiles_by_type(tiles_dir, groups_dir, min_score, remove_original=False):
    diff_color_factor = 300.0
    detected_tile_types = set()
    for tile_file_name in sorted(os.listdir(tiles_dir)):
        if not tile_file_name.endswith('.png'):
            continue
        tile_name = tile_file_name.replace('.png','')
        tile_file_path = os.path.join(tiles_dir, tile_file_name)
        tile_average_color = average_color(tile_file_path)
        tile_image = Image.open(tile_file_path)
        best_type = None
        best_score = None
        best_color_dist = None
        best_diff_score = None
        best_rotation = None
        for tile_type in tile_types.keys():
            for rotation in [0, 90, 180, 270, -1, -2]:
                for tile_variant in tile_types_variants[tile_type]:
                    tile_type_image = tile_types_images[tile_type][tile_variant][rotation]
                    diff_score = images_diff_score_in_memory(tile_image, tile_type_image)
                    diff_color = color_distance(tile_average_colors[tile_type][tile_variant], tile_average_color)
                    score = round(diff_score * (1.0 - (diff_color / diff_color_factor)), 3)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_type = tile_type
                        best_color_dist = diff_color
                        best_rotation = rotation
                        best_diff_score = diff_score
        if best_diff_score:
            best_diff_score = round(best_diff_score, 3)
        if best_score > min_score:
            tile_type_dir_path = os.path.join(groups_dir, best_type)
            if not os.path.exists(tile_type_dir_path):
                os.makedirs(tile_type_dir_path)
            tile_dest_file_path = os.path.join(tile_type_dir_path, f'{tile_name}.png')
            if remove_original:
                os.rename(tile_file_path, tile_dest_file_path)
            else:
                tile_image.save(tile_dest_file_path)
            detected_tile_types.add(best_type)
            print(f'{tile_name}    {best_type} {best_score} {best_diff_score} {best_color_dist} {best_rotation}           IS {best_type}')
        else:
            print(f'        {tile_name}    {best_type} {best_score} {best_diff_score} {best_color_dist}')
    return detected_tile_types


def move_tiles_by_corner_type(tiles_dir, groups_dir, selected_tile_types=None):
    corner_names = ['tl', 'tr', 'br', 'bl']
    color_distance_threshold = int((CORNER_SIZE * CORNER_SIZE) * 2.0)
    diff_hash_threshold = CORNER_SIZE * CORNER_SIZE * 2
    color_distance_factor = 400
    for tile_file_name in sorted(os.listdir(tiles_dir)):
        if not tile_file_name.endswith('.png'):
            continue
        tile_name = tile_file_name.replace('.png','')
        # if tile_name != '00050':
        #     continue
        tile_file_path = os.path.join(tiles_dir, tile_file_name)
        c = {}
        best_rotation = None
        best_rotation_score = None
        for rotation in [0, 90, 180, 270]:
            if rotation not in c:
                c[rotation] = {}
            for corner in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
                corner_x = 0 if corner in ['top_left', 'bottom_left'] else TILE_SIZE - CORNER_SIZE
                corner_y = 0 if corner in ['top_left', 'top_right'] else TILE_SIZE - CORNER_SIZE
                corner_crop = (corner_x, corner_y, corner_x + CORNER_SIZE, corner_y + CORNER_SIZE)
                corner_image_orig = Image.open(tile_file_path).crop(corner_crop)
                corner_image = get_rotated_image(corner_image_orig, rotation)
                corner_average_color = average_color_in_memory(corner_image)
                corner_hash = image_hash(corner_image)
                corner_best_score = None
                corner_best_type = None
                corner_best_color_dist = None
                corner_best_diff_score = None
                corner_best_hash_diff = None
                for tile_type in tile_types.keys():
                    if selected_tile_types and tile_type not in selected_tile_types:
                        continue
                    for tile_variant in tile_types_variants[tile_type]:
                        type_corner_image = tile_types_corners[tile_type][tile_variant][rotation][corner]
                        type_corner_average_color = tile_types_corners_average_color[tile_type][tile_variant][rotation][corner]
                        type_corner_hash = tile_types_corners_hash[tile_type][tile_variant][rotation][corner]
                        diff_color = color_distance(corner_average_color, type_corner_average_color)
                        diff_hash = corner_hash - type_corner_hash
                        diff_score = images_diff_score_in_memory(corner_image, type_corner_image)
                        score = diff_score
                        # if diff_hash > diff_hash_threshold:
                        #     continue
                        # if diff_color > color_distance_threshold:
                        #     continue
                        score = round(diff_score * (1.0 - (diff_color / color_distance_factor)), 3)
                        # if tile_name == '00050':
                        #     print(f'    {tile_type} {corner} {rotation}  diff_score={diff_score} diff_color={diff_color} diff_hash={diff_hash} score={score}')
                        if diff_score < 0:
                            continue
                        if corner_best_score is None or score > corner_best_score:
                            corner_best_score = score
                            corner_best_type = tile_type
                            corner_best_diff_score = diff_score
                            corner_best_color_dist = diff_color
                            corner_best_hash_diff = diff_hash
                if corner_best_type is None:
                    corner_best_type = 'unknown'
                    corner_best_score = 0.0
                    corner_best_color_dist = 999
                    corner_best_diff_score = 0.0
                    corner_best_hash_diff = 999
                c[rotation][corner] = (corner_best_type, corner_best_score, round(corner_best_diff_score, 3), corner_best_color_dist, corner_best_hash_diff, rotation)
            rotation_score = 1.0
            for corner in c[rotation].keys():
                rotation_score = min(rotation_score, c[rotation][corner][1])
            # print(f'{tile_name} rotation {rotation} score: {rotation_score}')
            if best_rotation_score is None or rotation_score > best_rotation_score:
                best_rotation_score = rotation_score
                best_rotation = rotation
        corn = c[best_rotation]
        tl_type = corn['top_left'][0]
        tr_type = corn['top_right'][0]
        bl_type = corn['bottom_left'][0]
        br_type = corn['bottom_right'][0]
        tl_score = round(corn['top_left'][1], 3)
        tr_score = round(corn['top_right'][1], 3)
        bl_score = round(corn['bottom_left'][1], 3)
        br_score = round(corn['bottom_right'][1], 3)
        tl_diff_score = corn['top_left'][2]
        tr_diff_score = corn['top_right'][2]
        bl_diff_score = corn['bottom_left'][2]
        br_diff_score = corn['bottom_right'][2]
        tl_color_dist = corn['top_left'][3]
        tr_color_dist = corn['top_right'][3]
        bl_color_dist = corn['bottom_left'][3]
        br_color_dist = corn['bottom_right'][3]
        tl_hash_diff = corn['top_left'][4]
        tr_hash_diff = corn['top_right'][4]
        bl_hash_diff = corn['bottom_left'][4]
        br_hash_diff = corn['bottom_right'][4]
        tl_rotation = corn['top_left'][5]
        tr_rotation = corn['top_right'][5]
        bl_rotation = corn['bottom_left'][5]
        br_rotation = corn['bottom_right'][5]
        if 'unknown' in [tl_type, tr_type, br_type, bl_type]:
            print(f'{tile_name}    {tl_type} {tl_score} {tl_diff_score} {tl_color_dist} {tl_hash_diff} {tl_rotation}   {tr_type} {tr_score} {tr_diff_score} {tr_color_dist} {tr_hash_diff} {tr_rotation}   {br_type} {br_score} {br_diff_score} {br_color_dist} {br_hash_diff} {bl_rotation}   {bl_type} {bl_score} {bl_diff_score} {bl_color_dist} {bl_hash_diff} {br_rotation}    is unknown')
            continue
        if tl_color_dist > color_distance_threshold or tr_color_dist > color_distance_threshold or bl_color_dist > color_distance_threshold or br_color_dist > color_distance_threshold:
            print(f'{tile_name}    {tl_type} {tl_score} {tl_diff_score} {tl_color_dist} {tl_hash_diff} {tl_rotation}   {tr_type} {tr_score} {tr_diff_score} {tr_color_dist} {tr_hash_diff} {tr_rotation}   {br_type} {br_score} {br_diff_score} {br_color_dist} {br_hash_diff} {bl_rotation}   {bl_type} {bl_score} {bl_diff_score} {bl_color_dist} {bl_hash_diff} {br_rotation}    color dist too high')
            continue
        if tl_hash_diff > diff_hash_threshold or tr_hash_diff > diff_hash_threshold or bl_hash_diff > diff_hash_threshold or br_hash_diff > diff_hash_threshold:
            print(f'{tile_name}    {tl_type} {tl_score} {tl_diff_score} {tl_color_dist} {tl_hash_diff} {tl_rotation}   {tr_type} {tr_score} {tr_diff_score} {tr_color_dist} {tr_hash_diff} {tr_rotation}   {br_type} {br_score} {br_diff_score} {br_color_dist} {br_hash_diff} {bl_rotation}   {bl_type} {bl_score} {bl_diff_score} {bl_color_dist} {bl_hash_diff} {br_rotation}    hash dist too high')
            continue
        corners = [tl_type, tr_type, br_type, bl_type]
        types_set = set(corners)
        types_list = list(types_set)
        types_counts = {}
        for t in types_list:
            types_counts[t] = corners.count(t)
        types_list.sort(key=lambda x: types_counts[x], reverse=True)
        # if len(types_set) == 1:
        #     best_type = tl_type
        #     tile_type_dir_path = os.path.join(groups_dir, best_type)
        #     if not os.path.exists(tile_type_dir_path):
        #         os.makedirs(tile_type_dir_path)
        #     tile_dest_file_path = os.path.join(tile_type_dir_path, f'{best_type}_{tile_name}.png')
        #     os.rename(tile_file_path, tile_dest_file_path)
        #     print(f'{tile_name}    {tl_type} {tl_score} {tl_diff_score} {tl_color_dist} {tl_hash_diff} {tl_rotation}   {tr_type} {tr_score} {tr_diff_score} {tr_color_dist} {tr_hash_diff} {tr_rotation}   {br_type} {br_score} {br_diff_score} {br_color_dist} {br_hash_diff} {bl_rotation}   {bl_type} {bl_score} {bl_diff_score} {bl_color_dist} {bl_hash_diff} {br_rotation}    IS {best_type}')
        #     continue
        if len(types_set) == 2:
            best_type = types_list[0]
            second_type = types_list[1]
            best_type_count = types_counts[best_type]
            second_type_count = types_counts[second_type]
            if best_type_count == 3 and second_type_count == 1:
                direction = corner_names[corners.index(second_type)]
                tile_type_dir_path = os.path.join(groups_dir, best_type)
                if not os.path.exists(tile_type_dir_path):
                    os.makedirs(tile_type_dir_path)
                tile_dest_file_path = os.path.join(tile_type_dir_path, f'{best_type}_{second_type}_{direction}_{tile_name}.png')
                os.rename(tile_file_path, tile_dest_file_path)
                print(f'{tile_name}    {tl_type} {tl_score} {tl_diff_score} {tl_color_dist} {tl_hash_diff} {tl_rotation}   {tr_type} {tr_score} {tr_diff_score} {tr_color_dist} {tr_hash_diff} {tr_rotation}   {br_type} {br_score} {br_diff_score} {br_color_dist} {br_hash_diff} {bl_rotation}   {bl_type} {bl_score} {bl_diff_score} {bl_color_dist} {bl_hash_diff} {br_rotation}    IS {best_type}_{second_type}')
            else:
                # best_type_corner_name = corner_names[corners.index(best_type)]
                # second_type_corner_name = corner_names[corners.index(second_type)]
                direction = ''
                if corners[0] == corners[1]:
                    direction = 'tb' if corners[0] == best_type else 'bt'
                else:
                    if corners[1] == corners[2]:
                        direction = 'lr' if corners[0] == best_type else 'rl'
                    else:
                        direction = 'trbl' if corners[0] == best_type else 'tlbr'
                tile_type_dir_path = os.path.join(groups_dir, best_type)
                if not os.path.exists(tile_type_dir_path):
                    os.makedirs(tile_type_dir_path)
                tile_dest_file_path = os.path.join(tile_type_dir_path, f'{best_type}_{second_type}_{direction}_{tile_name}.png')
                os.rename(tile_file_path, tile_dest_file_path)
                print(f'{tile_name}    {tl_type} {tl_score} {tl_diff_score} {tl_color_dist} {tl_hash_diff} {tl_rotation}   {tr_type} {tr_score} {tr_diff_score} {tr_color_dist} {tr_hash_diff} {tr_rotation}   {br_type} {br_score} {br_diff_score} {br_color_dist} {br_hash_diff} {bl_rotation}   {bl_type} {bl_score} {bl_diff_score} {bl_color_dist} {bl_hash_diff} {br_rotation}    IS {best_type}_{second_type}')
            continue
        print(f'{tile_name}    {tl_type} {tl_score} {tl_diff_score} {tl_color_dist} {tl_hash_diff} {tl_rotation}   {tr_type} {tr_score} {tr_diff_score} {tr_color_dist} {tr_hash_diff} {tr_rotation}   {br_type} {br_score} {br_diff_score} {br_color_dist} {br_hash_diff} {bl_rotation}   {bl_type} {bl_score} {bl_diff_score} {bl_color_dist} {bl_hash_diff} {br_rotation}    ??? {",".join(list(types_set))}')


def move_tiles_by_one_corner_similarity(tiles_dir, groups_dir, min_score, selected_corner, selected_types, remove_original=False):
    for tile_file_name in sorted(os.listdir(tiles_dir)):
        if not tile_file_name.endswith('.png'):
            continue
        tile_name = tile_file_name.replace('.png','')
        tile_file_path = os.path.join(tiles_dir, tile_file_name)
        tile_image = Image.open(tile_file_path)
        tile_corner_image = get_corner_image(tile_image, selected_corner)
        best_sample = None
        best_sample_rotation = None
        best_sample_corner = None
        best_sample_corner_score = None        
        for sample in tile_types_sides.keys():
            if sample not in selected_types:
                continue            
            for sample_rotation in [0, 90, 180, 270, -1, -2]:
                for sample_corner in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
                    sample_corner_image = tile_types_corners[sample][sample_rotation][sample_corner]
                    diff_score = images_diff_score_in_memory(tile_corner_image, sample_corner_image, round_digits=5)
                    if best_sample_corner_score is None or diff_score > best_sample_corner_score:
                        best_sample_corner_score = diff_score
                        best_sample = sample
                        best_sample_rotation = sample_rotation
                        best_sample_corner = sample_corner
        if best_sample_corner_score is None:
            continue
        if best_sample_corner_score < min_score:
            continue
        tile_dest_dir_path = os.path.join(groups_dir, best_sample)
        if not os.path.exists(tile_dest_dir_path):
            os.makedirs(tile_dest_dir_path)
        tile_dest_file_path = os.path.join(tile_dest_dir_path, f'{best_sample}_{selected_corner.replace("_","")}_{tile_name}.png')
        if remove_original:
            os.rename(tile_file_path, tile_dest_file_path)
        else:
            tile_image.save(tile_dest_file_path)
        print(f'{tile_name}    {best_sample} {best_sample_corner_score} {best_sample_corner} {best_sample_rotation}')


def move_tiles_by_one_side_similarity(tiles_dir, groups_dir, min_score, selected_side, selected_types, remove_original=False):
    for tile_file_name in sorted(os.listdir(tiles_dir)):
        if not tile_file_name.endswith('.png'):
            continue
        tile_name = tile_file_name.replace('.png','')
        tile_file_path = os.path.join(tiles_dir, tile_file_name)
        tile_image = Image.open(tile_file_path)
        best_tile_side_score = None
        # best_tile_rotation = None
        best_tile_side = None
        best_tile_image = None
        # best_tile_image_rotated = None
        best_sample = None
        best_sample_side = None
        # best_sample_rotation = None
        # best_sample_image = None
        best_sides = {}
        for sample in tile_types_sides.keys():
            if sample not in selected_types:
                continue
            for sample_side in ['top', 'bottom', 'left', 'right']:
                tile_side = selected_side
                if sample_side in ['top', 'bottom'] and tile_side in ['left', 'right']:
                    continue
                if sample_side in ['left', 'right'] and tile_side in ['top', 'bottom']:
                    continue
                if sample_side not in best_sides:
                    best_sides[sample_side] = (0, None, None, None)
                # for tile_rotation in [0, 90, 180, 270]:
                # tile_rotated_image = get_rotated_image(tile_image, tile_rotation)
                side_image = get_side_image_simple(tile_image, tile_side)
                # sample_rotation = 0
                sample_side_image = tile_types_sides[sample][0][sample_side]
                diff_score = images_diff_score_in_memory(side_image, sample_side_image)
                if best_tile_side_score is None or diff_score > best_tile_side_score:
                    best_tile_side_score = diff_score
                    # best_tile_rotation = tile_rotation
                    best_tile_side = tile_side
                    best_tile_image = tile_image
                    # best_tile_image_rotated = tile_rotated_image
                    best_sample = sample
                    best_sample_side = sample_side
                    # best_sample_rotation = sample_rotation
                    # best_sample_image = tile_types_images[best_sample][0]
                if diff_score > best_sides[sample_side][0]:
                    best_sides[sample_side] = (round(diff_score, 2), sample, sample_side, tile_side)
        if best_tile_side_score is None:
            # print(f'{tile_name}         no best side found for the tile')
            continue
        best_score = best_tile_side_score
        best_score = round(best_score, 2)
        if best_score < min_score:
            continue
        t = best_sides['top'] if 'top' in best_sides else (0, None, None, None)
        b = best_sides['bottom'] if 'bottom' in best_sides else (0, None, None, None)
        l = best_sides['left'] if 'left' in best_sides else (0, None, None, None)
        r = best_sides['right'] if 'right' in best_sides else (0, None, None, None)
        # im = Image.new("RGB", (192, 192), "black")
        # im.paste(best_sample_image, (64, 64))
        side_sample = best_sample_side
        side_tile = best_tile_side
        if side_sample == 'top' and side_tile == 'top':
            tile_rotation = -2
            # pos = (64, 0)
        elif side_sample == 'top' and side_tile == 'bottom':
            tile_rotation = 0
            # pos = (64, 0)
        elif side_sample == 'bottom' and side_tile == 'top':
            tile_rotation = 0
            # pos = (64, 128)
        elif side_sample == 'bottom' and side_tile == 'bottom':
            tile_rotation = -2
            # pos = (64, 128)
        elif side_sample == 'left' and side_tile == 'left':
            tile_rotation = -1
            # pos = (0, 64)
        elif side_sample == 'left' and side_tile == 'right':
            tile_rotation = 0
            # pos = (0, 64)
        elif side_sample == 'right' and side_tile == 'left':
            tile_rotation = 0
            # pos = (128, 64)
        elif side_sample == 'right' and side_tile == 'right':
            tile_rotation = -1
            # pos = (128, 64)
        tile_im = get_rotated_image(best_tile_image, tile_rotation)
        # im.paste(tile_im, pos)
        tile_dest_dir_path = os.path.join(groups_dir, best_sample)
        if not os.path.exists(tile_dest_dir_path):
            os.makedirs(tile_dest_dir_path)
        tile_dest_file_path = os.path.join(tile_dest_dir_path, f'{best_sample}_{side_sample}_{tile_name}.png')
        # im.save(tile_dest_file_path)
        if remove_original:
            os.rename(tile_file_path, tile_dest_file_path)
        else:
            tile_im.save(tile_dest_file_path)
        print(f'{tile_name}    {best_score}   sample:({best_sample} {best_sample_side})   tile:({best_tile_side})   top:({t[0]} {t[1]} {t[2]}/{t[3]})   bottom:({b[0]} {b[1]} {b[2]}/{b[3]})   left:({l[0]} {l[1]} {l[2]}/{l[3]})   right:({r[0]} {r[1]} {r[2]}/{r[3]})')


def move_tiles_by_two_sides_similarity(tiles_dir, groups_dir, min_score, selected_types):
    for tile_file_name in sorted(os.listdir(tiles_dir)):
        if not tile_file_name.endswith('.png'):
            continue
        tile_name = tile_file_name.replace('.png','')
        tile_file_path = os.path.join(tiles_dir, tile_file_name)
        tile_image = Image.open(tile_file_path)
        tile_sides = {}
        tile_sides_best = {}
        for tile_side in ['top', 'bottom', 'left', 'right']:
            best_tile_side_score = None
            best_sample = None
            best_sample_side = None
            best_sample_rotation = None
            best_tile_rotation = None
            best_tile_side = None
            for tile_rotation in [0, 90, 180, 270]:
                tile_rotated_image = get_rotated_image(tile_image, tile_rotation)
                side_image = get_side_image_deep(tile_rotated_image, tile_side)
                for sample in tile_types_sides.keys():
                    if sample not in selected_types:
                        continue
                    for sample_rotation in [0, 90, 180, 270]:
                        for sample_side in ['top', 'bottom', 'left', 'right']:
                            if sample_side in ['top', 'bottom'] and tile_side in ['left', 'right']:
                                continue
                            if sample_side in ['left', 'right'] and tile_side in ['top', 'bottom']:
                                continue
                            sample_side_image = tile_types_sides[sample][sample_rotation][sample_side]
                            diff_score = images_diff_score_in_memory(side_image, sample_side_image)
                            if best_tile_side_score is None or diff_score > best_tile_side_score:
                                best_tile_side_score = diff_score
                                best_sample = sample
                                best_sample_side = sample_side
                                best_sample_rotation = sample_rotation
                                best_tile_rotation = tile_rotation
                                best_tile_side = tile_side
            best_score = best_tile_side_score
            best_score = round(best_score, 3)
            if best_score > min_score:
                tile_sides[tile_side] = (best_sample, best_sample_side, best_sample_rotation, best_tile_side, best_tile_rotation)
            else:
                tile_sides[tile_side] = (None, None, None, None, None)
            tile_sides_best[tile_side] = (best_sample, best_score)
        sides = [tile_sides['left'][0], tile_sides['top'][0], tile_sides['right'][0], tile_sides['bottom'][0]]
        sides_set = set(filter(None, sides))
        if len(sides_set) == 0:
            print(f'    no sides matched for {tile_name}, best matches: left({tile_sides_best["left"][0]} {tile_sides_best["left"][1]}), top({tile_sides_best["top"][0]} {tile_sides_best["top"][1]}), right({tile_sides_best["right"][0]} {tile_sides_best["right"][1]}), bottom({tile_sides_best["bottom"][0]} {tile_sides_best["bottom"][1]})')
            continue
        # if len(sides_set) == 1:
        #     side = None
        #     for s in ['left', 'top', 'right', 'bottom']:
        #         if tile_sides[s][0] is not None:
        #             side = s
        #             break
        #     best_sample, best_sample_side, best_sample_rotation, best_tile_side, best_tile_rotation = tile_sides[side]
        #     tile_dest_dir_path = os.path.join(groups_dir, best_sample)
        #     if not os.path.exists(tile_dest_dir_path):
        #         os.makedirs(tile_dest_dir_path)
        #     tile_dest_file_path = os.path.join(tile_dest_dir_path, f'{best_sample}_{best_sample_side}_{best_sample_rotation}_{best_tile_side}_{best_tile_rotation}_{tile_name}.png')
        #     os.rename(tile_file_path, tile_dest_file_path)
        #     print(f'{tile_name}    {best_score}   sample:({best_sample} {best_sample_side} {best_sample_rotation})   tile:({best_tile_side} {best_tile_rotation})')
        #     continue
        if len(sides_set) == 2:
            side1 = None
            side2 = None
            for s in ['left', 'top', 'right', 'bottom']:
                if tile_sides[s][0] is not None:
                    if side1 is None:
                        side1 = s
                    else:
                        side2 = s
                        break
            best_sample1, best_sample_side1, best_sample_rotation1, best_tile_side1, best_tile_rotation1 = tile_sides[side1]
            best_sample2, best_sample_side2, best_sample_rotation2, best_tile_side2, best_tile_rotation2 = tile_sides[side2]
            tile_dest_dir_path = os.path.join(groups_dir, best_sample1)
            if not os.path.exists(tile_dest_dir_path):
                os.makedirs(tile_dest_dir_path)
            tile_dest_file_path = os.path.join(tile_dest_dir_path, f'{best_sample1}_{best_sample2}_{tile_name}.png')
            os.rename(tile_file_path, tile_dest_file_path)
            print(f'{tile_name}    {best_score}   side1:({side1} {best_sample1} {best_sample_side1} {best_sample_rotation1})    side2:({side2} {best_sample2} {best_sample_side2} {best_sample_rotation2})   tile1:({best_tile_side1} {best_tile_rotation1})   tile2:({best_tile_side2} {best_tile_rotation2})')
            continue


def move_tiles_within_group_by_second_side_similarity(groups_dir, dest_dir, min_score, selected_side, selected_types, remove_original=False):
    for group_name in sorted(os.listdir(groups_dir)):
        group_dir_path = os.path.join(groups_dir, group_name)
        if not os.path.isdir(group_dir_path):
            continue
        lst = sorted(os.listdir(group_dir_path))
        for tile_file_name in lst:
            if not tile_file_name.endswith('.png'):
                continue
            tile_name = tile_file_name.replace('.png','')
            tile_file_path = os.path.join(group_dir_path, tile_file_name)
            tile_image = Image.open(tile_file_path)
            parts = tile_name.split('_')
            if len(parts) != 3:
                continue
            tile_sample, tile_side, tile_id = parts
            if selected_side != tile_side:
                continue
            # if tile_side not in ['top', 'bottom', 'left', 'right']:
            #     continue
            # if tile_side 
            tile_other_side = {'top':'bottom', 'bottom':'top', 'left':'right', 'right':'left'}[tile_side]
            # tile_other_side = tile_side
            tile_other_side_image = get_side_image_simple(tile_image, tile_other_side)
            # tile_other_side_image = get_side_image_deep(tile_image, tile_other_side)
            # tile_other_side_average_color = average_color_in_memory(tile_other_side_image)
            best_tile_side_score = None
            # best_sample_side = None
            best_sample = None
            best_sample_rotation = None
            for sample in tile_types_sides.keys():
                if sample == tile_sample:
                    continue
                if sample not in selected_types:
                    continue
                # for sample_side in ['top', 'bottom', 'left', 'right']:
                #     if sample_side in ['top', 'bottom'] and tile_other_side in ['left', 'right']:
                #         continue
                #     if sample_side in ['left', 'right'] and tile_other_side in ['top', 'bottom']:
                #         continue
                for rotation in [0, 90, 180, 270, -1, -2]:
                    sample_side_image = tile_types_sides[sample][rotation][tile_other_side]
                    # sample_side_average_color = average_color_in_memory(sample_side_image)
                    diff_score = images_diff_score_in_memory(tile_other_side_image, sample_side_image)
                    # diff_color = color_distance(tile_other_side_average_color, sample_side_average_color)
                    # score = round(diff_score * (1.0 - (diff_color / color_distance_factor)), 3)
                    score = round(diff_score, 3)
                    # print(f'{group_name}/{tile_name}/{tile_other_side}    {score} {sample} {sample_side} {rotation}')
                    if best_tile_side_score is None or score > best_tile_side_score:
                        best_tile_side_score = score
                        best_sample = sample
                        # best_sample_side = sample_side
                        best_sample_rotation = rotation
            best_score = best_tile_side_score
            best_score = round(best_score, 3)
            # if best_score < min_score:
                # print(f'    no sides matched for {tile_name} in group {group_name}, best match: {best_score} {best_sample_side}   tile: {best_tile_side} {best_tile_rotation}')
            #     continue
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest_file_path = os.path.join(dest_dir, f'{tile_id}_{tile_sample}_{tile_side}_{best_sample}.png')
            if remove_original:
                os.rename(tile_file_path, dest_file_path)
            else:
                tile_image.save(dest_file_path)
            print(f'{group_name}/{tile_id}    {best_score}    {tile_sample} {best_sample} {best_sample_rotation}')


def move_tiles_within_group_by_second_corner_similarity(groups_dir, dest_dir, min_score, selected_corner, selected_types, remove_original=False):
    for group_name in sorted(os.listdir(groups_dir)):
        group_dir_path = os.path.join(groups_dir, group_name)
        if not os.path.isdir(group_dir_path):
            continue
        lst = sorted(os.listdir(group_dir_path))
        for tile_file_name in lst:
            if not tile_file_name.endswith('.png'):
                continue
            tile_name = tile_file_name.replace('.png','')
            tile_file_path = os.path.join(group_dir_path, tile_file_name)
            tile_image = Image.open(tile_file_path)
            parts = tile_name.split('_')
            if len(parts) != 3:
                continue
            tile_sample, tile_corner, tile_id = parts
            tile_corner = {'topleft':'top_left', 'topright':'top_right', 'bottomleft':'bottom_left', 'bottomright':'bottom_right'}.get(tile_corner, tile_corner)
            if selected_corner != tile_corner:
                continue
            tile_other_corner = {'top_left':'bottom_right', 'top_right':'bottom_left', 'bottom_left':'top_right', 'bottom_right':'top_left'}[tile_corner]
            tile_other_corner_image = get_corner_image(tile_image, tile_other_corner)
            best_tile_corner_score = None
            best_sample = None
            best_sample_rotation = None
            for sample in tile_types_corners.keys():
                if sample == tile_sample:
                    continue
                if sample not in selected_types:
                    continue
                for rotation in [0, 90, 180, 270, -1, -2]:
                    sample_corner_image = tile_types_corners[sample][rotation][tile_other_corner]
                    diff_score = images_diff_score_in_memory(tile_other_corner_image, sample_corner_image)
                    score = round(diff_score, 3)
                    if best_tile_corner_score is None or score > best_tile_corner_score:
                        best_tile_corner_score = score
                        best_sample = sample
                        best_sample_rotation = rotation
            best_score = best_tile_corner_score
            best_score = round(best_score, 3)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest_file_path = os.path.join(dest_dir, f'{tile_id}_{tile_sample}_{tile_corner}_{best_sample}.png')
            if remove_original:
                os.rename(tile_file_path, dest_file_path)
            else:
                tile_image.save(dest_file_path)
            print(f'{group_name}/{tile_id}    {best_score}    {tile_sample} {best_sample} {best_sample_rotation}')


# def move_tiles_50_50(tiles_dir, groups_dir, min_score, selected_types):
#     for tile_file_name in sorted(os.listdir(tiles_dir)):
#         if not tile_file_name.endswith('.png'):
#             continue
#         tile_name = tile_file_name.replace('.png','')
#         tile_file_path = os.path.join(tiles_dir, tile_file_name)
#         tile_image = Image.open(tile_file_path)
#         tile_sides = {}
#         tile_sides_best = {}
#         best_tile_side_score = None
#         best_sample = None
#         best_sample_side = None
#         best_sample_rotation = None
#         best_tile_rotation = None
#         best_tile_side = None
#         for tile_side in ['top', 'bottom', 'left', 'right']:
#             for tile_rotation in [0, 90, 180, 270]:
#                 tile_rotated_image = get_rotated_image(tile_image, tile_rotation)
#                 side_image = get_side_image_deep(tile_rotated_image, tile_side)
#                 for sample in tile_types_sides.keys():
#                     if sample not in selected_types:
#                         continue
#                     for sample_rotation in [0, 90, 180, 270]:
#                         for sample_side in ['top', 'bottom', 'left', 'right']:
#                             if sample_side in ['top', 'bottom'] and tile_side in ['left', 'right']:
#                                 continue
#                             if sample_side in ['left', 'right'] and tile_side in ['top', 'bottom']:
#                                 continue
#                             sample_side_image = tile_types_sides[sample][sample_rotation][sample_side]
#                             diff_score = images_diff_score_in_memory(side_image, sample_side_image)
#                             if best_tile_side_score is None or diff_score > best_tile_side_score:
#                                 best_tile_side_score = diff_score
#                                 best_sample = sample
#                                 best_sample_side = sample_side
#                                 best_sample_rotation = sample_rotation
#                                 best_tile_rotation = tile_rotation
#                                 best_tile_side = tile_side
#         best_score = best_tile_side_score
#         best_score = round(best_score, 3)
#         if best_score < min_score:
#             print(f'    no sides matched for {tile_name}, best match: {best_score} {best_sample} {best_sample_side} {best_sample_rotation} tile: {best_tile_side} {best_tile_rotation}')
#             continue
#         best_score = None
#         best_couple = None
#         for side1, side2 in [('top', 'bottom'), ('left', 'right'), ('top', 'left'), ('top', 'right'), ('left', 'bottom'), ('right', 'bottom')]:
#             side1_image = get_side_image_deep(tile_image, side1)
#             side2_image = get_side_image_deep(tile_image, side2)
#             for rotation in [0, 90, 180, 270]:
#                 side2_rotated_image = get_rotated_image(side2_image, rotation)
#                 if side2_rotated_image.width != side1_image.width or side2_rotated_image.height != side1_image.height:
#                     continue
#                 diff_score = images_diff_score_in_memory(side1_image, side2_rotated_image)
#                 if best_score is None or diff_score > best_score:
#                     best_score = diff_score
#                     best_couple = (side1, side2)
#         if best_couple not in [('top', 'bottom'), ('left', 'right')]:
#             print(f'    no matching sides for 50/50 for {tile_name}, best couple: {best_couple} score: {best_score}')
#             continue
#         side1, side2 = best_couple
#         tile_dest_dir_path = os.path.join(groups_dir, best_sample)
#         if not os.path.exists(tile_dest_dir_path):
#             os.makedirs(tile_dest_dir_path)
#         tile_dest_file_path = os.path.join(tile_dest_dir_path, f'{best_sample}_{side1}_{side2}_{tile_name}.png')
#         os.rename(tile_file_path, tile_dest_file_path)
#         print(f'{tile_name}    {best_score}   sample:({best_sample} {side1} {side2})   tile:({best_tile_side} {best_tile_rotation})')


def move_tiles_by_shape(tiles_dir, dest_dir, min_score, remove_original=False):
    corner_size = int(TILE_SIZE / 4)
    lst = list(sorted(os.listdir(tiles_dir)))
    for tile_file_name in lst:
        if not tile_file_name.endswith('.png'):
            continue
        tile_name = tile_file_name.replace('.png','')
        tile_file_path = os.path.join(tiles_dir, tile_file_name)
        tile_image = Image.open(tile_file_path)
        tile_image_transposed = tile_image.transpose(Transpose.TRANSPOSE)
        tile_image_transversed = tile_image.transpose(Transpose.TRANSVERSE)
        transposed_diff_score = images_diff_score_in_memory(tile_image, tile_image_transposed)
        transversed_diff_score = images_diff_score_in_memory(tile_image, tile_image_transversed)
        if transposed_diff_score < 0 or transversed_diff_score < 0:
            continue
        if transposed_diff_score < min_score and transversed_diff_score < min_score:
            continue
        if transposed_diff_score > transversed_diff_score:
            ratio = round(transposed_diff_score / transversed_diff_score, 4)
        else:
            ratio = round(transversed_diff_score / transposed_diff_score, 4)
        if ratio < 1.2:
            print(f'{tile_name}    ratio too small: {ratio} {transposed_diff_score} {transversed_diff_score}')
            continue
        direction = 'top_left_bottom_right' if transposed_diff_score > transversed_diff_score else 'top_right_bottom_left'
        corners = {}
        for corner, rotation in [('top_left', 0), ('top_right', 90), ('bottom_right', 180), ('bottom_left', 270)]:
            rotated_image = get_rotated_image(tile_image, rotation)
            corner_image = get_corner_image(rotated_image, 'top_left', size=corner_size)
            corners[corner] = corner_image
        diff_top = images_diff_score_in_memory(corners['top_left'], corners['top_right'], min_diff=0.00001, round_digits=5)
        diff_bottom = images_diff_score_in_memory(corners['bottom_left'], corners['bottom_right'], min_diff=0.00001, round_digits=5)
        diff_left = images_diff_score_in_memory(corners['top_left'], corners['bottom_left'], min_diff=0.00001, round_digits=5)
        diff_right = images_diff_score_in_memory(corners['top_right'], corners['bottom_right'], min_diff=0.00001, round_digits=5)
        diff_diag1 = images_diff_score_in_memory(corners['top_left'], corners['bottom_right'], min_diff=0.00001, round_digits=5)
        diff_diag2 = images_diff_score_in_memory(corners['top_right'], corners['bottom_left'], min_diff=0.00001, round_digits=5)
        ratio_diag = round(diff_diag1 / diff_diag2, 4) if diff_diag1 > diff_diag2 else round(diff_diag2 / diff_diag1, 4)
        if ratio_diag < 2:
            continue
        ratio_top_left = round(diff_left / diff_top if diff_left > diff_top else diff_top / diff_left, 4)
        ratio_top_right = round(diff_right / diff_top if diff_right > diff_top else diff_top / diff_right, 4)
        ratio_bottom_left = round(diff_left / diff_bottom if diff_left > diff_bottom else diff_bottom / diff_left, 4)
        ratio_bottom_right = round(diff_right / diff_bottom if diff_right > diff_bottom else diff_bottom / diff_right, 4)
        if ratio_top_left > 20 or ratio_top_right > 20 or ratio_bottom_left > 20 or ratio_bottom_right > 20:
            continue
        shape = None
        if direction == 'top_right_bottom_left':
            if ratio_bottom_left < ratio_top_right:
                shape = 'bottom_left'
            else:
                shape = 'top_right'
        else:
            if ratio_top_left < ratio_bottom_right:
                shape = 'top_left'
            else:
                shape = 'bottom_right'
        shape_dest_dir_path = os.path.join(dest_dir, shape)
        if not os.path.exists(shape_dest_dir_path):
            os.makedirs(shape_dest_dir_path)
        shape_dest_file_path = os.path.join(shape_dest_dir_path, f'{tile_name}.png')
        if remove_original:
            os.rename(tile_file_path, shape_dest_file_path)
        else:
            tile_image.save(shape_dest_file_path)
        print(f'{tile_name}:{shape}    {ratio} {direction} ({transposed_diff_score}:{transversed_diff_score}) {ratio_diag} top_left({ratio_top_left}) top_right({ratio_top_right}) bottom_right({ratio_bottom_right}) bottom_left({ratio_bottom_left})')
    lst = list(sorted(os.listdir(tiles_dir)))
    for tile_file_name in sorted(os.listdir(tiles_dir)):
        if not tile_file_name.endswith('.png'):
            continue
        tile_name = tile_file_name.replace('.png','')
        tile_file_path = os.path.join(tiles_dir, tile_file_name)
        tile_image = Image.open(tile_file_path)
        left_half_image = get_half_image(tile_image, 'left')
        right_half_image = get_half_image(tile_image, 'right').transpose(Transpose.FLIP_LEFT_RIGHT)
        top_half_image = get_half_image(tile_image, 'top')
        bottom_half_image = get_half_image(tile_image, 'bottom').transpose(Transpose.FLIP_TOP_BOTTOM)
        left_right_diff_score = images_diff_score_in_memory(left_half_image, right_half_image, round_digits=5)
        top_bottom_diff_score = images_diff_score_in_memory(top_half_image, bottom_half_image, round_digits=5)
        if left_right_diff_score < 0 or top_bottom_diff_score < 0:
            continue
        if left_right_diff_score < min_score and top_bottom_diff_score < min_score:
            continue
        if left_right_diff_score > top_bottom_diff_score:
            ratio = left_right_diff_score / top_bottom_diff_score
        else:
            ratio = top_bottom_diff_score / left_right_diff_score
        ratio = round(ratio, 4)
        if ratio < 2.8:
            continue
        if left_right_diff_score > top_bottom_diff_score:
            shape = 'top_bottom'
        else:
            shape = 'left_right'
        shape_dest_dir_path = os.path.join(dest_dir, shape)
        if not os.path.exists(shape_dest_dir_path):
            os.makedirs(shape_dest_dir_path)
        shape_dest_file_path = os.path.join(shape_dest_dir_path, f'{tile_name}.png')
        if remove_original:
            os.rename(tile_file_path, shape_dest_file_path)
        else:
            tile_image.save(shape_dest_file_path)
        print(f'{tile_name}    {ratio}    left_right:{left_right_diff_score}    top_bottom:{top_bottom_diff_score}')


def move_tiles_by_shape_by_four_corners(tiles_dir, dest_dir, min_score, corner_size=None, selected_tile_types=None, remove_original=False):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    diff_color_factor = 200.0
    if corner_size is None:
        corner_size = CORNER_SIZE
    for tile_file_name in sorted(os.listdir(tiles_dir)):
        if not tile_file_name.endswith('.png'):
            continue
        tile_id = tile_file_name.replace('.png','')
        tile_file_path = os.path.join(tiles_dir, tile_file_name)
        try:
            tile_image = Image.open(tile_file_path)
            tile_image.load()
        except:
            continue
        tile_average_color = average_color(tile_file_path)
        tile_corners_best = {}
        for tile_corner in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
            tile_corner_image = get_corner_image(tile_image, tile_corner, size=corner_size)
            best_score = None
            best_sample = None
            best_sample_corner = None
            best_sample_rotation = None
            for tile_rotation in [0, 90, 180, 270, -2, -1]:
                tile_corner_image_rotated = get_rotated_image(tile_corner_image, tile_rotation)
                for sample in tile_types.keys():
                    if selected_tile_types and sample not in selected_tile_types:
                        continue
                    for tile_variant in tile_types_variants[sample]:
                        for sample_corner in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
                            for sample_rotation in [0, 90, 180, 270, -2, -1]:
                                sample_corner_image = tile_types_corners[sample][tile_variant][sample_rotation][sample_corner]
                                diff_color = color_distance(tile_types_corners_average_color[sample][tile_variant][sample_rotation][sample_corner], tile_average_color)
                                # diff_score = images_diff_score_in_memory(tile_corner_image_rotated, sample_corner_image)
                                score = images_diff_score_in_memory(tile_corner_image_rotated, sample_corner_image)
                                diff_score = round(score * (1.0 - (diff_color / diff_color_factor)), 4)
                                if best_score is None or diff_score > best_score:
                                    best_score = diff_score
                                    best_sample = sample
                                    best_sample_corner = sample_corner
                                    best_sample_rotation = sample_rotation
            tile_corners_best[tile_corner] = (best_sample, best_score, best_sample_corner, best_sample_rotation)
        tl = tile_corners_best['top_left'][0]
        tr = tile_corners_best['top_right'][0]
        bl = tile_corners_best['bottom_left'][0]
        br = tile_corners_best['bottom_right'][0]
        tl_score = round(tile_corners_best['top_left'][1], 3)
        tr_score = round(tile_corners_best['top_right'][1], 3)
        bl_score = round(tile_corners_best['bottom_left'][1], 3)
        br_score = round(tile_corners_best['bottom_right'][1], 3)
        t = None
        b = None
        l = None
        r = None
        if tl_score >= min_score and tr_score >= min_score and tl == tr:
            t = tl
        if bl_score >= min_score and br_score >= min_score and bl == br:
            b = bl
        if tl_score >= min_score and bl_score >= min_score and tl == bl:
            l = tl
        if tr_score >= min_score and br_score >= min_score and tr == br:
            r = tr
        shape = None
        sample1 = None
        sample2 = None
        if t and b and t != b and not l and not r:
            shape = 'top_bottom'
            sample1 = t
            sample2 = b
        elif l and r and l != r and not t and not b:
            shape = 'left_right'
            sample1 = l
            sample2 = r
        elif t and l and t == l and br != tl:
            shape = 'top_left'
            sample1 = t
            sample2 = br
        elif t and r and t == r and bl != tr:
            shape = 'top_right'
            sample1 = t
            sample2 = bl
        elif b and l and b == l and tr != bl:
            shape = 'bottom_left'
            sample1 = b
            sample2 = tr
        elif b and r and b == r and tl != br:
            shape = 'bottom_right'
            sample1 = b
            sample2 = tl
        if not shape:
            print(f'        {tile_id}    tl:{tl}({tl_score}) tr:{tr}({tr_score}) bl:{bl}({bl_score}) br:{br}({br_score})')
            continue
        # shape_dest_dir_path = os.path.join(dest_dir, shape)
        shape_dest_file_path = os.path.join(dest_dir, f'{sample1}_{sample2}_{shape}_{tile_id}.png')
        # shape_dest_file_path = os.path.join(shape_dest_dir_path, f'{tile_id}_{sample1}_{sample2}.png')
        if remove_original:
            os.rename(tile_file_path, shape_dest_file_path)
        else:
            tile_image.save(shape_dest_file_path)
        print(f'{tile_id}    {shape}    {sample1} {sample2}    t:{t} b:{b} l:{l} r:{r}     tl:{tl}({tl_score}) tr:{tr}({tr_score}) bl:{bl}({bl_score}) br:{br}({br_score})')


def move_tiles_by_shape_by_four_corners_with_three_types(tiles_dir, dest_dir, min_score, corner_size=None, selected_tile_types=None, remove_original=False):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    diff_color_factor = 200.0
    if corner_size is None:
        corner_size = CORNER_SIZE
    for tile_file_name in sorted(os.listdir(tiles_dir)):
        if not tile_file_name.endswith('.png'):
            continue
        tile_id = tile_file_name.replace('.png','')
        tile_file_path = os.path.join(tiles_dir, tile_file_name)
        try:
            tile_image = Image.open(tile_file_path)
            tile_image.load()
        except:
            continue
        tile_average_color = average_color(tile_file_path)
        tile_corners_best = {}
        for tile_corner in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
            tile_corner_image = get_corner_image(tile_image, tile_corner, size=corner_size)
            best_score = None
            best_sample = None
            best_sample_corner = None
            best_sample_rotation = None
            for tile_rotation in [0, 90, 180, 270, -2, -1]:
                tile_corner_image_rotated = get_rotated_image(tile_corner_image, tile_rotation)
                for sample in tile_types.keys():
                    if selected_tile_types and sample not in selected_tile_types:
                        continue
                    for tile_variant in tile_types_variants[sample]:
                        for sample_corner in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
                            for sample_rotation in [0, 90, 180, 270, -2, -1]:
                                sample_corner_image = tile_types_corners[sample][tile_variant][sample_rotation][sample_corner]
                                diff_color = color_distance(tile_types_corners_average_color[sample][tile_variant][sample_rotation][sample_corner], tile_average_color)
                                # diff_score = images_diff_score_in_memory(tile_corner_image_rotated, sample_corner_image)
                                score = images_diff_score_in_memory(tile_corner_image_rotated, sample_corner_image)
                                diff_score = round(score * (1.0 - (diff_color / diff_color_factor)), 4)
                                if best_score is None or diff_score > best_score:
                                    best_score = diff_score
                                    best_sample = sample
                                    best_sample_corner = sample_corner
                                    best_sample_rotation = sample_rotation
            tile_corners_best[tile_corner] = (best_sample, best_score, best_sample_corner, best_sample_rotation)
        tl = tile_corners_best['top_left'][0]
        tr = tile_corners_best['top_right'][0]
        bl = tile_corners_best['bottom_left'][0]
        br = tile_corners_best['bottom_right'][0]
        tl_score = round(tile_corners_best['top_left'][1], 3)
        tr_score = round(tile_corners_best['top_right'][1], 3)
        bl_score = round(tile_corners_best['bottom_left'][1], 3)
        br_score = round(tile_corners_best['bottom_right'][1], 3)
        if tl_score < min_score:
            tl = None
        if tr_score < min_score:
            tr = None
        if bl_score < min_score:
            bl = None
        if br_score < min_score:
            br = None
        if not tl or not tr or not bl or not br:
            print(f'        {tile_id}    tl:{tl}({tl_score}) tr:{tr}({tr_score}) bl:{bl}({bl_score}) br:{br}({br_score})')
            continue
        t = None
        b = None
        l = None
        r = None
        tlbr = None
        trbl = None
        if tl == tr:
            t = tl
        if bl == br:
            b = bl
        if tl == bl:
            l = tl
        if tr == br:
            r = tr
        if tl == br:
            tlbr = tl
        if tr == bl:
            trbl = tr
        shape = None
        sample1 = None
        sample2 = None
        sample3 = None
        if t and not b and not l and not r:
            shape = 'top_top'
            sample1 = t
            sample2 = bl
            sample3 = br
        elif b and not t and not l and not r:
            shape = 'bottom_bottom'
            sample1 = b
            sample2 = tr
            sample3 = tl
        elif l and not t and not b and not r:
            shape = 'left_left'
            sample1 = l
            sample2 = tr
            sample3 = br
        elif r and not t and not b and not l:
            shape = 'right_right'
            sample1 = r
            sample2 = tl
            sample3 = bl
        elif tlbr and not trbl and not t and not b and not l and not r:
            shape = 'topleft_bottomright'
            sample1 = tlbr
            sample2 = tr
            sample3 = bl
        elif trbl and not tlbr and not t and not b and not l and not r:
            shape = 'topright_bottomleft'
            sample1 = trbl
            sample2 = tl
            sample3 = br
        if not shape:
            print(f'        {tile_id}    tl:{tl}({tl_score}) tr:{tr}({tr_score}) bl:{bl}({bl_score}) br:{br}({br_score})')
            continue
        # shape_dest_dir_path = os.path.join(dest_dir, shape)
        shape_dest_file_path = os.path.join(dest_dir, f'{sample1}_{sample2}_{sample3}_{shape}_{tile_id}.png')
        # shape_dest_file_path = os.path.join(shape_dest_dir_path, f'{tile_id}_{sample1}_{sample2}.png')
        if remove_original:
            os.rename(tile_file_path, shape_dest_file_path)
        else:
            tile_image.save(shape_dest_file_path)
        print(f'{tile_id}    {shape}    {sample1} {sample2} {sample3}   tl:{tl}({tl_score}) tr:{tr}({tr_score}) bl:{bl}({bl_score}) br:{br}({br_score})')


def merge_tiles(src_dir, group_dir, dest_dir, ready_dir, save_ready_tiles=False, save_merged_tiles=True):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    registry = {}
    graph = {}
    lst = sorted(os.listdir(src_dir))
    pairs = set()
    triples = set()
    for tile_file_name in lst:
        if not tile_file_name.endswith('.png'):
            continue
        tile_name = tile_file_name.replace('.png','').rstrip('_')
        tile_file_path = os.path.join(src_dir, tile_file_name)
        tile_image = Image.open(tile_file_path)
        tile_image.load()
        parts = tile_name.split('_')
        if len(parts) != 5:
            continue
        tile_sample2, tile_sample1, tile_side1, tile_side2, tile_id = parts
        tile_side = f'{tile_side1}_{tile_side2}'
        tile_side = {'top_left':'top_left', 'top_right':'top_right', 'bottom_left':'bottom_left', 'bottom_right':'bottom_right', 'top_bottom':'top', 'left_right':'left'}[tile_side]
        if tile_sample1 not in registry:
            registry[tile_sample1] = {}
        if tile_sample2 not in registry[tile_sample1]:
            registry[tile_sample1][tile_sample2] = {}
        if tile_side not in registry[tile_sample1][tile_sample2]:
            registry[tile_sample1][tile_sample2][tile_side] = []
        registry[tile_sample1][tile_sample2][tile_side].append((tile_image, tile_id, tile_file_path))
        if tile_sample1 not in graph:
            graph[tile_sample1] = {}
        if tile_sample2 not in graph[tile_sample1]:
            graph[tile_sample1][tile_sample2] = []
        if tile_sample2 not in graph:
            graph[tile_sample2] = {}
        if tile_sample1 not in graph[tile_sample2]:
            graph[tile_sample2][tile_sample1] = []
    for tile_sample1 in registry.keys():
        for tile_sample2 in registry[tile_sample1].keys():
            for side in ['top', 'left']:
                if side not in registry[tile_sample1][tile_sample2] or len(registry[tile_sample1][tile_sample2][side]) == 0:
                    if tile_sample2 in registry:
                        if tile_sample1 in registry[tile_sample2]:
                            if side in registry[tile_sample2][tile_sample1]:
                                tile_image, tile_id, tile_file_path = registry[tile_sample2][tile_sample1][side][0]
                                tile_image_flipped = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM) if side == 'top' else tile_image.transpose(Transpose.FLIP_LEFT_RIGHT)
                                registry[tile_sample1][tile_sample2][side] = [(tile_image_flipped, tile_id, tile_file_path)]
    for tile_sample1 in sorted(registry.keys()):
        for tile_sample2 in sorted(registry[tile_sample1].keys()):
            for tile_sample3 in sorted(registry.keys()):
                triples.add(tuple(sorted([tile_sample1, tile_sample2, tile_sample3])))
            fragments = registry[tile_sample1][tile_sample2]
            shortest_stack = min([len(fragments[side]) for side in fragments.keys()])
            longest_stack = max([len(fragments[side]) for side in fragments.keys()])
            stack_difference = longest_stack - shortest_stack
            # for i in range(shortest_stack):
            for j in range(longest_stack):
                im = Image.new("RGB", (TILE_SIZE * 3, TILE_SIZE * 3), "black")
                if tile_sample1 not in tile_types_variants:
                    raise Exception(f'combination {tile_sample1}_{tile_sample2} not found in tile types variants')
                # for tile_variant in tile_types_variants[tile_sample1]:
                for tile_variant in ['a', ]:
                    im.paste(tile_types_images[tile_sample1][tile_variant][0], (TILE_SIZE, TILE_SIZE))
                    tiles_ids = []
                    top_image = None
                    top_right_image = None
                    top_tile_file_path = None
                    top_right_tile_file_path = None
                    for side in ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top', 'left', 'bottom', 'right']:
                        tile_image = None
                        tile_id = None
                        if side in fragments:
                            k = j if j < len(fragments[side]) else len(fragments[side]) - 1
                            tile_image, tile_id, tile_file_path = fragments[side][k]
                        else:
                            if side == 'top':
                                if 'bottom' in fragments:
                                    i = j if j < len(fragments['bottom']) else len(fragments['bottom']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['bottom'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'left' in fragments:
                                    i = j if j < len(fragments['left']) else len(fragments['left']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['left'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM).transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'right' in fragments:
                                    i = j if j < len(fragments['right']) else len(fragments['right']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['right'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM).transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                            elif side == 'bottom':
                                if 'top' in fragments:
                                    i = j if j < len(fragments['top']) else len(fragments['top']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['top'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'left' in fragments:
                                    i = j if j < len(fragments['left']) else len(fragments['left']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['left'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM).transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'right' in fragments:
                                    i = j if j < len(fragments['right']) else len(fragments['right']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['right'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM).transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                            elif side == 'left':
                                if 'right' in fragments:
                                    i = j if j < len(fragments['right']) else len(fragments['right']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['right'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'bottom' in fragments:
                                    i = j if j < len(fragments['bottom']) else len(fragments['bottom']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['bottom'][i]
                                    tile_image = tile_image.transpose(Transpose.ROTATE_270)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'top' in fragments:
                                    i = j if j < len(fragments['top']) else len(fragments['top']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['top'][i]
                                    tile_image = tile_image.transpose(Transpose.ROTATE_90)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                            elif side == 'right':
                                if 'left' in fragments:
                                    i = j if j < len(fragments['left']) else len(fragments['left']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['left'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'bottom' in fragments:
                                    i = j if j < len(fragments['bottom']) else len(fragments['bottom']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['bottom'][i]
                                    tile_image = tile_image.transpose(Transpose.ROTATE_90)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'top' in fragments:
                                    i = j if j < len(fragments['top']) else len(fragments['top']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['top'][i]
                                    tile_image = tile_image.transpose(Transpose.ROTATE_270)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                            elif side == 'top_left':
                                if 'bottom_right' in fragments:
                                    i = j if j < len(fragments['bottom_right']) else len(fragments['bottom_right']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['bottom_right'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM).transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'top_right' in fragments:
                                    i = j if j < len(fragments['top_right']) else len(fragments['top_right']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['top_right'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'bottom_left' in fragments:
                                    i = j if j < len(fragments['bottom_left']) else len(fragments['bottom_left']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['bottom_left'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                            elif side == 'top_right':
                                if 'bottom_left' in fragments:
                                    i = j if j < len(fragments['bottom_left']) else len(fragments['bottom_left']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['bottom_left'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM).transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'top_left' in fragments:
                                    i = j if j < len(fragments['top_left']) else len(fragments['top_left']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['top_left'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'bottom_right' in fragments:
                                    i = j if j < len(fragments['bottom_right']) else len(fragments['bottom_right']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['bottom_right'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                            elif side == 'bottom_left':
                                if 'top_right' in fragments:
                                    i = j if j < len(fragments['top_right']) else len(fragments['top_right']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['top_right'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM).transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'top_left' in fragments:
                                    i = j if j < len(fragments['top_left']) else len(fragments['top_left']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['top_left'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'bottom_right' in fragments:
                                    i = j if j < len(fragments['bottom_right']) else len(fragments['bottom_right']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['bottom_right'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                            elif side == 'bottom_right':
                                if 'top_left' in fragments:
                                    i = j if j < len(fragments['top_left']) else len(fragments['top_left']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['top_left'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM).transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'top_right' in fragments:
                                    i = j if j < len(fragments['top_right']) else len(fragments['top_right']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['top_right'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                                elif 'bottom_left' in fragments:
                                    i = j if j < len(fragments['bottom_left']) else len(fragments['bottom_left']) - 1
                                    tile_image, tile_id, tile_file_path = fragments['bottom_left'][i]
                                    tile_image = tile_image.transpose(Transpose.FLIP_LEFT_RIGHT)
                                    if side not in registry[tile_sample1][tile_sample2]:
                                        registry[tile_sample1][tile_sample2][side] = [(tile_image, tile_id, tile_file_path)]
                        if not tile_image:
                            continue
                        if tile_id not in tiles_ids:
                            tiles_ids.append(tile_id)
                        if side == 'top':
                            pos = (TILE_SIZE, 0)
                        elif side == 'bottom':
                            pos = (TILE_SIZE, TILE_SIZE * 2)
                        elif side == 'left':
                            pos = (0, TILE_SIZE)
                        elif side == 'right':
                            pos = (TILE_SIZE * 2, TILE_SIZE)
                        elif side == 'top_left':
                            pos = (0, 0)
                        elif side == 'top_right':
                            pos = (TILE_SIZE * 2, 0)
                        elif side == 'bottom_left':
                            pos = (0, TILE_SIZE * 2)
                        elif side == 'bottom_right':
                            pos = (TILE_SIZE * 2, TILE_SIZE * 2)
                        im.paste(tile_image, pos)
                        if side == 'top':
                            top_tile_file_path = tile_file_path
                            top_image = tile_image
                        elif side == 'top_right':
                            top_right_tile_file_path = tile_file_path
                            top_right_image = tile_image
                    tiles_ids = '_'.join(tiles_ids)
                    merged_tile_name = f'{tile_sample1}_{tile_sample2}_{j}_{tiles_ids}.png'
                    merged_tile_file_path = os.path.join(dest_dir, merged_tile_name)
                    if save_merged_tiles:
                        im.save(merged_tile_file_path)
                        print(f'    saved merged fragment to {merged_tile_name}, more possible fragments: {stack_difference}')
                    if top_tile_file_path and top_right_tile_file_path:
                        graph[tile_sample1][tile_sample2].append({'top': top_image, 'top_right': top_right_image})
                        pairs.add((tile_sample1, tile_sample2))
    catalog = {}
    counter = 0
    if save_ready_tiles:
        for tile_sample in tile_types.keys():
            for tile_variant in tile_types_variants[tile_sample]:
                key = tile_sample
                if key not in catalog:
                    catalog[key] = []
                counter += 1
                tile_image = tile_types_images[tile_sample][tile_variant][0]
                ready_tile_name = f'{counter:05d}.png'
                tile_image.save(os.path.join(ready_dir, ready_tile_name))
                catalog[key].append(counter)
        for tile_sample1 in graph.keys():
            for tile_sample2 in graph[tile_sample1].keys():
                for i in range(len(graph[tile_sample1][tile_sample2])):
                    key = f'{tile_sample1}_{tile_sample2}'
                    if key not in catalog:
                        catalog[key] = {
                            's': [],
                            'c': [],
                        }
                    counter += 1
                    top_image = graph[tile_sample1][tile_sample2][i]['top']
                    ready_top_tile_name = f'{counter:05d}.png'
                    top_image.save(os.path.join(ready_dir, ready_top_tile_name))
                    catalog[key]['s'].append(counter)
                    counter += 1
                    top_right_image = graph[tile_sample1][tile_sample2][i]['top_right']
                    ready_top_right_tile_name = f'{counter:05d}.png'
                    top_right_image.save(os.path.join(ready_dir, ready_top_right_tile_name))
                    catalog[key]['c'].append(counter)
        open('catalog.json', 'w').write(json.dumps(catalog, indent=2))
    for pair in pairs:
        pair2 = (pair[1], pair[0])
        if pair2 not in pairs:
            print(f'Warning: missing pair {pair2} for {pair}')
    registry3types = {}
    missing = set()
    quadruples = set()
    lst = sorted(os.listdir(src_dir))
    for tile_file_name in lst:
        if not tile_file_name.endswith('.png'):
            continue
        tile_name = tile_file_name.replace('.png','').rstrip('_')
        tile_file_path = os.path.join(src_dir, tile_file_name)
        tile_image = Image.open(tile_file_path)
        tile_image.load()
        parts = tile_name.split('_')
        if len(parts) != 6:
            continue
        tile_sample1 = None
        tile_sample2 = None
        tile_sample3 = None
        tile_sample1, tile_sample2, tile_sample3, tile_side1, tile_side2, tile_id = parts
        tile_side = f'{tile_side1}_{tile_side2}'
        if tile_side not in ['top_top', 'bottom_bottom', 'left_left', 'right_right', 'topleft_bottomright', 'topright_bottomleft']:
            continue
        two_samples = f'{tile_sample2}_{tile_sample3}'
        if tile_sample1 not in registry3types:
            registry3types[tile_sample1] = {}
        if two_samples not in registry3types[tile_sample1]:
            registry3types[tile_sample1][two_samples] = []
        registry3types[tile_sample1][two_samples].append((tile_side, tile_image, tile_id, tile_file_path))
    count = 0
    ready = {}
    for tile_sample1 in sorted(registry3types.keys()):
        for two_samples in sorted(registry3types[tile_sample1].keys()):
            tile_sample2, tile_sample3 = two_samples.split('_')
            if tile_sample1 not in tile_types_variants:
                raise Exception(f'combination {tile_sample1} not found in tile types variants')
            if tile_sample2 not in tile_types_variants:
                raise Exception(f'combination {tile_sample2} not found in tile types variants')
            if tile_sample3 not in tile_types_variants:
                raise Exception(f'combination {tile_sample3} not found in tile types variants')
            if tile_sample1 not in registry:
                continue
            if tile_sample2 not in registry:
                continue
            if tile_sample3 not in registry:
                continue
            if tile_sample2 not in registry[tile_sample1]:
                missing.add(tuple(sorted((tile_sample1, tile_sample2))))
                # print(f'Warning: missing registry entry for {tile_sample1} {tile_sample2} pair, skipping merging for {tile_sample1}_{two_samples}')
                continue
            if tile_sample3 not in registry[tile_sample1]:
                missing.add(tuple(sorted((tile_sample1, tile_sample3))))
                # print(f'Warning: missing registry entry for {tile_sample1} {tile_sample3} pair, skipping merging for {tile_sample1}_{two_samples}')
                continue
            if tile_sample3 not in registry[tile_sample2]:
                missing.add(tuple(sorted((tile_sample2, tile_sample3))))
                # print(f'Warning: missing registry entry for {tile_sample2} {tile_sample3} pair, skipping merging for {tile_sample1}_{two_samples}')
                continue
            fragments = registry3types[tile_sample1][two_samples]
            for tile_side, tile_image, tile_id, tile_file_path in fragments:
                im = Image.new("RGB", (TILE_SIZE * 3, TILE_SIZE * 3), "black")
                im.paste(tile_image, (TILE_SIZE, TILE_SIZE))
                tile_variant = 'a'
                ready_image1 = None
                ready_image2 = None
                q1 = None
                q2 = None
                try:
                    if tile_side == 'top_top':
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (0, 0))
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (TILE_SIZE, 0))
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (TILE_SIZE * 2, 0))
                        im.paste(tile_types_images[tile_sample2][tile_variant][0], (0, TILE_SIZE * 2))
                        im.paste(tile_types_images[tile_sample3][tile_variant][0], (TILE_SIZE * 2, TILE_SIZE * 2))
                        im.paste(registry[tile_sample1][tile_sample2]['bottom'][0][0], (0, TILE_SIZE))
                        im.paste(registry[tile_sample1][tile_sample3]['bottom'][0][0], (TILE_SIZE * 2, TILE_SIZE))
                        im.paste(registry[tile_sample2][tile_sample3]['right'][0][0], (TILE_SIZE, TILE_SIZE * 2))
                        ready_image1 = tile_image
                        ready_image2 = tile_image.transpose(Transpose.FLIP_LEFT_RIGHT)
                        q1 = (tile_sample1, tile_sample1, tile_sample2, tile_sample3)
                        q2 = (tile_sample1, tile_sample1, tile_sample3, tile_sample2)
                    elif tile_side == 'bottom_bottom':
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (0, TILE_SIZE * 2))
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (TILE_SIZE, TILE_SIZE * 2))
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (TILE_SIZE * 2, TILE_SIZE * 2))
                        im.paste(tile_types_images[tile_sample2][tile_variant][0], (TILE_SIZE * 2, 0))
                        im.paste(tile_types_images[tile_sample3][tile_variant][0], (0, 0))
                        im.paste(registry[tile_sample1][tile_sample2]['top'][0][0], (TILE_SIZE * 2, TILE_SIZE))
                        im.paste(registry[tile_sample1][tile_sample3]['top'][0][0], (0, TILE_SIZE))
                        im.paste(registry[tile_sample2][tile_sample3]['left'][0][0], (TILE_SIZE, 0))
                        ready_image1 = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM).transpose(Transpose.FLIP_LEFT_RIGHT)
                        ready_image2 = tile_image.transpose(Transpose.FLIP_TOP_BOTTOM)
                        q1 = (tile_sample1, tile_sample1, tile_sample2, tile_sample3)
                        q2 = (tile_sample1, tile_sample1, tile_sample3, tile_sample2)
                    elif tile_side == 'left_left':
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (0, 0))
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (0, TILE_SIZE))
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (0, TILE_SIZE * 2))
                        im.paste(tile_types_images[tile_sample2][tile_variant][0], (TILE_SIZE * 2, 0))
                        im.paste(tile_types_images[tile_sample3][tile_variant][0], (TILE_SIZE * 2, TILE_SIZE * 2))
                        im.paste(registry[tile_sample1][tile_sample2]['right'][0][0], (TILE_SIZE, 0))
                        im.paste(registry[tile_sample1][tile_sample3]['right'][0][0], (TILE_SIZE, TILE_SIZE * 2))
                        im.paste(registry[tile_sample2][tile_sample3]['bottom'][0][0], (TILE_SIZE * 2, TILE_SIZE))
                        ready_image1 = tile_image.transpose(Transpose.ROTATE_90)
                        ready_image2 = tile_image.transpose(Transpose.ROTATE_90).transpose(Transpose.FLIP_LEFT_RIGHT)
                        q1 = (tile_sample1, tile_sample1, tile_sample2, tile_sample3)
                        q2 = (tile_sample1, tile_sample1, tile_sample3, tile_sample2)
                    elif tile_side == 'right_right':
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (TILE_SIZE * 2, 0))
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (TILE_SIZE * 2, TILE_SIZE))
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (TILE_SIZE * 2, TILE_SIZE * 2))
                        im.paste(tile_types_images[tile_sample2][tile_variant][0], (0, 0))
                        im.paste(tile_types_images[tile_sample3][tile_variant][0], (0, TILE_SIZE * 2))
                        im.paste(registry[tile_sample1][tile_sample2]['left'][0][0], (TILE_SIZE, 0))
                        im.paste(registry[tile_sample1][tile_sample3]['left'][0][0], (TILE_SIZE, TILE_SIZE * 2))
                        im.paste(registry[tile_sample2][tile_sample3]['bottom'][0][0], (0, TILE_SIZE))
                        ready_image1 = tile_image.transpose(Transpose.ROTATE_270)
                        ready_image2 = tile_image.transpose(Transpose.ROTATE_270).transpose(Transpose.FLIP_LEFT_RIGHT)
                        q1 = (tile_sample1, tile_sample1, tile_sample2, tile_sample3)
                        q2 = (tile_sample1, tile_sample1, tile_sample3, tile_sample2)
                    elif tile_side == 'topleft_bottomright':
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (0, 0))
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (TILE_SIZE * 2, TILE_SIZE * 2))
                        im.paste(tile_types_images[tile_sample2][tile_variant][0], (TILE_SIZE * 2, 0))
                        im.paste(tile_types_images[tile_sample3][tile_variant][0], (0, TILE_SIZE * 2))
                        im.paste(registry[tile_sample1][tile_sample2]['right'][0][0], (TILE_SIZE, 0))
                        im.paste(registry[tile_sample1][tile_sample3]['left'][0][0], (TILE_SIZE, TILE_SIZE * 2))
                        im.paste(registry[tile_sample1][tile_sample2]['top'][0][0], (TILE_SIZE * 2, TILE_SIZE))
                        im.paste(registry[tile_sample1][tile_sample3]['bottom'][0][0], (0, TILE_SIZE))
                        ready_image1 = tile_image
                        ready_image2 = tile_image.transpose(Transpose.FLIP_LEFT_RIGHT).transpose(Transpose.FLIP_TOP_BOTTOM)
                        q1 = (tile_sample1, tile_sample2, tile_sample1, tile_sample3)
                        q2 = (tile_sample1, tile_sample3, tile_sample1, tile_sample2)
                    elif tile_side == 'topright_bottomleft':
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (TILE_SIZE * 2, 0))
                        im.paste(tile_types_images[tile_sample1][tile_variant][0], (0, TILE_SIZE * 2))
                        im.paste(tile_types_images[tile_sample2][tile_variant][0], (0, 0))
                        im.paste(tile_types_images[tile_sample3][tile_variant][0], (TILE_SIZE * 2, TILE_SIZE * 2))
                        im.paste(registry[tile_sample1][tile_sample2]['left'][0][0], (TILE_SIZE, 0))
                        im.paste(registry[tile_sample1][tile_sample3]['right'][0][0], (TILE_SIZE, TILE_SIZE * 2))
                        im.paste(registry[tile_sample1][tile_sample2]['top'][0][0], (0, TILE_SIZE))
                        im.paste(registry[tile_sample1][tile_sample3]['bottom'][0][0], (TILE_SIZE * 2, TILE_SIZE))
                        ready_image1 = tile_image.transpose(Transpose.ROTATE_270)
                        ready_image2 = tile_image.transpose(Transpose.ROTATE_270).transpose(Transpose.FLIP_LEFT_RIGHT).transpose(Transpose.FLIP_TOP_BOTTOM)
                        q1 = (tile_sample1, tile_sample2, tile_sample1, tile_sample3)
                        q2 = (tile_sample1, tile_sample3, tile_sample1, tile_sample2)

                except:
                    print(f'Error merging tile_sample1={tile_sample1} tile_sample2={tile_sample2} tile_sample3={tile_sample3} for tile {tile_id} with side {tile_side}')
                    raise
                quadruples.add(q1)
                quadruples.add(q2)
                merged_tile_name = f'{q1[0]}_{q1[1]}_{q1[2]}_{q1[3]}_{tile_id}.png'
                merged_tile_file_path = os.path.join(dest_dir, merged_tile_name)
                if save_merged_tiles:
                    im.save(merged_tile_file_path)
                if q1 not in ready:
                    ready[q1] = []
                if q2 not in ready:
                    ready[q2] = []
                ready[q1].append(ready_image1)
                ready[q2].append(ready_image2)
                count += 1
                print(f'    saved merged fragment to {merged_tile_name}')
    print(f'processed {count} 3-types tiles, {len(missing)} missing pairs')
    print(f'merged {len(quadruples)} quadruples:')
    for q in sorted(quadruples):
        print(f'        {q},')
    if save_ready_tiles:
        for q in ready.keys():
            tile_sample1, tile_sample2, tile_sample3, tile_sample4 = q
            key = f'{tile_sample1}_{tile_sample2}_{tile_sample3}_{tile_sample4}'
            if key not in catalog:
                catalog[key] = []
            for ready_image in ready[q]:
                # ready_image = ready[q]
                counter += 1
                ready_tile_name = f'{counter:05d}.png'
                ready_image.save(os.path.join(ready_dir, ready_tile_name))
                catalog[key].append(counter)
        open('catalog.json', 'w').write(json.dumps(catalog, indent=2))


def build_index(src_dir, dest_dir):
    global index_by_hash
    processed = 0
    total_tiles = 0
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file_name in sorted(os.listdir(src_dir)):
        if not file_name.endswith('.png'):
            print(f'Skipping {file_name}, not a .png file')
            continue
        processed += 1
        img_name = file_name.replace('.png', '')
        map_name = img_name[:-3]
        map_image_variant = int(img_name[-3:])
        file_path = os.path.join(src_dir, file_name)
        source_image = Image.open(file_path)
        source_image.load()
        if source_image.width != 512 or source_image.height != 512:
            print(f'Skipping {file_path}, wrong dimensions: {source_image.width}x{source_image.height}')
            continue
        count = 0
        for h in range(8):
            for w in range(8):
                tile_texture, tile_hash = image_tile(source_image, w, h)
                if tile_hash in index_by_hash:
                    continue
                hash_found = False
                for rotation in [90, 180, 270, -1, -2]:
                    tile_texture_rotated = get_rotated_image(tile_texture, rotation)
                    tile_hash_rotated = hashlib.md5(tile_texture_rotated.tobytes()).hexdigest()
                    if tile_hash_rotated in index_by_hash:
                        hash_found = True
                        break
                if hash_found:
                    continue
                for rotation in [0, ]:  # [0, 90, 180, 270]:
                    total_tiles += 1
                    tile_file_path = os.path.abspath(os.path.join(dest_dir, f'{total_tiles:05d}.png'))
                    # if os.path.exists(tile_file_path):
                    #     if tile_hash not in index_by_hash:
                    #         add_to_index(tile_hash, map_name, map_image_variant, w, h)
                    #     continue
                    tile_texture_rotated = get_rotated_image(tile_texture, rotation)
                    tile_texture_rotated.save(tile_file_path)
                add_to_index(tile_hash, map_name, map_image_variant, w, h)
                count += 1
        print(f'read {count} tiles from {file_path}')
    print(f'processed {processed} source images, generated {total_tiles} tiles')
    open('index.json', 'w').write(json.dumps(index_by_hash, indent=2))
    print(f'indexed {len(index_by_hash)} unique tiles')


def main():
    stage = sys.argv[1]
    if stage == 'stage1':
        build_index(sys.argv[2], sys.argv[3])
    elif stage == 'stage2':
        read_tile_types()
        detected_tile_types = move_tiles_by_type(sys.argv[2], sys.argv[3], float(sys.argv[4]))
        print(','.join(sorted(list(detected_tile_types))))
    elif stage == 'stage3':
        read_tile_types(corner_size=int(sys.argv[5]))
        move_tiles_by_shape_by_four_corners(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6].split(','))
    elif stage == 'stage4':
        read_tile_types(corner_size=int(sys.argv[5]))
        move_tiles_by_shape_by_four_corners_with_three_types(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6].split(','))
    # elif stage == 'stage3a':
    #     move_tiles_by_shape(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    # elif stage == 'stage4':
    #     read_tile_types()
    #     move_tiles_by_one_side_similarity(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5], sys.argv[6].split(','))
    # elif stage == 'stage5':
    #     read_tile_types()
    #     move_tiles_by_one_corner_similarity(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5], sys.argv[6].split(','))
    # elif stage == 'stage6':
    #     read_tile_types()
    #     move_tiles_within_group_by_second_side_similarity(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5], sys.argv[6].split(','))
    # elif stage == 'stage7':
    #     read_tile_types()
    #     move_tiles_within_group_by_second_corner_similarity(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5], sys.argv[6].split(','))
    elif stage == 'stage8':
        read_tile_types()
        merge_tiles(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], save_ready_tiles=False)
    elif stage == 'stage9':
        read_tile_types()
        merge_tiles(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], save_ready_tiles=True)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage:')
        print('python3 tiles.py stage1 <source folder with .png files of the land tile> <tiles folder>')
        print('python3 tiles.py stage2 <tiles folder> <grouped tiles folder> <similarity score threshold> [selected tile types]')
        print('python3 tiles.py stage3 <tiles folder> <shape tiles folder> <similarity score threshold> <corner size> <selected tile types>')
        print('python3 tiles.py stage4 <tiles folder> <shape tiles folder> <similarity score threshold> <corner size> <selected tile types>')
        # print('python3 tiles.py stage3a <tiles folder> <shape tiles folder> <similarity score threshold>')
        # print('python3 tiles.py stage4 <tiles folder> <grouped tiles folder> <similarity score threshold> <selected side> <selected tile types>')
        # print('python3 tiles.py stage5 <tiles folder> <grouped tiles folder> <similarity score threshold> <selected corner> <selected tile types>')
        # print('python3 tiles.py stage6 <grouped tiles folder> <destination folder> <similarity score threshold> <selected side> <selected tile types>')
        # print('python3 tiles.py stage7 <grouped tiles folder> <destination folder> <similarity score threshold> <selected corner> <selected tile types>')
        print('python3 tiles.py stage8 <source folder> <grouped tiles folder> <destination folder> <ready tiles folder>')
        print('python3 tiles.py stage9 <source folder> <grouped tiles folder> <destination folder> <ready tiles folder>')
        sys.exit(-1)
    main()
