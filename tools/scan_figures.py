import os
import sys
import json

from binary_readers import *
from textures_link import textures


def add_child_to_dict(dct, parent_name, child_name):
    if parent_name in dct:
        if child_name in dct[parent_name]:
            return
        dct[parent_name][child_name] = {}
        return
    for other_name in dct.keys():
        add_child_to_dict(dct[other_name], parent_name, child_name)


def add_child_to_list(arr, parent_name, child_name):
    if arr[0] == parent_name:
        arr[1].append([child_name, []])
    else:
        for part_name in arr[1]:
            add_child_to_list(part_name, parent_name, child_name)


def flat_tree(tree, arr=None):
    if arr is None:
        arr = []
    arr.append(tree[0])
    if len(tree[1]) != 0:
        for leaf in tree[1]:
            flat_tree(leaf, arr)
    return arr


def read_lnk_info(file_name):
    lst = []
    parents = {}
    tree = {}
    childs = {}
    with open(file_name, "rb") as file:
        for _ in range(read_uint(file)):
            name_len = read_uint(file)
            new_name = read_str(file, name_len)
            parent_name_len = read_uint(file)
            if parent_name_len == 0:
                lst = [new_name, []]
                parents[new_name] = None
                if new_name not in childs:
                    childs[new_name] = []
                tree[new_name] = {}
            else:
                parent_name = read_str(file, parent_name_len)
                parents[new_name] = parent_name
                if new_name not in childs:
                    childs[new_name] = []
                if parent_name not in childs:
                    childs[parent_name] = []
                childs[parent_name].append(new_name)
                add_child_to_list(lst, parent_name, new_name)
                add_child_to_dict(tree, parent_name, new_name)
    return lst, tree, parents, childs


def main():
    figures_catalog = []
    for model_name in os.listdir(sys.argv[1]):
        model_name = model_name.lower()
        dir_path = os.path.join(sys.argv[1], model_name)
        if not os.path.isdir(dir_path):
            continue
        lnk_path = os.path.join(dir_path, f"{model_name}.lnk")
        if not os.path.isfile(lnk_path):
            print(f"WARNING file not found: {lnk_path}")
            continue
        lst, tree, parents, childs = read_lnk_info(lnk_path)
        parts_list = flat_tree(lst)
        texture_name = textures.get(model_name, ['default0', ])[0]
        template_name = f"{model_name}#{texture_name}#{':'.join(parts_list)}"
        figures_catalog.append(template_name)
        print(template_name)
    figures_catalog.sort()
    open('catalog_figures.json', 'wt').write(json.dumps(figures_catalog, indent=2))


if __name__ == '__main__':
    main()
