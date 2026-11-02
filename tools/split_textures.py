"""
Splits big textures.res into several smaller .res files: textures are grouped by what they belong to
(trees, armor, weapons, creatures, terrain of every zone and so on), every group goes into <group>.res.
Groups are defined by TEXTURE_GROUPS in src/lib/const.py, textures which match no rule go to objects.res.

With --by-model (on by default for redress.res) textures are grouped by the model they belong to instead:
names are <model><part / armor / weapon / material>.mmp, the model is the longest model name from figures.res
the texture starts with, every model goes into <model>.res. figures.res is taken next to the source file or from --figures.
Models from REDRESS_SPLIT_MODELS in src/lib/const.py (unhuma, unhufe) are split further by the kind of armor / weapon:
unhufeax_01.br.1 -> unhufe_ax.res, unhufeskin_00 -> unhufe_skin.res, and pictures of wounds (unhufelhw1) -> unhufe_w.res.
So the client downloads only the files of the models it shows.

Also writes <source>_index.json next to the files (textures_index.json, redress_index.json): {texture: group},
so the client knows which file holds a texture.
Every small file is read with the same functions from src/lib/res.py as textures.res before.

    PYTHONPATH=src venv/bin/python3 tools/split_textures.py ~/.worlds/data/textures.res ./textures/
    PYTHONPATH=src venv/bin/python3 tools/split_textures.py ~/.worlds/data/textures.res ./textures/ --groups trees,armor
    PYTHONPATH=src venv/bin/python3 tools/split_textures.py ~/.worlds/data/textures.res ./textures/ --list
    PYTHONPATH=src venv/bin/python3 tools/split_textures.py ~/.worlds/data/redress.res ./redress/
    PYTHONPATH=src venv/bin/python3 tools/split_textures.py ~/.worlds/data/redress.res ./redress/ --groups unhufe --list

With --unpack nothing is split: every .mmp file from the source .res is written as a separate file
into the destination folder (any .res file works, also one of the split files). --groups limits it to those groups.

    PYTHONPATH=src venv/bin/python3 tools/split_textures.py ~/.worlds/data/textures.res ./textures_mmp/ --unpack
    PYTHONPATH=src venv/bin/python3 tools/split_textures.py ./textures/trees.res ./trees_mmp/ --unpack
    PYTHONPATH=src venv/bin/python3 tools/split_textures.py ~/.worlds/data/textures.res ./trees_mmp/ --unpack --groups trees
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import sys
import json
import argparse

#------------------------------------------------------------------------------

from lib import res

#------------------------------------------------------------------------------


def unpack_mmp(res_file, filetree_dict, names, destination):
    """
    Writes every .mmp element from the list of names as a separate file into destination folder.
    Sub folders inside the .res file are kept, existing files are overwritten.
    """
    dest_root = os.path.abspath(destination)
    written = 0
    total_size = 0
    skipped = 0
    for name in sorted(names):
        if not name.lower().endswith('.mmp'):
            continue
        _, offset, size = filetree_dict[name]
        mmp_file_path = os.path.abspath(os.path.join(dest_root, name))
        # element names come from the file, do not let them point outside of destination
        if not mmp_file_path.startswith(dest_root + os.sep):
            print(f'WARNING: skipped {name}, path is outside of {destination}')
            skipped += 1
            continue
        os.makedirs(os.path.dirname(mmp_file_path), exist_ok=True)
        res_file.seek(offset)
        png_file_path = mmp_file_path.replace('.mmp', '.png')
        if os.path.exists(png_file_path):
            raise Exception(png_file_path)
        with open(mmp_file_path, 'wb') as f:
            f.write(res_file.read(size))
        pil_img = res.read_mmp(mmp_file_path)
        pil_img.save(png_file_path)
        os.remove(mmp_file_path)
        written += 1
        total_size += size
    print(f'unpacked {written} .mmp files into {destination}, total size {total_size / 1e6:.1f} MB')
    if skipped:
        print(f'{skipped} files skipped')
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description='split textures.res (or redress.res) into one .res file per group of textures')
    parser.add_argument('source', help='path to textures.res or redress.res')
    parser.add_argument('destination', help='output folder for <group>.res files')
    parser.add_argument('--groups', default='', help='comma separated list of groups, all groups by default')
    parser.add_argument('--list', action='store_true', help='only print the groups and their textures, write nothing')
    parser.add_argument('--unpack', action='store_true', help='do not split, just unpack all .mmp files into the destination folder')
    parser.add_argument('--by-model', action='store_true', help='one group per model, on by default for redress.res')
    parser.add_argument('--figures', default='', help='path to figures.res with the model names, by default next to the source')
    args = parser.parse_args()
    selected = set(g.strip().lower() for g in args.groups.split(',') if g.strip())
    source_name = os.path.splitext(os.path.basename(args.source))[0].lower()
    by_model = args.by_model or source_name == 'redress'
    templates = None
    if by_model and not (args.unpack and not selected):
        figures_path = args.figures or os.path.join(os.path.dirname(args.source), 'figures.res')
        if not os.path.isfile(figures_path):
            print(f'figures.res is not found at {figures_path}, use --figures')
            return 1
        with open(figures_path, 'rb') as figures_file:
            templates = res.figures_template_names(figures_file)
    with open(args.source, 'rb') as textures_file:
        filetree_dict = res.read_res_filetree(textures_file, return_dict=True)
        if args.unpack and not selected:
            return unpack_mmp(textures_file, filetree_dict, list(filetree_dict.keys()), args.destination)
        if by_model:
            groups = res.textures_models(textures_file, templates, filetree_dict)
        else:
            groups = res.textures_groups(textures_file, filetree_dict)
        if args.unpack:
            unknown = selected - set(groups.keys())
            if unknown:
                print(f'unknown groups: {sorted(unknown)}')
                return 1
            names = [n for g in sorted(selected) for n in groups[g]]
            return unpack_mmp(textures_file, filetree_dict, names, args.destination)
        if args.list:
            for g in sorted(groups.keys()):
                size = sum(filetree_dict[n][2] for n in groups[g])
                names = sorted(os.path.splitext(n)[0].lower() for n in groups[g])
                print(f'{g}: {len(names)} textures, {size / 1e6:.1f} MB: {" ".join(names)}')
            return 0
        if selected:
            unknown = selected - set(groups.keys())
            if unknown:
                print(f'unknown groups: {sorted(unknown)}')
                return 1
        if not os.path.isdir(args.destination):
            os.makedirs(args.destination)
        written = 0
        total_size = 0
        sizes = []
        errors = 0
        for g in sorted(groups.keys()):
            if selected and g not in selected:
                continue
            dest_file_path = os.path.join(args.destination, g + '.res')
            res.extract_textures_group(textures_file, filetree_dict, groups[g], dest_file_path)
            # read the new file back and compare every element with the source
            with open(dest_file_path, 'rb') as new_file:
                new_filetree = res.read_res_filetree(new_file, return_dict=True)
                if len(new_filetree) != len(groups[g]):
                    print(f'ERROR: {g} has {len(new_filetree)} elements, expected {len(groups[g])}')
                    errors += 1
                for name in groups[g]:
                    element = new_filetree.get(name.lower())
                    if not element:
                        print(f'ERROR: {name} is missing in {dest_file_path}')
                        errors += 1
                        continue
                    _, src_offset, src_size = filetree_dict[name]
                    textures_file.seek(src_offset)
                    src_data = textures_file.read(src_size)
                    new_file.seek(element[1])
                    if new_file.read(element[2]) != src_data:
                        print(f'ERROR: {name} content differs in {dest_file_path}')
                        errors += 1
            size = os.path.getsize(dest_file_path)
            sizes.append((size, g))
            total_size += size
            written += 1
    if not selected:
        index = {}
        for g, names in groups.items():
            for name in names:
                index[os.path.splitext(name)[0].lower()] = g
        with open(os.path.join(args.destination, source_name + '_index.json'), 'w') as f:
            json.dump(index, f, indent=0, sort_keys=True)
    print(f'written {written} files into {args.destination}, total size {total_size / 1e6:.1f} MB')
    print('largest: ' + ', '.join(f'{g} {s / 1e6:.1f} MB' for s, g in sorted(sizes, reverse=True)[:10]))
    if not by_model and 'objects' in groups:
        print(f'{len(groups["objects"])} textures did not match any group and went to objects.res')
    if errors:
        print(f'{errors} errors found')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
