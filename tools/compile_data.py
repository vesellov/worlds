import os
import sys
import json


def flat_tree(tree, arr=None):
    if arr is None:
        arr = []
    arr.append(tree[0])
    if len(tree[1]) != 0:
        for leaf in tree[1]:
            flat_tree(leaf, arr)
    return arr


def main():
    figures = json.loads(open('figures.json', 'rt').read())
    figures_names = json.loads(open('figures_names.json', 'rt').read())
    textures = json.loads(open('textures.json', 'rt').read())
    race_models = {r['name']:r for r in json.loads(open('db_race_models.json', 'rt').read())}
    armors = {r['name']:r for r in json.loads(open('db_armors.json', 'rt').read())}
    materials = {r['name']:r for r in json.loads(open('db_materials.json', 'rt').read())}
    monster_prototypes = {r['name']:r for r in json.loads(open('db_monster_prototypes.json', 'rt').read())}

    result = {}
    for monster_prototype_name in monster_prototypes.keys():
        monster_prototype = monster_prototypes[monster_prototype_name]
        skin_index = monster_prototype['skin index']
        hair = monster_prototype["hair"]
        base_race = race_models[monster_prototype['base race']]
        base_race_model_name = base_race['mask name']
        base_race_textures = base_race['textures']
        if base_race_textures and isinstance(base_race_textures, list):
            base_race_textures = ':'.join(base_race_textures)
        base_race_textures = base_race_textures.lower()
        base_race_textures_list = base_race_textures.split(':')
        base_race_textures2 = base_race['textures2']
        if base_race_textures2 and isinstance(base_race_textures2, list):
            base_race_textures2 = ':'.join(base_race_textures2)
        base_race_textures2 = base_race_textures2.lower()
        textures = {}
        if base_race_model_name in ['unhufe', 'unhuma', 'unorfe', 'unorma']:
            textures['*'] = f'{base_race_model_name}skin_{skin_index:02}:0'
        else:
            textures['*'] = f'{base_race_textures_list[0]}:0'
        wears = monster_prototype['wears'] or []
        if not isinstance(wears, list):
            wears = [wears, ]
        parts_ordered_tree = figures[base_race_model_name].copy()
        parts_list_flat = flat_tree(parts_ordered_tree)
        parts_list_minimal = []
        for part_name in parts_list_flat:
            if part_name.count('.') == 1:
                continue
            ignore_prefix_found = False
            for ignore_prefix in ['r_shell', 'l_shell', 'bwpart', 'bwtetiva', 'baserh', 'basearrow', 'basepike', 'baseaxe',
                                  'baseclub', 'crbow', 'basesword', 'basedagger', 'basesword', 'quiver', 'arrows']:
                if part_name.startswith(ignore_prefix):
                    ignore_prefix_found = True
                    break
            if ignore_prefix_found:
                continue
            if part_name not in parts_list_minimal:
                parts_list_minimal.append(part_name)
        if not parts_list_minimal:
            parts_list_minimal = parts_list_flat.copy()
        is_human = base_race_model_name in ['unhuma', 'unhufe']
        has_helm = False
        has_plate = False
        has_pants = False
        has_shirt = False
        for wear in wears:
            armor_name, material_name = wear.split('.')
            material_name = material_name.strip()
            armor = armors[armor_name]
            armor_type = armor['type']
            if armor_type == 'helm':
                has_helm = True
            elif armor_type == 'plate':
                has_plate = True
            elif armor_type == 'pants':
                has_pants = True
            elif armor_type == 'shirt':
                has_shirt = True
            armor_code = dict(
                plate='pl', 
                gloves='gl',
                leggings='lg',
                boots='bt',
                shirt='sh',
                helm='hl',
                pants='pt',
            ).get(armor_type)
            texture_type_1 = armor['texture type 1']
            texture_type_2 = armor['texture type 2']
            material = materials[material_name]
            material_code = material['code']
            body_parts = dict(
                plate='bd.a,rh1.a,lh1.a',
                gloves='lh3,rh3',
                leggings='hp,rl1,rl2,ll1,ll2',
                boots='ll3,rl3',
                shirt='bd,rh1,rh2,lh1,lh2',
                helm='hd.a',
                pants='hp,rl1,rl2,ll1,ll2',
            ).get(armor_type).split(',')
            for body_part in body_parts:
                body_part = body_part.replace('.a', f'.armor{texture_type_1:02}' if texture_type_1 else '')
                body_part_texture = f"{base_race_model_name}{armor_code}_{texture_type_1:02}.{material_code}.{texture_type_2}"
                if body_part.count('.armor'):
                    if armor_type == 'helm':
                        body_part_texture = f'{body_part_texture}:3'
                    else:
                        body_part_texture = f'{body_part_texture}:1'
                elif body_part in parts_list_minimal:
                    body_part_texture = f'{body_part_texture}:0'
                else:
                    body_part_texture = f'{body_part_texture}:2'
                textures[body_part] = body_part_texture
            for body_part in body_parts:
                body_part = body_part.replace('.a', f'.armor{texture_type_1:02}' if texture_type_1 else '')
                if body_part not in parts_list_minimal:
                    parts_list_minimal.append(body_part)
        if is_human:
            if not has_helm and hair >= 0:
                parts_list_minimal.append(f'hr.{hair:02}')
            if has_plate:
                parts_list_minimal[parts_list_minimal.index('bd')] = 'bd.hidden'
        # if has_pants:
        #     parts_list_minimal[parts_list_minimal.index('hp')] = 'hp.hidden'
        #     parts_list_minimal[parts_list_minimal.index('rl1')] = 'rl1.hidden'
        #     parts_list_minimal[parts_list_minimal.index('rl2')] = 'rl2.hidden'
        #     parts_list_minimal[parts_list_minimal.index('ll1')] = 'll1.hidden'
        #     parts_list_minimal[parts_list_minimal.index('ll2')] = 'll2.hidden'
        result[monster_prototype_name] = dict(
            model=base_race_model_name,
            parts=parts_list_minimal,
            textures=textures,
        )

    open('compiled.json', 'wt').write(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
