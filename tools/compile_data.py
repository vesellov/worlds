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
    weapons = {r['name']:r for r in json.loads(open('db_weapons.json', 'rt').read())}
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
        weapon_kind = monster_prototype['weapon']
        parts_ordered_tree = figures[base_race_model_name].copy()
        parts_list_flat = flat_tree(parts_ordered_tree)
        parts = []
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
            if part_name not in parts:
                parts.append(part_name)
        if not parts:
            parts = parts_list_flat.copy()
        is_human = base_race_model_name in ['unhuma', 'unhufe']
        has_helm = False
        has_plate = False
        has_pants = False
        has_shirt = False
        has_boots = False
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
            elif armor_type == 'boots':
                has_boots = True
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
            armor_id = int(texture_type_1)
            material = materials[material_name]
            material_code = material['code']
            body_parts = dict(
                plate='bd.a,rh1.a,rh2.a,lh1.a,lh2.a',
                gloves='lh3,rh3',
                leggings='hp.a,rl1.a,rl2.a,rl3.a,ll1.a,ll2.a,ll3.a',
                boots='ll3,rl3',
                shirt='bd,rh1,rh2,rh3,lh1,lh2,lh3',
                helm='hd.a',
                pants='hp,rl1,rl2,ll1,ll2',
            ).get(armor_type).split(',')
            if armor_type == 'pants' and armor_id in [1, 2, 3, 6]:
                body_parts.append('l_shell.a')
                body_parts.append('r_shell.a')
            for body_part in body_parts:
                body_part = body_part.replace('.a', f'.armor{armor_id:02}' if armor_id else '')
                body_part_texture = f"{base_race_model_name}{armor_code}_{texture_type_1:02}.{material_code}.{texture_type_2}"
                if body_part.count('.armor'):
                    if armor_type in ['helm', ]:
                        body_part_texture = f'{body_part_texture}:3'
                    elif armor_type in ['boots', 'gloves', ]:
                        body_part_texture = f'{body_part_texture}:0'
                    elif armor_type in ['plate', 'pants']:
                        body_part_texture = f'{body_part_texture}:0'
                    else:
                        body_part_texture = f'{body_part_texture}:0'  # 0 maybe?
                elif body_part in parts:
                    body_part_texture = f'{body_part_texture}:0'
                else:
                    body_part_texture = f'{body_part_texture}:2'
                textures[body_part] = body_part_texture
            for body_part in body_parts:
                armor_id = texture_type_1
                body_part = body_part.replace('.a', f'.armor{armor_id:02}' if armor_id else '')
                if body_part not in parts:
                    parts.append(body_part)
        if is_human:
            if not has_helm and hair >= 0:
                parts.append(f'hr.{hair:02}')
            # if has_boots:
            #     parts[parts.index('ll3')] = 'll3.hidden'
            #     parts[parts.index('rl3')] = 'rl3.hidden'
            # if has_plate:
            #     parts[parts.index('bd')] = 'bd.hidden'
            #     parts[parts.index('rh1')] = 'rh1.hidden'
            #     parts[parts.index('lh1')] = 'lh1.hidden'
            #     parts[parts.index('rh2')] = 'rh1.hidden'
            #     parts[parts.index('lh2')] = 'lh1.hidden'
            # if has_pants:
            #     parts[parts.index('hp')] = 'hp.hidden'
            #     parts[parts.index('rl1')] = 'rl1.hidden'
            #     parts[parts.index('rl2')] = 'rl2.hidden'
            #     parts[parts.index('ll1')] = 'll1.hidden'
            #     parts[parts.index('ll2')] = 'll2.hidden'
        # if has_pants:
        #     parts[parts.index('hp')] = 'hp.hidden'
        #     parts[parts.index('rl1')] = 'rl1.hidden'
        #     parts[parts.index('rl2')] = 'rl2.hidden'
        #     parts[parts.index('ll1')] = 'll1.hidden'
        #     parts[parts.index('ll2')] = 'll2.hidden'
        if weapon_kind:
            weapon_name, material_name = weapon_kind.split('.')
            if material_name.count(' ['):
                material_name = material_name.split(' ')[0].strip()
            weapon = weapons[weapon_name]
            weapon_type = weapon['type']
            material = materials[material_name]
            material_code = material['code']
            texture_type_1 = weapon['texture type 1']
            texture_type_2 = weapon['texture type 2']
            weapon_id = int(texture_type_1)
            weapon_code = dict(
                hammer='hm',
                dagger='dg',
                spear='sp',
                crossbow='cb',
                sword='sw',
                axe='ax',
                bow='bw',
            ).get(weapon_type)
            body_parts = ''
            weapon_texture = f"{base_race_model_name}{weapon_code}_{texture_type_1:02}.{material_code}.{texture_type_2}:3"
            if weapon_type == 'bow':
                if weapon_id == 0:
                    parts.append('lh3.bwpartb00')
                    parts.append('bwparta00')
                    parts.append('bwtetivaa00')
                    parts.append('bwtetivab00')
                    textures['lh3.bwpartb00'] = weapon_texture
                    textures['bwparta00'] = weapon_texture
                    textures['bwtetivaa00'] = weapon_texture
                    textures['bwtetivab00'] = weapon_texture
                else:
                    parts.append('lh3.bwpartb00.hidden')
                    parts.append(f'lh3.bwpartb{weapon_id:02}')
                    textures[f'lh3.bwpartb{weapon_id:02}'] = weapon_texture
                    parts.append('bwparta00.hidden')
                    parts.append(f'bwparta{weapon_id:02}')
                    textures[f'bwparta{weapon_id:02}'] = weapon_texture
                    parts.append('bwtetivaa00.hidden')
                    parts.append(f'bwtetivaa{weapon_id:02}')
                    textures[f'bwtetivaa{weapon_id:02}'] = weapon_texture
                    parts.append('bwtetivab00.hidden')
                    parts.append(f'bwtetivab{weapon_id:02}')
                    textures[f'bwtetivab{weapon_id:02}'] = weapon_texture
                parts.append('rh3.arrow00')
                textures['rh3.arrow00'] = weapon_texture
                # parts.append('basearrow00')
                # textures['basearrow00'] = weapon_texture
                parts.append('quiver')
                textures['quiver'] = weapon_texture
                parts.append('arrows')
                textures['arrows'] = weapon_texture
            elif weapon_type == 'crossbow':
                if weapon_id == 1:
                    parts.append('rh3.crbow01main')
                    parts.append('crbow01part01')
                    parts.append('crbow01tetiva01')
                    parts.append('crbow01part02')
                    parts.append('crbow01tetiva02')
                    textures['rh3.crbow01main'] = weapon_texture
                    textures['crbow01part01'] = weapon_texture
                    textures['crbow01tetiva01'] = weapon_texture
                    textures['crbow01part02'] = weapon_texture
                    textures['crbow01tetiva02'] = weapon_texture
                else:
                    parts.append('rh3.crbow01main.hidden')
                    parts.append(f'rh3.crbow{weapon_id:02}main')
                    textures[f'rh3.crbow{weapon_id:02}main'] = weapon_texture
                    parts.append('crbow01part01.hidden')
                    parts.append(f'crbow{weapon_id:02}part01')
                    textures[f'crbow{weapon_id:02}part01'] = weapon_texture
                    parts.append('crbow01tetiva01.hidden')
                    parts.append(f'crbow{weapon_id:02}tetiva01')
                    textures[f'crbow{weapon_id:02}tetiva01'] = weapon_texture
                    parts.append('crbow01part02.hidden')
                    parts.append(f'crbow{weapon_id:02}part02')
                    textures[f'crbow{weapon_id:02}part02'] = weapon_texture
                    parts.append('crbow01tetiva02.hidden')
                    parts.append(f'crbow{weapon_id:02}tetiva02')
                    textures[f'crbow{weapon_id:02}tetiva02'] = weapon_texture
            elif weapon_type == 'spear':
                if weapon_id == 0:
                    parts.append('rh3.pike00')
                    parts.append('basepike00')
                    textures['rh3.pike00'] = weapon_texture
                    textures['basepike00'] = weapon_texture
                else:
                    parts.append('rh3.pike00.hidden')
                    parts.append(f'rh3.pike{weapon_id:02}')
                    parts.append(f'basepike{weapon_id:02}')
                    textures[f'rh3.pike{weapon_id:02}'] = weapon_texture
                    textures[f'basepike{weapon_id:02}'] = weapon_texture
            elif weapon_type == 'sword':
                if weapon_id == 0:
                    parts.append('rh3.sword00')
                    parts.append('basesword00')
                    textures['rh3.sword00'] = weapon_texture
                    textures['basesword00'] = weapon_texture
                else:
                    weapon_id = weapon_id % 5
                    parts.append('rh3.sword00.hidden')
                    parts.append(f'rh3.sword{weapon_id:02}')
                    parts.append(f'basesword{weapon_id:02}')
                    textures[f'rh3.sword{weapon_id:02}'] = weapon_texture
                    textures[f'basesword{weapon_id:02}'] = weapon_texture
            elif weapon_type == 'dagger':
                if weapon_id == 0:
                    parts.append('rh3.sword00')
                    parts.append('basesword00')
                    textures['rh3.sword00'] = weapon_texture
                    textures['basesword00'] = weapon_texture
                else:
                    parts.append('rh3.sword00.hidden')
                    parts.append(f'rh3.dagger{weapon_id:02}')
                    parts.append(f'basedagger{weapon_id:02}')
                    textures[f'rh3.dagger{weapon_id:02}'] = weapon_texture
                    textures[f'basedagger{weapon_id:02}'] = weapon_texture
            elif weapon_type == 'axe':
                if weapon_id == 0:
                    parts.append('rh3.axe00')
                    parts.append('baseaxe00')
                    textures['rh3.axe00'] = weapon_texture
                    textures['baseaxe00'] = weapon_texture
                else:
                    parts.append('rh3.axe00.hidden')
                    parts.append(f'rh3.axe{weapon_id:02}')
                    parts.append(f'baseaxe{weapon_id:02}')
                    textures[f'rh3.axe{weapon_id:02}'] = weapon_texture
                    textures[f'baseaxe{weapon_id:02}'] = weapon_texture
            elif weapon_type == 'hammer':
                if weapon_id == 0:
                    parts.append('rh3.axe00')
                    parts.append('baseaxe00')
                    textures['rh3.axe00'] = weapon_texture
                    textures['baseaxe00'] = weapon_texture
                else:
                    parts.append('rh3.axe00.hidden')
                    parts.append(f'rh3.club{weapon_id:02}')
                    parts.append(f'baseclub{weapon_id:02}')
                    textures[f'rh3.club{weapon_id:02}'] = weapon_texture
                    textures[f'baseclub{weapon_id:02}'] = weapon_texture
        result[monster_prototype_name] = dict(
            model=base_race_model_name,
            parts=parts,
            textures=textures,
        )

    open('compiled.json', 'wt').write(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
