"""
Look of the figures: which parts and textures the armors and the weapons give them.
Shared by the client, which builds the models, and the server, which chooses the look of new units.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

# armor type -> code of the armor in the names of its textures: <model><code>_<texture_type_1>.<material code>.<texture_type_2>
ARMOR_TYPE_CODES = {
    'plate': 'pl',
    'gloves': 'gl',
    'leggings': 'lg',
    'boots': 'bt',
    'shirt': 'sh',
    'helm': 'hl',
    'pants': 'pt',
}

# armor type -> parts of the figure it covers, ".a" is replaced with ".armor<texture_type_1>" when the armor has its own parts
ARMOR_BODY_PARTS = {
    'plate': ('bd.a', 'rh1.a', 'rh2.a', 'lh1.a', 'lh2.a'),
    'gloves': ('lh3', 'rh3'),
    'leggings': ('hp.a', 'rl1.a', 'rl2.a', 'rl3.a', 'll1.a', 'll2.a', 'll3.a'),
    'boots': ('ll3', 'rl3'),
    'shirt': ('bd', 'rh1', 'rh2', 'rh3', 'lh1', 'lh2', 'lh3'),
    'helm': ('hd.a', ),
    'pants': ('hp', 'rl1', 'rl2', 'll1', 'll2'),
}

# pants of these texture_type_1 also have the shells on the hips
ARMOR_PANTS_WITH_SHELLS = (1, 2, 3, 6)

# weapon type -> code of the weapon in the names of its textures: <model><code>_<texture_type_1>.<material code>.<texture_type_2>
WEAPON_TYPE_CODES = {
    'hammer': 'hm',
    'dagger': 'dg',
    'spear': 'sp',
    'crossbow': 'cb',
    'sword': 'sw',
    'axe': 'ax',
    'bow': 'bw',
}

#------------------------------------------------------------------------------


def armor_body_parts(armor_type, armor_id):
    """
    Parts of the figure which the armor covers, see ARMOR_BODY_PARTS.
    """
    body_parts = list(ARMOR_BODY_PARTS[armor_type])
    if armor_type == 'pants' and armor_id in ARMOR_PANTS_WITH_SHELLS:
        body_parts.append('l_shell.a')
        body_parts.append('r_shell.a')
    return [body_part.replace('.a', f'.armor{armor_id:02}' if armor_id else '') for body_part in body_parts]


def armor_texture_name(model_name, armor_type, armor, material):
    """
    Name of the texture of the armor made of the material for that model, without the layer number.
    """
    return f"{model_name}{ARMOR_TYPE_CODES[armor_type]}_{armor['texture_type_1']:02}.{material['code']}.{armor['texture_type_2']}"


def weapon_parts(model_name, weapon, weapons, materials):
    """
    Parts of the figure the weapon "<weapon name>.<material name>" adds and their textures, weapons and materials
    are the catalog. Parts with ".hidden" at the end are the parts of the figure the weapon hides.
    Returns (weapon type, parts, textures).
    """
    parts = []
    textures = {}
    weapon_name, material_name = weapon.split('.')
    if material_name.count(' ['):
        material_name = material_name.split(' ')[0].strip()
    weapon_info = weapons[weapon_name]
    weapon_type = weapon_info['type']
    material = materials[material_name]
    material_code = material['code']
    texture_type_1 = weapon_info['texture_type_1']
    texture_type_2 = weapon_info['texture_type_2']
    weapon_id = int(texture_type_1)
    weapon_code = WEAPON_TYPE_CODES.get(weapon_type)
    weapon_texture = f"{model_name}{weapon_code}_{texture_type_1:02}.{material_code}.{texture_type_2}:1"
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
    return weapon_type, parts, textures
