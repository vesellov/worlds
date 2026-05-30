import os
import sys
import json


def main():
    armors = {r['name']: {
        'type': r['type'],
        'material_type': r['material type'].lower(),
        'texture_type_1': r['texture type 1'],
        'texture_type_2': r['texture type 2'],
    } for r in json.loads(open('items_armors.json', 'rt').read()) if not r['name'].count('prototype')}
    open('armors.json', 'wt').write(json.dumps(armors, indent=2))

    weapons = {r['name']: {
        'type': r['type'],
        'material_type': r['material type'].lower(),
        'texture_type_1': r['texture type 1'],
        'texture_type_2': r['texture type 2'],
        'range': r['range'],
        'min_damage': r['min damage'],
        'max_damage': r['max damage'],
        'attack': r['attack'],
        'defence': r['defence'],
        'actions': r['actions'],
    } for r in json.loads(open('items_weapons.json', 'rt').read())}
    open('weapons.json', 'wt').write(json.dumps(weapons, indent=2))

    materials = {r['name']: {
        'code': r['code'].lower(),
        'type': r['type'].lower(),
    } for r in json.loads(open('items_materials.json', 'rt').read())}
    open('materials.json', 'wt').write(json.dumps(materials, indent=2))

    animations = {}
    for file_name in os.listdir('.'):
        if not file_name.endswith('.json'):
            continue
        if not file_name.startswith('animations_'):
            continue
        model_name = file_name.replace('animations_', '').replace('.json', '')
        anim_data = json.loads(open(file_name, 'rt').read())
        animations[model_name] = anim_data
    open('animations.json', 'wt').write(json.dumps(animations, indent=2))


if __name__ == "__main__":
    main()
