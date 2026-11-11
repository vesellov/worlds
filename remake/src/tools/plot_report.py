"""
Builds the HTML page of the scenarios of Evil Islands and its sequel from the JSON of tools.scenario
and the PNG portraits of tools.portraits, with the template plot_report.html next to this file.

    PYTHONPATH=src venv/bin/python -m tools.plot_report <dir with ei1.json, ei2.json> <portraits ei1> <portraits ei2> report.html
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import io
import os
import re
import sys
import json
import base64
import collections

from PIL import Image

#------------------------------------------------------------------------------

PORTRAIT_SIZE = 168
FULL_SIZE = 220

SIDE_PREFIXES = ('fq', 'qk', 'qh')

GAMES = {
    'ei1': {
        'title': 'Проклятые Земли',
        'subtitle': 'Evil Islands: Curse of the Lost Soul, 2000',
        'intro': (
            'Зак приходит в себя среди Развалин на аллоде Гипат и ничего не помнит. Жители единственного '
            'людского Поселка встречают его как Избранного из старой легенды, и с этого начинается цепочка '
            'поручений старосты, воеводы и мастера. Постепенно выясняется, что кто-то в Поселке хочет смерти '
            'Избранного, а нити тянутся к оркам, разбойникам, хадаганцам и в конце концов к Проклятию, '
            'которое держит аллоды. Сюжет проходит через три аллода и возвращается на Гипат.'
        ),
        'chapters': [
            {'allod': 'gipat', 'title': 'Гипат: Избранный поневоле', 'text': (
                'Старейшина Эрфар-Краснобай, воевода Горт-Костолом и мастер Бабур-Скряга дают первые задания: '
                'убить гоблина-дозорного и Белого волка, вернуть инструменты, украденные «вольными охотниками», '
                'наказать гоблинов и обыскать Рыжего разбойника, который тайно встречается с орками. Дорога через '
                'ловушки ведет к знахарке Эстере-Вредной и ее внучке Айри. Знахарка ищет историю Синего Камня и '
                'Магического Шара: Зак крадет идола гоблинов, добывает шестеренку для орковского моста, грамоту и '
                'летопись островных орков, ракушку мира ящеров, драконий амулет и зеленый кристалл, исследует '
                'Обсерваторию и Мертвый Город. Главная улика раскрывает заговор: воевода Горт поднимает мятеж, '
                'а проверка засадой показывает, кто из старосты и мастера желает Избранному смерти. Айри похищают '
                'гоблины и передают вождю орков Шай-Вару, который готов отдать ее Демону; Зак побеждает Демона. '
                'Хадаганская экспедиция и Ящер-Отшельник открывают путь через тоннель к жилищу Великого Мага.'
            )},
            {'allod': 'ingos', 'title': 'Ингос: похищенная Принцесса', 'text': (
                'Великий Маг отправляет Зака на Ингос искать девушку. В окрестностях города он работает на '
                'охотника Свера-Зайчатника, ищет тело пропавшего сыщика, меняет Священный Цветок на рассказ '
                'вождя Зеленого клана лесовиков Серого Листа и освобождает их Почетного Старейшину. Торговец '
                'Карансул и губернатор Вальрасиан торгуют сведениями за бухгалтерскую книгу, коллекцию губернатора '
                'и золотой слиток. След приводит к магессе Диэре, которая спрятала Принцессу Лу-а-Джаллу в '
                'Запретном подземелье.'
            )},
            {'allod': 'suslanger', 'title': 'Суслангер: Каньон Смерти и беглые', 'text': (
                'На Суслангере Зак попадает в число узников, на которых хадаганцы тренируются в Каньоне Смерти. '
                'Наложница помогает украсть ключ у Кардаура, побег идет через Пещеру Червелицых в Пустыню '
                'некромантов. Беглые из Последнего Укрытия (Брат Хатуль, Брат Кузнец, воровка Сахиля) проверяют '
                'его заданиями: Императорская Печать, месть надсмотрщикам, тайник коллекционера. Старый некромант '
                'Холай помогает узнать, куда телепортировался юноша-джун Ат-Зако. Через Портал Зак возвращается '
                'на Гипат.'
            )},
            {'allod': 'gipat', 'quests': ['q73g'], 'title': 'Финал: Проклятие', 'text': (
                'Великий Маг дает последнее задание: в Подземелье уничтожить Проклятие.'
            )},
        ],
    },
    'ei2': {
        'title': 'Затерянные в Астрале',
        'subtitle': 'Evil Islands 2: Lost in Astral, 2006',
        'intro': (
            'Герой продолжения — краснокожий юноша-джун Кир, раб с аллода Джигран, где правит Фирез. Вместе с '
            'Кель он бежит из хадаганского рабства на Суслангере, затем через Гипат и Ингос ищет дорогу домой '
            'и в конце возвращается на Джигран, чтобы вместе с канийцами свергнуть Фиреза. На Гипате '
            'появляются знакомые по первой игре персонажи: Бабур, Эстера, Айри, Великий Маг.'
        ),
        'chapters': [
            {'allod': 'jigran', 'quests': ['q1r'], 'title': 'Пролог: Новая земля', 'text': (
                'Короткое вступление: единственный шанс спастись — покинуть эти земли через тайный лагерь.'
            )},
            {'allod': 'suslanger', 'title': 'Суслангер: побег из рабства', 'text': (
                'Во время нападения Ужаса Кир сбегает из загона для рабов, проходит катакомбы и вместе с Кель '
                'находит приют у ученого Риндейла, собирателя гигантских муравьев. За покровительство приходится '
                'платить: выкрасть списки рабов, достать маскировку, раздобыть метеоритное «небесное железо». '
                'Тетя Зара просит разобраться с шантажистами, вор Фейруз ведет на Черный рынок, а его невесту '
                'Алию нужно вызволить из тюрьмы. После мести убийцам Фейруза Риндейл отказывает в покровительстве, '
                'и герои уходят в Последнее Укрытие к Брату Хатулю. Там их ждут Сахиля, червелицый Мозгодер с '
                'Древним договором и, наконец, вербовка в хадаганскую армию: Северная крепость и Каньон Смерти.'
            )},
            {'allod': 'gipat2', 'title': 'Гипат: по следам Избранного', 'text': (
                'Хадаганцы отправляют новобранцев на Гипат к экспедиции. Кир и Кель подделывают донесение и '
                'образцы пород, сближаются с орками вождя Урк-Тара и шамана Тыначея, а Ящер-Отшельник за мир '
                'с людьми рассказывает, что Избранный ушел к жилищу Великого Мага. В Предгорьях их принимают '
                'разбойники Бора-Хитреца, мастер Бабур делает вещи и ключи, знахарка Эстера с внучкой Айри '
                'помогают найти и починить Магический Шар. Он нужен Великому Магу Тка-Рику, чтобы отыскать '
                'запечатанный Джигран; путь к нему лежит через тоннель с големами.'
            )},
            {'allod': 'ingos', 'title': 'Ингос: артефакт и гибель Бриссена', 'text': (
                'Тка-Рик чувствует на Ингосе магический артефакт. В городе нужно найти мага Хиса, через '
                'Свера-Зайчатника и Серого Листа выяснить, как погиб рыцарь Бриссен, поработать на Карансула, '
                'губернатора Вальрасиана и разбойников Геральдины и Ловкача. Старый маг Ям-Акара ведет к тайным '
                'воротам за Снежной пустошью, где артефакт оказывается старым Порталом.'
            )},
            {'allod': 'jigran', 'title': 'Джигран: падение Фиреза', 'text': (
                'Портал приводит на Джигран. Канийский город Дар-Леанос осаждают мертвецы; барон Виррил, мастер '
                'Старк и дядюшка Рад посылают Кира к старому некроманту Джаминину и за зельем, упокоивающим '
                'кладбища. Затем — штурм хадаганской крепости Дар-Улгорн с помощью отравленного рагу, переправа '
                'Двух рек, доспехи джунов от мудреца Туи-Лена, освобождение семьи и Тай-Илары из плена, союз с '
                'дикими существами, штурм Цитадели и бой с Фирезом в Тронном зале.'
            )},
        ],
    },
}

#------------------------------------------------------------------------------


def image_uri(file_path, size):
    if not os.path.isfile(file_path):
        return ''
    image = Image.open(file_path).convert('RGBA').resize((size, size), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, 'WEBP', quality=82, method=6)
    return 'data:image/webp;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def file_base_name(character_id):
    return re.sub(r'[^0-9a-z_-]+', '_', character_id.lower()).strip('_')


def mention_keys(name):
    """
    The words of a Russian name which find it in the texts: "Эрфар-Краснобай" -> "Эрфар", "Краснобай".
    """
    words = [w for w in re.split(r'[\s\-,.«»"()]+', name) if len(w) >= 4 and w[0].isupper()]
    skip = {'Вождь', 'Брат', 'Мастер', 'Старый', 'Великий', 'Барон', 'Рыцарь', 'Дядюшка'}
    return [w for w in words if w not in skip]


def chapter_of(quest, chapters, zones):
    for index, chapter in enumerate(chapters):
        if quest['id'] in chapter.get('quests', ()):
            return index
    allod = zones.get(quest['zone'], {}).get('allod', '')
    for index, chapter in enumerate(chapters):
        if chapter['allod'] == allod and 'quests' not in chapter:
            return index
    return None


def build_game(key, scenario, portraits_dir_path):
    meta = GAMES[key]
    zones = {z['id']: z for z in scenario['zones']}
    briefings = {b['id']: b for b in scenario['briefings']}
    named_prototypes = set((c.get('unit') or {}).get('prototype') for c in scenario['characters'] if c['name'])
    characters = [c for c in scenario['characters'] if c['name'] or (
        c.get('unit') and c['unit']['prototype'] not in named_prototypes and (c['lines'] or c['owns']))]
    by_id = {c['id']: c for c in characters}
    persons = {k.lower(): v for k, v in scenario['persons'].items()}
    quest_titles = {q['id']: q['title'] for q in scenario['quests']}
    # givers of the quests: owners of the briefings
    givers = collections.defaultdict(list)
    for c in characters:
        for q in c['quests_given']:
            givers[q].append(c['id'])
    targets = collections.defaultdict(list)
    for c in characters:
        for q in c['quests_target']:
            targets[q].append(c['id'])
    # relations: appearing in one briefing and naming each other in the phrases
    keys = {c['id']: mention_keys(c['name']) for c in characters if c['kind'] != 'hero' and c['name']}
    weights = collections.Counter()
    for b in scenario['briefings']:
        present = set(o for o in b['owners'] if o in by_id)
        present |= set(s.lower() for s in b['speakers'] + b['cast'] if s.lower() in by_id)
        present = sorted(p for p in present if by_id[p]['kind'] != 'hero')
        for i, a in enumerate(present):
            for other in present[i + 1:]:
                weights[(a, other)] += 2
        for speaker, text, _desc in b['phrases']:
            speaker = speaker.lower()
            if speaker not in by_id:
                continue
            for other, words in keys.items():
                if other == speaker or by_id[speaker]['kind'] == 'hero':
                    continue
                if any(re.search(r'\b%s' % re.escape(w), text) for w in words):
                    weights[tuple(sorted((speaker, other)))] += 1
    # quest giver -> target of the quest
    for q, ids in targets.items():
        for giver in givers.get(q, ()):
            for target in ids:
                if giver != target:
                    weights[tuple(sorted((giver, target)))] += 1
    links = [{'a': a, 'b': b, 'w': w} for (a, b), w in weights.items() if w >= 1]
    # chapters
    chapters = meta['chapters']
    quests_out = []
    last_index = 0
    for q in scenario['quests']:
        if not q['title']:
            continue
        index = chapter_of(q, chapters, zones)
        if index is None:
            # a zone missing in map.txt: the chapter of the quest before it
            index = last_index
        last_index = index
        zone = zones.get(q['zone'], {})
        quests_out.append({
            'id': q['id'], 'n': q['index'], 'ch': index, 'title': q['title'], 'text': q['description'],
            'exp': q['experience'], 'zone': zone.get('title', q['zone']), 'zid': q['zone'],
            'side': q['id'].startswith(SIDE_PREFIXES) or not q['given_by'],
            'by': givers.get(q['id'], []), 'targets': targets.get(q['id'], []),
            'steps': [s[1] for s in q['subobjectives']],
            'briefs': [{'id': b, 'title': briefings[b]['title']} for b in q['given_by'] if b in briefings],
        })
    chars_out = []
    for c in characters:
        unit = c.get('unit') or {}
        base = file_base_name(c['id'])
        proto = c.get('prototype') or {}
        stats = {k: proto.get(k) for k in ('health', 'mana', 'attack', 'parry', 'damage', 'armor')} if proto else {}
        rel = collections.Counter()
        for link in links:
            if link['a'] == c['id']:
                rel[link['b']] += link['w']
            elif link['b'] == c['id']:
                rel[link['a']] += link['w']
        chars_out.append({
            'id': c['id'], 'name': c['name'] or c['id'], 'kind': c['kind'],
            'proto': unit.get('prototype', ''), 'protoName': c.get('prototype_name', ''),
            'model': unit.get('model', ''), 'found': c.get('unit_found_by', ''),
            'zones': [zones[m]['title'] if m in zones else m for m in c['brief_zones']][:4],
            'maps': c['maps'][:6],
            'armors': unit.get('armors', []), 'weapons': unit.get('weapons', []) or proto.get('weapons', []),
            'spells': unit.get('spells', []),
            'stats': stats, 'mapStats': unit.get('stats', {}),
            'owns': len(c['owns']), 'lines': c['lines'], 'speaks': len(c['speaks_in']),
            'given': [{'id': q, 'title': quest_titles.get(q, q)} for q in c['quests_given']],
            'target': [{'id': q, 'title': quest_titles.get(q, q)} for q in c['quests_target']],
            'briefs': [{'id': b, 'title': briefings[b]['title']} for b in c['owns'] if b in briefings][:40],
            'money': c['money'],
            'rel': [{'id': k, 'w': w} for k, w in rel.most_common(8)],
            'img': image_uri(os.path.join(portraits_dir_path, base + '.png'), PORTRAIT_SIZE),
            'full': image_uri(os.path.join(portraits_dir_path, base + '_full.png'), FULL_SIZE),
        })
    stats = {
        'quests': len(scenario['quests']), 'briefings': len(scenario['briefings']),
        'characters': len([c for c in chars_out if c['kind'] != 'target']),
        'targets': len([c for c in chars_out if c['kind'] == 'target']),
        'zones': len([z for z in scenario['zones'] if z['type'] == 'game']),
        'portraits': len([c for c in chars_out if c['img']]),
    }
    return {
        'title': meta['title'], 'subtitle': meta['subtitle'], 'intro': meta['intro'],
        'chapters': [{'title': ch['title'], 'text': ch['text']} for ch in chapters],
        'quests': quests_out, 'characters': chars_out, 'links': links, 'stats': stats,
    }


def main():
    out_dir_path, portraits1_dir_path, portraits2_dir_path, report_file_path = sys.argv[1:5]
    data = {}
    for key, portraits_dir_path in (('ei1', portraits1_dir_path), ('ei2', portraits2_dir_path)):
        with open(os.path.join(out_dir_path, key + '.json'), encoding='utf-8') as f:
            data[key] = build_game(key, json.load(f), portraits_dir_path)
    template_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_report.html')
    with open(template_file_path, encoding='utf-8') as f:
        template = f.read()
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write(template.replace('/*DATA*/null', payload))
    if _Debug:
        print('%s: %d bytes' % (report_file_path, os.path.getsize(report_file_path)))


if __name__ == '__main__':
    main()
