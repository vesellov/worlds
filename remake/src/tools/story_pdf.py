"""
Builds docs/story.pdf from docs/story.md: a cover, the cast page and the portraits of the key characters
next to the paragraphs where they first matter.

    python -m tools.story_pdf ../docs/story.md <portraits ei1 dir> <portraits ei2 dir> <work dir> ../docs/story.pdf

Needs weasyprint and markdown (not in requirements.txt: pip install weasyprint markdown into a separate venv).
The portraits are the PNG of tools.portraits; Ат-Зако is drawn from the prototype "Jun Male Hero" of EI1.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import re
import sys
import html

import markdown
from PIL import Image
import weasyprint

#------------------------------------------------------------------------------

# (game, id, name, role, how to crop: 'head' keeps the top of a small full figure)
PEOPLE = {
    'zak': ('p1', 'hero', 'Зак', 'Избранный из Развалин Гипата', None),
    'atzako': ('p1', 'junboy', 'Ат-Зако', 'юноша-джун, унесший Проклятие', None),
    'lua': ('p1', 'deva', 'Лу-а-Джалла', 'принцесса джунов', None),
    'rick': ('p1', 'rick', 'Тка-Рик', 'Великий Маг Гипата', 'head'),
    'erfar': ('p1', 'elder', 'Эрфар-Краснобай', 'староста Поселка', None),
    'estera': ('p1', 'witch', 'Эстера-Вредная', 'знахарка', None),
    'airi': ('p1', 'daughter', 'Айри-Упрямая', 'внучка знахарки', None),
    'babur': ('p1', 'smith', 'Бабур-Скряга', 'мастер Поселка', None),
    'gort': ('p1', 'sheriff', 'Горт-Костолом', 'воевода-мятежник', None),
    'holai': ('p1', 'oldnekro', 'Холай', 'опальный некромант', None),
    'kir': ('p2', 'hero', 'Кир-А-Чит-Ла', 'беглый раб-джун', None),
    'kel': ('p2', 'merc2', 'Кель', 'воровка, верная спутница', None),
    'rindail': ('p2', 'elder', 'Риндейл', 'ученый, собиратель муравьев', None),
    'guru': ('p2', 'guru', 'Мозгодер', 'повелитель червелицых', 'head'),
    'urktar': ('p2', 'orcc', 'Урк-Тар', 'вождь орков', None),
    'bor': ('p2', 'karh', 'Бор-Хитрец', 'атаман вольных охотников', None),
    'yamakara': ('p2', 'kmage', 'Ям-Акара', 'старый маг Ингоса', None),
    'baron': ('p2', 'baron', 'Барон Виррил', 'последний оплот Кании', None),
    'tuilen': ('p2', 'junofil', 'Туи-Лен', 'мудрец джунов', None),
    'tailara': ('p2', 'deva', 'Тай-Илара', 'пленница Фиреза', None),
    'firez': ('p2', 'firez', 'Фирез', 'джун-некромант, владыка Джиграна', None),
}

# the paragraph (by its first words) -> the portraits floated next to it
PLACES = [
    ('На Гипате, который', ['zak'], 'right'),
    ('Избранному сразу', ['estera', 'airi'], 'left'),
    ('Главная улика', ['erfar'], 'right'),
    ('Тем временем гоблины', ['rick'], 'left'),
    ('Пленница — джун', ['lua'], 'right'),
    ('Тка-Рик ждал', ['holai'], 'left'),
    ('Избранный исчез', ['babur'], 'right'),
    ('Юный джун Кир', ['kir'], 'left'),
    ('На Суслангере краснокожий', ['kel', 'rindail'], 'right'),
    ('Кир и Кель помогают', ['guru'], 'left'),
    ('Кель уходит с орками', ['urktar'], 'right'),
    ('Все на Гипате помнят', ['bor'], 'left'),
    ('Тот же снежный Ингос', ['yamakara'], 'right'),
    ('Портал выводит', ['baron'], 'left'),
    ('Мудрец джунов Туи-Лен', ['tuilen', 'tailara'], 'right'),
    ('С амулетом Туи-Лена', ['firez'], 'left'),
]

# a wide figure: the transformation of the finale
REVEAL_AT = 'Зак возвращается один'

CAST_ONE = ['zak', 'lua', 'rick', 'erfar', 'estera', 'babur', 'gort', 'holai']
CAST_TWO = ['kir', 'kel', 'rindail', 'urktar', 'bor', 'baron', 'tuilen', 'firez']

CSS = """
@page {
  size: A4; margin: 22mm 21mm 24mm 21mm; background: #f4ecda;
  @bottom-center { content: counter(page); font: 9pt 'Noto Serif'; color: #8a6d45; }
  @top-center { content: string(part); font: italic 8.5pt 'Noto Serif'; color: #9a8160; }
}
@page cover { margin: 0; background: #1d1712; @bottom-center { content: none; } @top-center { content: none; } }
@page bare { @bottom-center { content: none; } @top-center { content: none; } }
html { font-family: 'Noto Serif', serif; font-size: 10.6pt; line-height: 1.5; color: #2b2116; hyphens: auto; }
body { margin: 0; }
p { margin: 0 0 0.55em; text-align: justify; text-indent: 1.3em; orphans: 2; widows: 2; }
h2 + p, h3 + p, .fig + p, p.first { text-indent: 0; }
h2 { page: auto; break-before: page; string-set: part content(text); font-family: 'Noto Serif Display', serif;
     font-weight: 600; font-size: 25pt; line-height: 1.15; color: #4a2f17; text-align: center; margin: 14mm 0 2mm; }
h2::after { content: '❦'; display: block; font-size: 14pt; color: #a6763c; margin-top: 3mm; }
h3 { font-family: 'Noto Serif Display', serif; font-weight: 600; font-size: 14.5pt; color: #6b4320;
     margin: 1.3em 0 0.5em; break-after: avoid; letter-spacing: 0.01em; }
h2 + h3 { margin-top: 0.4em; }
main h2 + p::first-letter, main h3 + p::first-letter, main .fig + p::first-letter { font-family: 'Noto Serif Display', serif;
     font-size: 1.75em; line-height: 0.8; color: #8b4a1c; font-weight: 600; }
main h3 + .fig + p::first-letter, main h3 + .fig + .fig + p::first-letter { font-size: 1.75em; }
.chapter-lead { text-align: center; font-style: italic; color: #7a6040; margin-bottom: 8mm; text-indent: 0; }
hr { border: 0; text-align: center; margin: 1em 0; }
hr::after { content: '✦  ✦  ✦'; color: #a6763c; font-size: 9pt; letter-spacing: 0.3em; }
.fig { width: 41mm; margin: 0.3em 0 0.6em; break-inside: avoid; }
.fig.right { float: right; margin-left: 6mm; }
.fig.left { float: left; margin-right: 6mm; }
.frame { width: 41mm; height: 41mm; border-radius: 4mm; overflow: hidden; border: 0.7mm solid #a6763c;
         box-shadow: inset 0 0 0 0.6mm #f4ecda, inset 0 0 0 0.9mm #a6763c;
         background: radial-gradient(circle at 50% 38%, #fbf3df 0%, #e8d6b0 62%, #c9ab7a 100%); }
.frame img { width: 100%; height: 100%; display: block; }
.cap { text-align: center; margin-top: 1.6mm; line-height: 1.2; }
.cap b { display: block; font-family: 'Noto Serif Display', serif; font-weight: 600; font-size: 10pt; color: #4a2f17; }
.cap i { font-size: 7.8pt; color: #7a6040; }
.reveal { break-inside: avoid; margin: 0.6em 0 1em; text-align: center; }
.reveal table { margin: 0 auto; border-collapse: collapse; }
.reveal td { padding: 0 3mm; vertical-align: middle; }
.reveal .arrow { font-size: 22pt; color: #a6763c; }
.reveal .frame { width: 46mm; height: 46mm; }
blockquote { margin: 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; color: #6b4320; }
table.src { width: 100%; border-collapse: collapse; font-size: 8pt; line-height: 1.25; margin: 0.5em 0 1em; }
table.src th { text-align: left; font-family: 'Noto Serif Display', serif; color: #4a2f17;
               border-bottom: 0.5mm solid #a6763c; padding: 1.5mm 2mm; }
table.src td { border-bottom: 0.2mm solid #d8c49d; padding: 1mm 2mm; vertical-align: top; }
ul { padding-left: 1.2em; margin: 0.3em 0 0.8em; } li { margin-bottom: 0.3em; text-align: justify; }
.appendix p, .appendix li { font-size: 9pt; }
.appendix .note { color: #5a4630; margin-bottom: 0.8em; }

/* cover */
.cover { page: cover; height: 297mm; position: relative; color: #f1e2c2; text-align: center;
         background: radial-gradient(ellipse at 50% 42%, #4a3420 0%, #2a1f16 55%, #140f0b 100%); }
.cover .inner { position: absolute; top: 24mm; left: 18mm; right: 18mm; bottom: 20mm;
                border: 0.6mm solid #b58a4d; outline: 0.25mm solid #7d5b30; outline-offset: 2.2mm; }
.cover .kicker { margin-top: 22mm; font-size: 10pt; letter-spacing: 0.45em; text-transform: uppercase; color: #c9a46a; text-indent: 0; text-align: center; }
.cover h1 { font-family: 'Noto Serif Display', serif; font-weight: 600; font-size: 46pt; line-height: 1.05;
            margin: 8mm 0 3mm; color: #f6e7c4; }
.cover .sub { font-family: 'Noto Serif Display', serif; font-style: italic; font-size: 19pt; color: #d9b77c; text-indent: 0; text-align: center; }
.cover .orn { color: #b58a4d; font-size: 16pt; margin: 6mm 0; text-indent: 0; text-align: center; }
.cover .duo { margin: 6mm auto 0; border-collapse: collapse; }
.cover .duo td { padding: 0 7mm; text-align: center; vertical-align: top; }
.cover .medal { width: 58mm; height: 58mm; border-radius: 50%; overflow: hidden; border: 0.8mm solid #b58a4d;
                box-shadow: 0 0 0 1.6mm #2a1f16, 0 0 0 1.9mm #7d5b30;
                background: radial-gradient(circle at 50% 38%, #f3e2bd 0%, #c8a36a 70%, #8c6a3c 100%); }
.cover .medal img { width: 100%; height: 100%; }
.cover .who { margin-top: 4mm; font-family: 'Noto Serif Display', serif; font-size: 13pt; color: #f1e2c2; }
.cover .who i { display: block; font-family: 'Noto Serif', serif; font-size: 8.5pt; color: #c9a46a; margin-top: 1mm; }
.cover .games { position: absolute; bottom: 14mm; left: 0; right: 0; font-size: 9.5pt; line-height: 1.6; color: #c9a46a; text-align: center; text-indent: 0; }

/* cast page */
.cast { page: bare; break-before: page; }
.cast h2 { break-before: auto; margin-top: 0; }
.cast h4 { font-family: 'Noto Serif Display', serif; font-weight: 600; font-size: 12.5pt; color: #6b4320; text-align: center;
           margin: 3mm 0 2.5mm; letter-spacing: 0.04em; }
.grid { width: 100%; border-collapse: collapse; }
.grid td { width: 25%; text-align: center; vertical-align: top; padding: 0 1mm 2.5mm; }
.grid .frame { width: 29mm; height: 29mm; margin: 0 auto; }
.grid .cap b { font-size: 9.6pt; } .grid .cap i { font-size: 7.4pt; }
.intro { font-size: 9.6pt; color: #5a4630; text-align: center; text-indent: 0; margin: 0 8mm 2mm; }
"""

#------------------------------------------------------------------------------


def prepared_image(src_path, work_dir_path, crop):
    """
    The portrait trimmed to its figure and centered on a square, the top of a small full figure for 'head'.
    """
    image = Image.open(src_path).convert('RGBA')
    box = image.getchannel('A').getbbox()
    if box:
        left, top, right, bottom = box
        if crop == 'head':
            height = bottom - top
            bottom = top + int(height * 0.34)
            middle = (left + right) // 2
            half = (bottom - top) // 2
            left, right = middle - half, middle + half
        image = image.crop((left, top, right, bottom))
    side = max(image.size)
    # the busts keep their shoulders on the bottom edge, the rest is centered
    canvas = Image.new('RGBA', (side, side), (0, 0, 0, 0))
    x = (side - image.size[0]) // 2
    y = side - image.size[1] if crop is None else (side - image.size[1]) // 2
    canvas.paste(image, (x, y))
    canvas = canvas.resize((420, 420), Image.LANCZOS)
    out_path = os.path.join(work_dir_path, os.path.basename(os.path.dirname(src_path)) + '_' + os.path.basename(src_path))
    canvas.save(out_path, optimize=True)
    return out_path


def figure_html(key, images, side=None, frame_class='frame'):
    _game, _id, name, role, _crop = PEOPLE[key]
    inner = '<div class="%s"><img src="file://%s"></div><div class="cap"><b>%s</b><i>%s</i></div>' % (
        frame_class, images[key], html.escape(name), html.escape(role))
    if side is None:
        return inner
    return '<div class="fig %s">%s</div>' % (side, inner)


def main():
    story_file_path, p1_dir_path, p2_dir_path, work_dir_path, pdf_file_path = sys.argv[1:6]
    work_dir_path = os.path.abspath(work_dir_path)
    os.makedirs(work_dir_path, exist_ok=True)
    dirs = {'p1': p1_dir_path, 'p2': p2_dir_path}
    images = {}
    for key, (game, char_id, _name, _role, crop) in PEOPLE.items():
        images[key] = prepared_image(os.path.join(dirs[game], char_id + '.png'), work_dir_path, crop)

    with open(story_file_path, encoding='utf-8') as f:
        text = f.read()
    # the title and the note about the sources go to the cover and the end
    title_line, rest = text.split('\n', 1)
    note, body = rest.split('\n---\n', 1)
    appendix_at = body.index('## Приложение')
    story, appendix = body[:appendix_at], body[appendix_at:]
    story_html = markdown.markdown(story, extensions=['tables'])
    appendix_html = markdown.markdown(appendix, extensions=['tables'])
    appendix_html = appendix_html.replace('<table>', '<table class="src">')
    note_html = markdown.markdown(note.strip())

    for start, keys, side in PLACES:
        marker = '<p>' + start
        assert story_html.count(marker) == 1, start
        figures = ''.join(figure_html(k, images, side) for k in keys)
        story_html = story_html.replace(marker, figures + marker)
    marker = '<p>' + REVEAL_AT
    assert story_html.count(marker) == 1
    reveal = ('<div class="reveal"><table><tr><td>%s</td><td class="arrow">⟶</td><td>%s</td></tr></table></div>' % (
        figure_html('zak', images), figure_html('atzako', images)))
    end_of_paragraph = story_html.index('</p>', story_html.index(marker)) + len('</p>')
    story_html = story_html[:end_of_paragraph] + reveal + story_html[end_of_paragraph:]

    def grid(keys):
        cells = ['<td>%s</td>' % figure_html(k, images) for k in keys]
        rows = ['<tr>%s</tr>' % ''.join(cells[i:i + 4]) for i in range(0, len(cells), 4)]
        return '<table class="grid">%s</table>' % ''.join(rows)

    cover = """
<section class="cover"><div class="inner">
  <p class="kicker">Летопись аллодов</p>
  <h1>Проклятие<br>джунов</h1>
  <p class="sub">повесть о двух беглецах</p>
  <p class="orn">❦</p>
  <table class="duo"><tr>
    <td><div class="medal"><img src="file://%s"></div><div class="who">Зак<i>Избранный · «Проклятые Земли»</i></div></td>
    <td><div class="medal"><img src="file://%s"></div><div class="who">Кир<i>беглый раб · «Затерянные в Астрале»</i></div></td>
  </tr></table>
  <p class="games">Сюжет «Проклятых Земель» (Evil Islands, 2000)<br>и «Затерянных в Астрале» (Evil Islands 2, 2006),<br>сведенный в одно повествование</p>
</div></section>""" % (images['zak'], images['kir'])
    cast = """
<section class="cast">
  <h2>Действующие лица</h2>
  <p class="intro">Портреты нарисованы по моделям и текстурам персонажей из файлов игр.</p>
  <h4>Часть первая · Избранный</h4>%s
  <h4>Часть вторая · Кир-А-Чит-Ла</h4>%s
</section>""" % (grid(CAST_ONE), grid(CAST_TWO))
    document = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>%s</title><style>%s</style></head>
<body>%s%s<main>%s</main><section class="appendix">%s</section></body></html>""" % (
        html.escape(title_line.lstrip('# ')), CSS, cover, cast, story_html,
        appendix_html.replace('</h2>', '</h2><p class="first note"><i>%s</i></p>' % re.sub(r'^<p>|</p>$', '', note_html), 1))
    with open(os.path.join(work_dir_path, 'story.html'), 'w', encoding='utf-8') as f:
        f.write(document)
    weasyprint.HTML(string=document, base_url=work_dir_path).write_pdf(pdf_file_path)
    if _Debug:
        print('%s: %d bytes' % (pdf_file_path, os.path.getsize(pdf_file_path)))


if __name__ == '__main__':
    main()
