"""
The map of an allod and the journal of its quests (game.exe 64cbd0 and 64dc20).

map.txt lists the allods, the zones and the quests: "#allod <name>" starts the figures of an allod,
"#zone <name> <allod> <type>" has a "#position" of three floats, "#quest <name>" has a position and
sometimes a point on the map of the allod in its 512 x 512 texture (the two numbers of the next line).
The suffix of the name of a quest tells its allod: g is gipat, k is ingos, h is suslanger.

64cbd0 draws the map texture "<allod>map" 300 pixels wide and high in the centre (580, 190) of the screen
(the half 150 is scaled by the sides of the map), the markers of the zones and the quests over it.
64dc20 draws the chosen quest: its title and its description from "quest <name>" of texts.res, the status
from the variables of the quest, the experience and the money of QuestComplete.

The remake reads the quests of map.txt and the points of their markers; the zones are only listed by their
names, the rectangles of the zones of the allod map data of game.exe are not read. The tests cover the list,
the allod of a quest, the point of a marker and the lines of the description.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import re

#------------------------------------------------------------------------------

from eilib import zones as zones_module

#------------------------------------------------------------------------------

# 64cbd0: the map "<allod>map" is 300 pixels wide and high around (580, 190) of the screen
MAP_CENTER = (580.0, 190.0)
MAP_SIZE = 300.0
MAP_TEXTURE_SIZE = 512.0
MAP_RECT = (MAP_CENTER[0] - MAP_SIZE * 0.5, MAP_CENTER[1] - MAP_SIZE * 0.5,
            MAP_CENTER[0] + MAP_SIZE * 0.5, MAP_CENTER[1] + MAP_SIZE * 0.5)
MARKER_RADIUS = 6.0
MARKER_SELECTED_RADIUS = 10.0

# the allod of a quest by the last letter of its name
ALLOD_SUFFIXES = {'g': 'gipat', 'k': 'ingos', 'h': 'suslanger'}
DEFAULT_ALLOD = 'gipat'
ALLODS = ('gipat', 'ingos', 'suslanger')

# 64dc20: the journal of the quests of the remake
LIST_RECT = (10, 70, 410, 330)
LIST_ROWS_TOP = 82
LIST_ROW_HEIGHT = 20
LIST_VISIBLE_ROWS = 12
DETAILS_RECT = (10, 340, 410, 570)
DETAILS_BOX = (18, 348, 402, 562)
TITLE_BOX = (10, 40, 410, 66)
CLOSE_RECT = (740, 40, 765, 65)

TITLE_TEXT = 'Journal'

# the directives of the description of a quest
SUBOBJECTIVE = '#subobj'
DIRECTIVE = re.compile(r'^#\w+')

#------------------------------------------------------------------------------


class QuestPoint(object):
    """
    A quest of map.txt: its name, the position of the editor and the point on the map of its allod.
    """

    def __init__(self, name, position=None, point=None):
        self.name = name
        self.position = tuple(position) if position else None
        self.point = tuple(point) if point else None

    def __repr__(self):
        return 'QuestPoint(%s, %s)' % (self.name, self.point)

    @property
    def allod(self):
        return ALLOD_SUFFIXES.get(self.name[-1:].lower(), DEFAULT_ALLOD)


class ZonePoint(object):
    """
    A zone of map.txt: its name, its allod and the position of the editor.
    """

    def __init__(self, name, allod, zone_type, position=None):
        self.name = name
        self.allod = allod
        self.type = zone_type
        self.position = tuple(position) if position else None

    def __repr__(self):
        return 'ZonePoint(%s, %s)' % (self.name, self.allod)


def parse_map_txt(text):
    """
    (quests, zones) of map.txt: the list of QuestPoint and of ZonePoint.
    """
    quests = []
    zones = []
    quest = None
    zone = None
    number_lines = 0
    # "#position" waits for the line of the numbers, the later numbers of the block (deploy, restrict)
    # do not take its place
    position_line = False
    for line in (text or '').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('##'):
            continue
        words = stripped.split()
        if words[0] == '#quest' and len(words) > 1:
            quest = QuestPoint(words[1])
            quests.append(quest)
            zone = None
            number_lines = 0
            continue
        if words[0] == '#zone' and len(words) > 3:
            zone = ZonePoint(words[1], words[2], words[3])
            zones.append(zone)
            quest = None
            number_lines = 0
            continue
        if stripped.startswith('#'):
            number_lines = 0
            position_line = words[0] == '#position'
            continue
        numbers = _numbers(words)
        if numbers is None:
            continue
        number_lines += 1
        if quest is not None:
            if number_lines == 1:
                quest.position = tuple(numbers[:3])
            elif number_lines == 2 and len(numbers) >= 2:
                quest.point = (numbers[0], numbers[1])
        elif zone is not None and position_line:
            zone.position = tuple(numbers[:3])
            position_line = False
    return quests, zones


def parse_quest_text(text):
    """
    (title, lines) of "quest <name>": the first line is the title, the lines with #subobj are the
    subobjectives, the other directives are skipped.
    """
    lines = [line.strip() for line in (text or '').replace('\r', '').split('\n')]
    lines = [line for line in lines if line]
    if not lines:
        return '', []
    title = lines[0]
    result = []
    for line in lines[1:]:
        if line.startswith(SUBOBJECTIVE):
            continue
        if DIRECTIVE.match(line):
            continue
        result.append(line)
    return title, result

#------------------------------------------------------------------------------


class QuestMap(object):
    """
    The quests of one allod and the journal of the chosen one.
    """

    def __init__(self, quests=(), texts=None, completed=()):
        self.quests = list(quests)
        self.texts = texts
        self.completed = list(completed)
        self.allod = DEFAULT_ALLOD
        # the allod of a zone -> the figures of its markers (#allod of map.txt)
        self.allod_models = {}
        self.selected = 0
        self.scroll = 0

    def __repr__(self):
        return 'QuestMap(%s, %d quests)' % (self.allod, len(self.quests))

    @classmethod
    def from_game(cls, game_data, allod=None, completed=()):
        text = game_data.text('map.txt') or ''
        quests, zones = parse_map_txt(text)
        result = cls(quests, game_data.text, completed)
        result.allod_models = {allod.name: list(allod.model_names)
                               for allod in zones_module.parse_allods(text)}
        if allod is not None:
            result.allod = allod
        return result

    def map_quests(self):
        """
        The quests of the chosen allod with a point on its map, sorted by name.
        """
        return [quest for quest in self.quests if quest.allod == self.allod and quest.point is not None]

    def journal(self):
        """
        The quests of the chosen allod sorted by name: the journal of game.exe knows them all.
        """
        return sorted((quest for quest in self.quests if quest.allod == self.allod), key=lambda q: q.name.lower())

    def set_allod(self, allod):
        self.allod = allod if allod in ALLODS else DEFAULT_ALLOD
        self.selected = 0
        self.scroll = 0

    @property
    def entries(self):
        return self.journal()

    @property
    def selected_quest(self):
        entries = self.entries
        if 0 <= self.selected < len(entries):
            return entries[self.selected]
        return None

    def select(self, index):
        entries = self.entries
        if not entries:
            return False
        index = min(max(index, 0), len(entries) - 1)
        self.selected = index
        if index < self.scroll:
            self.scroll = index
        elif index >= self.scroll + LIST_VISIBLE_ROWS:
            self.scroll = index - LIST_VISIBLE_ROWS + 1
        return True

    def is_completed(self, quest):
        return quest.name.lower() in [name.lower() for name in self.completed]

    def map_point(self, quest):
        """
        The point of the marker of the quest on the screen of the map.
        """
        if quest.point is None:
            return None
        u, v = quest.point
        x = MAP_RECT[0] + u / MAP_TEXTURE_SIZE * MAP_SIZE
        y = MAP_RECT[1] + v / MAP_TEXTURE_SIZE * MAP_SIZE
        return (x, y)

    def quest_at(self, x, y):
        best = None
        for quest in self.map_quests():
            point = self.map_point(quest)
            if point is None:
                continue
            distance = (point[0] - x) ** 2 + (point[1] - y) ** 2
            if distance <= MARKER_SELECTED_RADIUS * MARKER_SELECTED_RADIUS:
                if best is None or distance < best[0]:
                    best = (distance, quest)
        return best[1] if best is not None else None

    def row_rect(self, row):
        top = LIST_ROWS_TOP + row * LIST_ROW_HEIGHT
        return (LIST_RECT[0] + 4, top, LIST_RECT[2] - 4, top + LIST_ROW_HEIGHT)

    def click(self, x, y):
        """
        "select", "close" or None.
        """
        for row in range(LIST_VISIBLE_ROWS):
            left, top, right, bottom = self.row_rect(row)
            if left <= x < right and top <= y < bottom:
                index = self.scroll + row
                if index < len(self.entries):
                    self.select(index)
                    return 'select'
        quest = self.quest_at(x, y)
        if quest is not None:
            entries = self.entries
            if quest in entries:
                self.select(entries.index(quest))
                return 'select'
        if _inside(CLOSE_RECT, x, y):
            return 'close'
        return None

    def on_key(self, name):
        if name == 'up':
            self.select(self.selected - 1)
        elif name == 'down':
            self.select(self.selected + 1)
        elif name == 'escape':
            return 'close'
        return None

    def description(self):
        """
        (title, [lines]) of the chosen quest.
        """
        quest = self.selected_quest
        if quest is None or self.texts is None:
            return '', []
        return parse_quest_text(self.texts('quest %s' % quest.name))

#------------------------------------------------------------------------------


def _numbers(words):
    try:
        return [float(word) for word in words]
    except ValueError:
        return None


def _inside(rect, x, y):
    left, top, right, bottom = rect
    return left <= x < right and top <= y < bottom
