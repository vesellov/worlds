"""
Tests of engine.questmap: map.txt, the allod of a quest, the journal, the markers and the description.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from engine import questmap

#------------------------------------------------------------------------------

MAP_TXT = """#allod gipat
ingm2arrow

#zone gz1g gipat game
#res
zone1 zone1
#maps
8 6
zone1map zone1quest
#position
-2.370 0.120 0.805

#zone bz1g gipat brief
#position
-2.749 -0.186 0.261

#quest q0g
-2.871 -0.307 0.235
190.0 106.0
#quest q1g
-2.391 0.335 0.222
44.0 250.0
#quest q9k
-0.100 0.200 0.300
100.0 100.0
#quest qz
0.0 0.0 0.0
"""

TEXTS = {
    'quest q0g': 'Reaching people\r\nFind a settlement.\r\n#subobj 1\r\nLeave the ruins\r\n'
                 'The ruins are dangerous.\r\n',
}

#------------------------------------------------------------------------------


def text_of(name):
    return TEXTS.get(name)


class QuestMapTest(unittest.TestCase):

    def test_map_txt(self):
        quests, zones = questmap.parse_map_txt(MAP_TXT)
        self.assertEqual([q.name for q in quests], ['q0g', 'q1g', 'q9k', 'qz'])
        self.assertEqual(quests[0].position, (-2.871, -0.307, 0.235))
        self.assertEqual(quests[0].point, (190.0, 106.0))
        self.assertEqual(quests[2].allod, 'ingos')
        self.assertEqual(quests[0].allod, 'gipat')
        self.assertEqual([z.name for z in zones], ['gz1g', 'bz1g'])
        self.assertEqual(zones[0].position, (-2.370, 0.120, 0.805))
        self.assertEqual(zones[0].type, 'game')

    def test_journal_of_an_allod(self):
        quests, zones = questmap.parse_map_txt(MAP_TXT)
        model = questmap.QuestMap(quests, text_of)
        model.set_allod('gipat')
        self.assertEqual([q.name for q in model.journal()], ['q0g', 'q1g', 'qz'])
        self.assertEqual([q.name for q in model.map_quests()], ['q0g', 'q1g'])
        model.set_allod('ingos')
        self.assertEqual([q.name for q in model.journal()], ['q9k'])
        self.assertEqual(model.map_quests()[0].point, (100.0, 100.0))

    def test_markers_on_the_map(self):
        model = questmap.QuestMap(questmap.parse_map_txt(MAP_TXT)[0])
        point = model.map_point(questmap.QuestPoint('q0g', point=(0.0, 0.0)))
        self.assertAlmostEqual(point[0], questmap.MAP_RECT[0])
        self.assertAlmostEqual(point[1], questmap.MAP_RECT[1])
        quest = model.quest_at(point[0], point[1])
        self.assertIsNone(quest)
        model.set_allod('gipat')
        selected = model.quest_at(*model.map_point(model.entries[0]))
        self.assertEqual(selected.name, 'q0g')

    def test_click_and_description(self):
        quests, _ = questmap.parse_map_txt(MAP_TXT)
        model = questmap.QuestMap(quests, text_of)
        self.assertEqual(model.click(*_center(model.row_rect(1))), 'select')
        self.assertEqual(model.selected_quest.name, 'q1g')
        self.assertEqual(model.on_key('up'), None)
        self.assertEqual(model.on_key('escape'), 'close')
        model.select(0)
        title, lines = model.description()
        self.assertEqual(title, 'Reaching people')
        self.assertEqual(lines, ['Find a settlement.', 'Leave the ruins', 'The ruins are dangerous.'])


def _center(rect):
    left, top, right, bottom = rect
    return (left + right) // 2, (top + bottom) // 2
