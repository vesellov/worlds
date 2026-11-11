"""
Tests of engine.briefing: the title, the speech of the units and the pages of a briefing.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from engine import briefing

#------------------------------------------------------------------------------

TEXT = """The meeting in the village
#show Hero 0 weapon
#show Elder
#ANIMATION 30
#NOLIPS
#phrase Hero  1
Oh dear! Looks like the whole village is here!
#DESC To himself
#CAMERA 8
#phrase Sheriff  2
Right, everyone here?
#phrase Elder  3
My people! Today is a great day.
"""

#------------------------------------------------------------------------------


class BriefingTest(unittest.TestCase):

    def test_parse(self):
        value = briefing.parse('s1', TEXT)
        self.assertEqual(value.name, 's1')
        self.assertEqual(value.title, 'The meeting in the village')
        self.assertEqual(value.lines, [('Hero', 'Oh dear! Looks like the whole village is here!'),
                                       ('Sheriff', 'Right, everyone here?'),
                                       ('Elder', 'My people! Today is a great day.')])
        self.assertEqual([(u.speaker, u.voice) for u in value.utterances],
                         [('Hero', 1), ('Sheriff', 2), ('Elder', 3)])
        self.assertEqual(briefing.parse('empty', ''), None)

    def test_speech_pages_keep_the_phrases_whole(self):
        value = briefing.parse('s1', TEXT)
        pages = value.speech_pages()
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].voices, [1, 2, 3])
        self.assertEqual(len(pages[0].lines), 3)
        two = briefing.parse('two', 'Title\n#phrase A 1\nfirst\nsecond\n#phrase B 2\nthird\n')
        pages = two.speech_pages()
        self.assertEqual(pages[0].lines, [('A', 'first'), ('A', 'second'), ('B', 'third')])
        self.assertEqual(pages[0].voices, [1, 2])
        self.assertEqual(briefing.voice_name('z1', 4), 'briefing/z1/4.mp3')

    def test_pages(self):
        lines = [('Hero', 'line %d' % i) for i in range(briefing.PAGE_LINES * 2 + 3)]
        value = briefing.Briefing('long', 'Long', lines)
        pages = value.pages()
        self.assertEqual(len(pages), 3)
        self.assertEqual(len(pages[0]), briefing.PAGE_LINES)
        self.assertEqual(len(pages[2]), 3)
        self.assertEqual(briefing.Briefing('empty', 'Empty').pages(), [[]])

    def test_load(self):
        texts = {'briefing s1': TEXT}
        value = briefing.load('s1', texts.get)
        self.assertEqual(value.title, 'The meeting in the village')
        self.assertIsNone(briefing.load('unknown', texts.get))
