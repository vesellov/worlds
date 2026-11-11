"""
H5: the 3D scene of a briefing: the cast of "#show", "#ANIMATION", "#CAMERA" and "#NOLIPS" of the phrases.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from engine import briefing

#------------------------------------------------------------------------------

TEXT = """The meeting
#show Hero 1
#show Merc3 2
#ANIMATION 30
#NOLIPS
#phrase Hero 1
Hello.
#CAMERA 8
#ANIMATION 45
#phrase Merc3 2
Hi.
#hide Merc3
#phrase Hero 3
Bye.
"""

#------------------------------------------------------------------------------


class BriefingSceneTest(unittest.TestCase):

    def test_the_cast(self):
        value = briefing.parse('scene', TEXT)
        self.assertEqual(value.cast, [('Hero', 1), ('Merc3', 2)])

    def test_the_directives_of_the_phrases(self):
        value = briefing.parse('scene', TEXT)
        first, second, third = value.utterances
        self.assertEqual((first.speaker, first.animation, first.lips), ('Hero', 30, False))
        self.assertIsNone(first.camera)
        self.assertEqual((second.speaker, second.animation, second.camera, second.lips),
                         ('Merc3', 45, 8, True))
        self.assertEqual(third.speaker, 'Hero')
        self.assertEqual(third.animation, 45)

    def test_a_cast_without_slots(self):
        value = briefing.parse('plain', 'Title\n#show Hero weapon\n#phrase Hero 1\nHi\n')
        self.assertEqual(value.cast, [('Hero', 0)])


if __name__ == '__main__':
    unittest.main()
