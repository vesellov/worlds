"""
Tests of the credits: the RTF text of res/outro.res, credits.scr and the scrolling of engine.credits.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from eilib import rtf
from engine import credits

#------------------------------------------------------------------------------

RTF = (b"{\\rtf1\\ansi{\\fonttbl{\\f0\\froman Times New Roman;}}"
       b"Hello \\b bold\\b0 \\par Second \\'c0\\'e5\\'f2 \\u1040?\\par"
       b"{\\*\\unknown hidden}End}")

#------------------------------------------------------------------------------


def _game_data():
    try:
        return gamedata.GameData()
    except IOError:
        return None

#------------------------------------------------------------------------------


class RtfTest(unittest.TestCase):

    def test_plain_text(self):
        text = rtf.plain_text(RTF)
        self.assertIn('Hello bold', text)
        self.assertIn('Second', text)
        self.assertIn('Ает', text)
        self.assertIn('А', text)
        self.assertNotIn('hidden', text)
        self.assertTrue(text.endswith('End'))

    def test_line_breaks(self):
        self.assertEqual(rtf.plain_text(b'{\\rtf1 one\\par two\\line three}'), 'one\ntwo\nthree')

    def test_skipped_groups(self):
        data = b'{\\rtf1 {\\colortbl;\\red255\\green0\\blue0;}Text{\\pict \\bin1 x}End}'
        self.assertEqual(rtf.plain_text(data), 'TextEnd')


class CreditsTest(unittest.TestCase):

    def test_script(self):
        self.assertEqual(credits.parse_script(b'#scroll 25 outro.rtf\r\n'), (25.0, 'outro.rtf'))
        self.assertEqual(credits.parse_script(b'#scroll 10 x.rtf'), (10.0, 'x.rtf'))

    def test_scroll(self):
        item = credits.Credits('one\ntwo', speed=10.0, height=100.0)
        self.assertEqual((item.line_count, item.text_height), (2, 2 * credits.LINE_HEIGHT))
        self.assertEqual(item.offset, 100.0)
        self.assertFalse(item.update(1000.0))
        self.assertAlmostEqual(item.offset, 90.0)
        self.assertFalse(item.is_done)
        item.update(20000.0)
        self.assertTrue(item.is_done)

    def test_credits_of_the_game(self):
        game_data = _game_data()
        if game_data is None:
            self.skipTest('no game data')
        item = credits.Credits.from_game(game_data)
        self.assertIsNotNone(item)
        self.assertEqual(item.speed, 25.0)
        self.assertGreater(item.line_count, 100)
        self.assertIn('Nival Interactive', item.text)
