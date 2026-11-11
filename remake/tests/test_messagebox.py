"""
Tests of engine.messagebox: the buttons and the keys of the message box of 634750 and the texts of
the question of "Exit to main menu".
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import messagebox

#------------------------------------------------------------------------------

def _game_data():
    try:
        return gamedata.GameData()
    except IOError:
        return None


def _center(rect):
    left, top, right, bottom = rect
    return (left + right) // 2, (top + bottom) // 2

#------------------------------------------------------------------------------


class MessageBoxTest(unittest.TestCase):

    def test_buttons(self):
        dialog = messagebox.Dialog()
        self.assertEqual(dialog.click(*_center(messagebox.OK_REGION)), 'confirm')
        self.assertEqual(dialog.click(*_center(messagebox.CANCEL_REGION)), 'cancel')
        self.assertIsNone(dialog.click(*_center(messagebox.PANEL_RECT)))

    def test_keys(self):
        dialog = messagebox.Dialog()
        self.assertEqual(dialog.on_key('enter'), 'confirm')
        self.assertEqual(dialog.on_key('return'), 'confirm')
        self.assertEqual(dialog.on_key('escape'), 'cancel')
        self.assertIsNone(dialog.on_key('a'))

    def test_rects_of_the_window(self):
        rects = [messagebox.PANEL_RECT, messagebox.FRAME_RECT, messagebox.TITLE_BOX, messagebox.MESSAGE_BOX,
                 messagebox.OK_RECT, messagebox.CANCEL_RECT, messagebox.OK_REGION, messagebox.CANCEL_REGION]
        for left, top, right, bottom in rects:
            self.assertTrue(0 <= left < right <= 800 and 0 <= top < bottom <= 600, (left, top, right, bottom))
        # the frame holds the panel, the buttons stand under it and their centers are on the sides
        self.assertLessEqual(messagebox.FRAME_RECT[0], messagebox.PANEL_RECT[0])
        self.assertLessEqual(messagebox.FRAME_RECT[1], messagebox.PANEL_RECT[1])
        self.assertGreaterEqual(messagebox.FRAME_RECT[2], messagebox.PANEL_RECT[2])
        self.assertGreaterEqual(messagebox.FRAME_RECT[3], messagebox.PANEL_RECT[3])
        self.assertGreaterEqual(messagebox.OK_RECT[1], messagebox.PANEL_RECT[3])
        self.assertEqual(messagebox.OK_RECT[1], messagebox.CANCEL_RECT[1])
        self.assertEqual(messagebox.OK_RECT[3], messagebox.CANCEL_RECT[3])
        self.assertLess(messagebox.OK_RECT[2], messagebox.CANCEL_RECT[0])

    def test_texts_of_the_game(self):
        game_data = _game_data()
        if game_data is None:
            self.skipTest('no game data')
        self.assertEqual(game_data.text(messagebox.TITLE_KEY).strip(), 'Quit game')
        self.assertIn('quit this game', game_data.text(messagebox.MESSAGE_KEY))
