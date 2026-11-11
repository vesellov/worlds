"""
Tests of engine.difficulty: the levels of a new game, their game.exe index and the clicks of the window
of 633770.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import difficulty

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


class DifficultyTest(unittest.TestCase):

    def test_levels_and_game_index(self):
        # Normal is the default; Novice is the index 1 of [DifficultyLevels] (weaker enemies)
        self.assertEqual(difficulty.DEFAULT, difficulty.NORMAL)
        self.assertEqual(difficulty.GAME_INDEX[difficulty.NOVICE], 1)
        self.assertEqual(difficulty.GAME_INDEX[difficulty.NORMAL], 0)
        self.assertEqual(difficulty.Dialog().game_index, 0)

    def test_click_selects_a_row(self):
        dialog = difficulty.Dialog()
        self.assertEqual(dialog.click(*_center(difficulty.ROW_RECTS[difficulty.NOVICE])), 'select')
        self.assertEqual(dialog.selection, difficulty.NOVICE)
        self.assertEqual(dialog.game_index, 1)
        self.assertEqual(dialog.click(*_center(difficulty.ROW_RECTS[difficulty.NORMAL])), 'select')
        self.assertEqual(dialog.selection, difficulty.NORMAL)

    def test_buttons(self):
        dialog = difficulty.Dialog()
        self.assertEqual(dialog.click(*_center(difficulty.OK_RECT)), 'confirm')
        self.assertEqual(dialog.click(*_center(difficulty.CANCEL_RECT)), 'cancel')
        self.assertIsNone(dialog.click(5, 5))

    def test_keys(self):
        dialog = difficulty.Dialog()
        self.assertEqual(dialog.on_key('escape'), 'cancel')
        self.assertIsNone(dialog.on_key('a'))

    def test_rects_of_the_window(self):
        rects = [difficulty.PANEL_RECT, difficulty.OK_RECT, difficulty.CANCEL_RECT]
        rects.extend(difficulty.ROW_RECTS.values())
        rects.extend(difficulty.CHECKBOX_RECTS.values())
        rects.extend(difficulty.LABEL_BOXES.values())
        for left, top, right, bottom in rects:
            self.assertTrue(0 <= left < right <= 800 and 0 <= top < bottom <= 600, (left, top, right, bottom))
        novice = difficulty.ROW_RECTS[difficulty.NOVICE]
        normal = difficulty.ROW_RECTS[difficulty.NORMAL]
        self.assertLessEqual(novice[3], normal[1])

    def test_texts_of_the_game(self):
        game_data = _game_data()
        if game_data is None:
            self.skipTest('no game data')
        self.assertEqual(difficulty.level_label(game_data.text, difficulty.NOVICE), 'Novice')
        self.assertEqual(difficulty.level_label(game_data.text, difficulty.NORMAL), 'Normal')
        self.assertIn('Novice', game_data.text(difficulty.TIP_KEY))
        self.assertIn('Normal', game_data.text(difficulty.TIP_KEY))
