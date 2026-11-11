"""
Tests of engine.tooltip: the delay and the place of the tooltips of the windows (65e540).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from engine import options
from engine import tooltip

#------------------------------------------------------------------------------


class TooltipTest(unittest.TestCase):

    def test_box(self):
        # the frame of plan 3: the mouse at (330, 167), the box of 302 x 53 at (354, 191)
        self.assertEqual(tooltip.box(330, 167, 302, 53), (354, 191, 656, 244))
        # beyond the right edge it moves left, beyond the bottom its bottom goes to the mouse
        self.assertEqual(tooltip.box(700, 100, 200, 50), (600, 124, 800, 174))
        self.assertEqual(tooltip.box(100, 580, 200, 50), (124, 530, 324, 580))

    def test_size(self):
        lines = tooltip.lines_of('Show path\r\nOn: show the path.\r\nOff: do not show path.\r\n')
        self.assertEqual(lines, ['Show path', 'On: show the path.', 'Off: do not show path.'])
        self.assertEqual(tooltip.size(lines, len), (22 + 12, 3 * 15 + 8))

    def test_delay(self):
        tip = tooltip.Tooltip()
        delay = tooltip.delay_ms(options.TOOLTIP_TIME)
        self.assertEqual(delay, 500.0)
        tip.hover('tip option_gamma', 10, 10)
        self.assertFalse(tip.update(400.0, delay))
        self.assertTrue(tip.update(100.0, delay))
        self.assertFalse(tip.update(100.0, delay))
        self.assertFalse(tip.hover('tip option_gamma', 12, 10))
        self.assertTrue(tip.hover(None, 0, 0))
        self.assertFalse(tip.update(1000.0, delay))
        tip.hover('tip option_contrast', 10, 10)
        self.assertFalse(tip.update(1000.0, 0.0))

    def test_tips_of_the_options_window(self):
        dialog = options.Dialog(options.Settings())
        group = dialog.group_row_rect(2)
        self.assertEqual(dialog.tip_at(group[0] + 5, group[1] + 5), 'tip option_group_sens')
        row = dialog.option_row_rect(7)
        self.assertEqual(dialog.tip_at(row[0] + 5, row[1] + 5), 'tip option_show_path')
        empty = dialog.option_row_rect(3)
        self.assertIsNone(dialog.tip_at(empty[0] + 5, empty[1] + 5))
        dialog.select_group(9)
        row = dialog.option_row_rect(0)
        self.assertEqual(dialog.tip_at(row[0] + 5, row[1] + 5), 'tip action_cs_head')
        self.assertIsNone(dialog.tip_at(10, 10))


if __name__ == '__main__':
    unittest.main()
