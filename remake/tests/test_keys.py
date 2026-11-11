"""
Tests of eilib.keys and of the hotkey groups of engine.options: config/keyboard.ini, the names of the
keys of Kivy and the capture of a key in the options window.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import keys
from engine import options

#------------------------------------------------------------------------------


class KeysTest(unittest.TestCase):

    def test_defaults_cover_the_actions(self):
        for action in keys.ACTIONS:
            self.assertIn(action, keys.DEFAULTS)
        self.assertEqual(keys.DEFAULTS['run'], 'Z')
        self.assertEqual(keys.DEFAULTS['cs_head'], 'KP_UP')
        self.assertEqual(keys.DEFAULTS['pause'], 'SPACE')
        # 787080: the 59 hotkeys of game.exe, accel and decel among them
        self.assertEqual(len(keys.ACTIONS), 59)

    def test_parse_keyboard_ini(self):
        text = '1 spell1\n2 spell2\nZ run\nKP_UP cs_head\n; a comment\nQ unknown\n\nSPACE pause\n'
        parsed = keys.parse(text)
        self.assertEqual(parsed, {'spell1': '1', 'spell2': '2', 'run': 'Z', 'cs_head': 'KP_UP',
                                  'pause': 'SPACE'})

    def test_names_of_the_keys(self):
        self.assertEqual(keys.kivy_name('SPACE'), 'spacebar')
        self.assertEqual(keys.kivy_name('KP_UP'), 'numpad8')
        self.assertEqual(keys.kivy_name('Z'), 'z')
        self.assertEqual(keys.kivy_name("'"), "'")
        self.assertEqual(keys.windows_name('spacebar'), 'SPACE')
        self.assertEqual(keys.windows_name('numpad8'), 'KP_UP')
        self.assertEqual(keys.windows_name('z'), 'Z')
        self.assertEqual(keys.windows_name('unknown-key'), '')
        self.assertEqual(keys.kivy_name('NOPE'), '')

    def test_bindings_of_the_options(self):
        values = {'key_run': 'W'}
        bound = keys.bindings(values)
        self.assertEqual(bound['run'], 'W')
        self.assertEqual(bound['walk'], 'X')
        self.assertEqual(bound['cs_body'], 'KP_CENTER')

    def test_hotkey_groups_of_the_options(self):
        names = [name for name, _keys in options.GROUPS]
        for name in ('actions', 'items', 'spells', 'cshots', 'windows'):
            self.assertIn(name, names)
        self.assertEqual(options.OPTIONS['key_run'][0], options.KEY)
        self.assertEqual(options.option_label(lambda key: None, 'key_run'), 'Run')
        self.assertEqual(options.option_label({'string action_cs_head': 'Aim at head\r\n'}.get, 'key_cs_head'),
                         'Aim at head')
        self.assertEqual(options.option_label(lambda key: None, 'key_cs_head'), 'Cs head')
        self.assertEqual(options.value_text(lambda key: None, 'key_run', 'Z'), 'Z')
        self.assertEqual(options.option_tip(lambda key: None, 'key_run'), '')
        # the key names of the options window of game.exe (P3-D10)
        self.assertEqual(keys.display_name('KP_CENTER'), 'KP_Begin')
        self.assertEqual(keys.display_name('KP_UP'), 'KP_Up')
        self.assertEqual(keys.display_name('PGUP'), 'Prior')
        self.assertEqual(keys.display_name('PGDN'), 'Next')
        self.assertEqual(keys.display_name('SPACE'), 'space')
        self.assertEqual(keys.display_name('TAB'), 'Tab')
        # the speed keys of the group Game (plan 3)
        self.assertEqual(keys.display_name('KP_PLUS'), 'KP_Add')
        self.assertEqual(keys.display_name('KP_MINUS'), 'KP_Subtract')
        self.assertEqual(options.value_text(lambda key: None, 'key_cs_body', 'KP_CENTER'), 'KP_Begin')

    def test_keyboard_ini_fills_the_settings(self):
        settings = options.Settings({'key_run': 'W'})
        options.load_hotkeys(settings, 'Z run\nX walk\nM w_minimap\nQ unknown\n')
        self.assertEqual(settings.value('key_run'), 'W')
        self.assertEqual(settings.value('key_walk'), 'X')
        self.assertEqual(settings.value('key_w_minimap'), 'M')
        settings = options.Settings()
        options.load_hotkeys(settings, 'Z run\nXY zz\n')
        self.assertEqual(settings.value('key_run'), 'Z')
        self.assertEqual(settings.value('key_walk'), keys.DEFAULTS['walk'])

    def test_tooltip_time(self):
        # 6522c7: ToolTip Time 0.5 s of clean settings is the slider 2, 0.25 s a step
        self.assertEqual(options.Settings().value('tooltip_time'), 2)
        self.assertEqual(options.Settings().value('tooltip_time') * 250.0, 500.0)

    def test_capture_of_a_key(self):
        settings = options.Settings()
        dialog = options.Dialog(settings)
        values = [value for value, _keys in options.GROUPS]
        dialog.select_group(values.index('actions'))
        dialog.select_option(dialog.group_keys.index('key_run'))
        self.assertEqual(dialog.on_key('enter'), 'capture')
        self.assertTrue(dialog.capture)
        self.assertEqual(dialog.on_key('k'), 'change')
        self.assertFalse(dialog.capture)
        self.assertEqual(settings.value('key_run'), 'K')
        self.assertEqual(dialog.on_key('enter'), 'capture')
        self.assertEqual(dialog.on_key('escape'), 'capture')
        self.assertFalse(dialog.capture)
        self.assertEqual(settings.value('key_run'), 'K')

    def test_capture_of_an_unknown_key_waits(self):
        settings = options.Settings()
        dialog = options.Dialog(settings)
        values = [value for value, _keys in options.GROUPS]
        dialog.select_group(values.index('items'))
        dialog.select_option(dialog.group_keys.index('key_weapon1'))
        self.assertEqual(dialog.on_key('enter'), 'capture')
        self.assertEqual(dialog.on_key('nonsense'), 'capture')
        self.assertTrue(dialog.capture)

    def test_a_hotkey_does_not_change_with_left_and_right(self):
        settings = options.Settings()
        dialog = options.Dialog(settings)
        values = [value for value, _keys in options.GROUPS]
        dialog.select_group(values.index('cshots'))
        dialog.select_option(dialog.group_keys.index('key_cs_head'))
        before = settings.value('key_cs_head')
        self.assertIsNone(dialog.on_key('left'))
        self.assertIsNone(dialog.on_key('right'))
        self.assertEqual(settings.value('key_cs_head'), before)


if __name__ == '__main__':
    unittest.main()
