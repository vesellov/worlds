"""
Tests of engine.options: the values of the options, the file of the remake, the changes of the window
of 650380 and the volumes which they give to the client.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import tempfile
import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import difficulty
from engine import figure
from engine import options

#------------------------------------------------------------------------------


def _game_data():
    try:
        return gamedata.GameData()
    except IOError:
        return None


class FakeAudio(object):

    def __init__(self):
        self.volume_sfx = 1.0
        self.volume_voice = 1.0


class FakeMusic(object):

    def __init__(self):
        self.volume = 1.0

    def set_volume(self, volume):
        self.volume = volume


class FakeApp(object):

    def __init__(self):
        self.audio = FakeAudio()
        self.music = FakeMusic()
        self.difficulty = difficulty.NORMAL

#------------------------------------------------------------------------------


class SettingsTest(unittest.TestCase):

    def test_defaults(self):
        settings = options.Settings()
        self.assertEqual(settings.value('volume_sfx'), 100)
        self.assertEqual(settings.value('difficulty'), difficulty.NORMAL)
        self.assertEqual(settings.value('shadow_units'), 1)

    def test_change_and_limits(self):
        settings = options.Settings()
        self.assertEqual(settings.change('volume_sfx', -20), 80)
        settings.set_value('volume_sfx', 0)
        self.assertEqual(settings.change('volume_sfx', -5), 0)
        settings.set_value('gamma', 100)
        self.assertEqual(settings.change('gamma', 5), 100)
        self.assertEqual(settings.change('shadow_units', 1), 0)
        self.assertEqual(settings.change('shadow_units', 1), 1)

    def test_difficulty_choice(self):
        settings = options.Settings()
        self.assertEqual(settings.change('difficulty', 1), difficulty.NOVICE)
        self.assertEqual(settings.change('difficulty', 1), difficulty.NORMAL)
        self.assertEqual(settings.change('difficulty', -1), difficulty.NOVICE)

    def test_file(self):
        with tempfile.TemporaryDirectory() as dir_path:
            file_path = os.path.join(dir_path, 'settings.json')
            settings = options.Settings(file_path=file_path)
            settings.set_value('volume_stream', 40)
            settings.change('difficulty', 1)
            self.assertTrue(settings.save())
            loaded = options.Settings.load(file_path)
            self.assertEqual(loaded.value('volume_stream'), 40)
            self.assertEqual(loaded.value('difficulty'), difficulty.NOVICE)

    def test_apply(self):
        app = FakeApp()
        settings = options.Settings()
        settings.set_value('volume_sfx', 25)
        settings.set_value('volume_voice', 50)
        settings.set_value('volume_stream', 75)
        settings.set_value('difficulty', difficulty.NOVICE)
        settings.apply(app)
        self.assertAlmostEqual(app.audio.volume_sfx, 0.25)
        self.assertAlmostEqual(app.audio.volume_voice, 0.5)
        self.assertAlmostEqual(app.music.volume, 0.75)
        self.assertEqual(app.difficulty, difficulty.NOVICE)


class DialogTest(unittest.TestCase):

    def setUp(self):
        self.settings = options.Settings()
        self.dialog = options.Dialog(self.settings)

    def test_groups(self):
        # 79479c: the 11 groups of the window; 787430 and 787080: the rows of each group, None empty
        self.assertEqual(self.dialog.group, 'video')
        self.assertEqual(self.dialog.option_key, 'shadow_units')
        self.assertEqual([group for group, _keys in options.GROUPS], [
            'video', 'sound', 'sens', 'game', 'select', 'actions', 'camera', 'items', 'spells', 'cshots',
            'windows'])
        rows = dict(options.GROUPS)
        self.assertEqual(rows['video'], (
            'shadow_units', 'shadow_buildings', 'shadow_flora', None, 'marks', 'footprints', 'select_type',
            'show_path', 'show_flying_hp', None, 'brightness', 'contrast', 'gamma'))
        self.assertEqual(rows['game'], (
            'difficulty', None, 'key_accel', 'key_decel', 'key_pause', None, 'key_obj', None, 'key_quicksave',
            'key_quickload', None, 'autosave', 'switch_filters'))
        self.assertEqual(rows['windows'][-2:], ('key_tutorial_script', 'show_tutorial'))
        self.assertEqual(rows['cshots'], ('key_cs_head', 'key_cs_body', 'key_cs_rleg', 'key_cs_lleg',
                                          'key_cs_rhand', 'key_cs_lhand'))
        for _group, keys in options.GROUPS:
            self.assertLessEqual(len(keys), options.VISIBLE_ROWS)
            for key in keys:
                self.assertTrue(key is None or key in options.OPTIONS, key)
        shown = [key for _group, keys in options.GROUPS for key in keys if key]
        self.assertEqual(len(shown), len(set(shown)))
        self.assertEqual(set(shown), set(options.OPTIONS))

    def test_sensitivity_defaults(self):
        # the registry of clean settings: PowerKeyboard 0.5 and PowerMouse 1.0 are slider * 0.02,
        # CameraBorderScrollArea 5, CameraFrameSelectionSensetiveArea 20 / 10, the camera not flipped
        settings = options.Settings()
        self.assertEqual([settings.value(key) for key in ('power_kbd', 'power_mouse', 'scroll_border',
                                                          'rubber_select')], [25, 50, 5, 2])
        self.assertEqual(options.OPTIONS['scroll_border'][:4], (options.SLIDER, 5, 0, 10))
        self.assertEqual((settings.value('camera_reverse_x'), settings.value('camera_reverse_y')), (0, 0))
        self.assertEqual([settings.value(key) for key in ('marks', 'footprints', 'show_path')], [1, 1, 1])

    def test_empty_rows_are_skipped(self):
        self.dialog.select_option(2)
        self.assertEqual(self.dialog.option_key, 'shadow_flora')
        self.dialog.on_key('down')
        self.assertEqual(self.dialog.option_key, 'marks')
        self.dialog.on_key('up')
        self.assertEqual(self.dialog.option_key, 'shadow_flora')
        self.assertIsNone(self.dialog.click(*_center(self.dialog.option_row_rect(3))))
        self.assertEqual(self.dialog.option_key, 'shadow_flora')
        # the last group can be clicked (P3-D3)
        self.assertEqual(self.dialog.click(*_center(self.dialog.group_row_rect(10))), 'group')
        self.assertEqual(self.dialog.group, 'windows')

    def test_older_file_drops_the_changed_options(self):
        with tempfile.TemporaryDirectory() as dir_path:
            file_path = os.path.join(dir_path, 'settings.json')
            with open(file_path, 'w') as f:
                f.write('{"camera_reverse_x": 1, "scroll_border": 1, "volume_sfx": 40}')
            loaded = options.Settings.load(file_path)
            self.assertEqual(loaded.value('camera_reverse_x'), 0)
            self.assertEqual(loaded.value('scroll_border'), 5)
            self.assertEqual(loaded.value('volume_sfx'), 40)
            loaded.set_value('camera_reverse_x', 1)
            loaded.save()
            self.assertEqual(options.Settings.load(file_path).value('camera_reverse_x'), 1)

    def test_click_group_and_option(self):
        self.assertEqual(self.dialog.click(*_center(self.dialog.group_row_rect(1))), 'group')
        self.assertEqual(self.dialog.group, 'sound')
        self.assertEqual(self.dialog.click(*_center(self.dialog.option_row_rect(0))), 'option')
        self.assertEqual(self.dialog.option_key, 'volume_sfx')

    def test_click_value_changes(self):
        # the toggle of 651900 takes a click in the column of the values at once
        rect = self.dialog.option_row_rect(1)
        y = (rect[1] + rect[3]) // 2
        self.assertEqual(self.dialog.click(rect[0] + 10, y), 'option')
        self.assertEqual(self.dialog.option_key, 'shadow_buildings')
        rect = self.dialog.option_row_rect(0)
        self.assertEqual(self.dialog.click(options.VALUE_X + 10, (rect[1] + rect[3]) // 2), 'change')
        self.assertEqual(self.settings.value('shadow_units'), 0)

    def test_slider_geometry(self):
        # 65b870 and 65b6f0 for the row 10 (Brightness) at the middle: the arrows at the ends of
        # (530, 403)-(680, 413), the track between them, the knob at 556 + 98 * 50 / 100
        top = self.dialog.option_row_rect(10)[1]
        self.assertEqual(top, 396)
        left, track, knob, right = options.slider_rects(top, 50, 100)
        self.assertEqual(left, (530, 403, 542, 413))
        self.assertEqual(right, (668, 403, 680, 413))
        self.assertEqual(track, (534, 405, 676, 411))
        self.assertEqual(knob, (591, 403, 619, 413))
        self.assertEqual(options.slider_rects(top, 0, 10)[2][0] + 14, 556)
        self.assertEqual(options.slider_rects(top, 10, 10)[2][0] + 14, 654)
        self.assertEqual(options.slider_value(556, 100), 0)
        self.assertEqual(options.slider_value(605, 100), 50)
        self.assertEqual(options.slider_value(700, 10), 10)

    def test_slider_clicks(self):
        y = self.dialog.option_row_rect(10)[1] + 10
        self.assertEqual(self.dialog.click(535, y), 'change')
        self.assertEqual(self.dialog.option_key, 'brightness')
        self.assertEqual(self.settings.value('brightness'), 49)
        self.assertEqual(self.dialog.click(675, y), 'change')
        self.assertEqual(self.settings.value('brightness'), 50)
        self.assertEqual(self.dialog.click(654, y), 'change')
        self.assertEqual(self.settings.value('brightness'), 100)
        self.assertEqual(self.dialog.click(685, y), 'option')

    def test_buttons_and_keys(self):
        self.assertEqual(self.dialog.click(*_center(options.OK_RECT)), 'confirm')
        self.assertEqual(self.dialog.click(*_center(options.CANCEL_RECT)), 'cancel')
        self.assertEqual(self.dialog.on_key('down'), None)
        self.assertEqual(self.dialog.on_key('right'), 'change')
        self.assertEqual(self.dialog.on_key('pagedown'), 'group')
        self.assertEqual(self.dialog.on_key('escape'), 'cancel')
        self.assertEqual(self.dialog.on_key('enter'), 'confirm')

    def test_cancel_restores(self):
        self.dialog.select_group(1)
        self.dialog.select_option(0)
        self.dialog.change(-1)
        self.dialog.cancel()
        self.assertEqual(self.settings.value('volume_sfx'), 100)

    def test_keyboard_power(self):
        # 652460: CameraDefaultScrollPowerKeyboard is the slider * 0.02
        self.assertAlmostEqual(options.keyboard_power({}), 0.5)
        self.assertAlmostEqual(options.keyboard_power({'power_kbd': 50}), 1.0)

    def test_plaque_parts(self):
        # 650380: CInterfaceModel::ShowPart of options and optionslabel of the model of the escape menu
        game_data = _game_data()
        if game_data is None:
            self.skipTest('no game data')
        model = figure.FigureModel(game_data, 'unmoco1', parts=('options', 'optionslabel'))
        self.assertEqual(model.skeleton.visible, ['options', 'optionslabel'])
        self.assertEqual(model.skeleton.order, ['escmenu00', 'options', 'optionslabel'])

    def test_texts_of_the_game(self):
        game_data = _game_data()
        if game_data is None:
            self.skipTest('no game data')
        self.assertEqual(options.group_label(game_data.text, 'video'), 'Graphics')
        self.assertEqual(options.option_label(game_data.text, 'volume_sfx'), 'Sound effects volume:')
        self.assertEqual(options.value_text(game_data.text, 'shadow_units', 1), 'On')
        self.assertEqual(options.value_text(game_data.text, 'difficulty', difficulty.NOVICE), 'Novice')
        self.assertEqual(options.option_label(game_data.text, 'marks'), 'Show blood')
        self.assertEqual(options.option_label(game_data.text, 'key_accel'), 'Accelerated speed')
        self.assertEqual(options.option_label(game_data.text, 'key_cs_head'), 'Aim at head')
        self.assertTrue(options.option_tip(game_data.text, 'volume_sfx'))


def _center(rect):
    left, top, right, bottom = rect
    return (left + right) // 2, (top + bottom) // 2
