"""
Skeleton.pose_arrays (all the parts at once) gives the same pose as the bone update part by part (pose_slow).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

import numpy as np

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import figure

#------------------------------------------------------------------------------


class PoseArraysTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()

    def check(self, skeleton, tracks, time):
        fast = skeleton.pose(tracks, time)
        slow = skeleton.pose_slow(tracks, time)
        self.assertEqual(set(fast), set(slow))
        for name, (position, rotation) in slow.items():
            self.assertTrue(np.allclose(fast[name][0], position, atol=1e-9), name)
            self.assertTrue(np.allclose(fast[name][1], rotation, atol=1e-9), name)

    def test_clips_of_a_human(self):
        model = figure.FigureModel(self.game_data, 'unhuma')
        model.skeleton.scale = 1.2
        checked = 0
        for action in model.database.actions[:25]:
            clip = model.clip(action)
            if clip is None:
                continue
            for time in (0.0, 0.35, clip.length * 0.5, clip.length - 0.1, clip.length + 3.0):
                self.check(model.skeleton, clip.tracks, time)
                checked += 1
        self.assertGreater(checked, 20)

    def test_no_tracks(self):
        model = figure.FigureModel(self.game_data, 'unhufe')
        self.check(model.skeleton, {}, 0.0)
        self.check(model.skeleton, {}, 7.5)


if __name__ == '__main__':
    unittest.main()
