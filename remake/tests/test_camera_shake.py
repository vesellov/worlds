"""
G9: the shake of the camera from a blow or a lightning (4113a0).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from engine import camera

#------------------------------------------------------------------------------


class ShakeTest(unittest.TestCase):

    def test_shake_moves_the_eye(self):
        value = camera.Camera((0.0, 0.0, 1.0), yaw=0.0, pitch=45.0, distance=30.0)
        steady = value.eye()
        self.assertTrue(value.shake(point=(0.0, 0.0, 1.0)))
        value.advance_shake(0.1)
        moved = value.eye()
        self.assertFalse((steady == moved).all())

    def test_shake_ends(self):
        value = camera.Camera()
        value.shake(strength=5.0, time=1.0)
        value.advance_shake(1.0)
        self.assertEqual(value.shake_strength, 0.0)
        self.assertEqual(value.shake_offset(), (0.0, 0.0, 0.0))

    def test_far_point_does_not_shake(self):
        value = camera.Camera((0.0, 0.0, 1.0))
        self.assertFalse(value.shake(point=(0.0, camera.SHAKE_DISTANCE + 1.0, 1.0)))
        self.assertFalse(value.shake(strength=0.0))
        self.assertTrue(value.shake(point=(0.0, 0.0, 1.0)))

    def test_mouse_drag_and_reverse(self):
        value = camera.Camera()
        yaw, pitch = value.yaw, value.pitch
        value.drag(10.0, 5.0)
        self.assertLess(value.yaw, yaw)
        self.assertLess(value.pitch, pitch)
        normal_yaw, normal_pitch = value.yaw, value.pitch
        value.reverse_x = True
        value.reverse_y = True
        value.drag(10.0, 5.0)
        self.assertGreater(value.yaw, normal_yaw)
        self.assertGreater(value.pitch, normal_pitch)


if __name__ == '__main__':
    unittest.main()
