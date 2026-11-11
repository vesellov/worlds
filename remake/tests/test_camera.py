"""
Tests of engine.camera: the defaults of [Camera Settings], the eye of the camera, the limits of the pitch,
of the zoom and of the turn.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import unittest

#------------------------------------------------------------------------------

from engine import camera

#------------------------------------------------------------------------------


class Land(object):
    """
    A flat land with a step 6 high at x >= 10.
    """

    def height_at(self, x, y):
        return 6.0 if x >= 10.0 else 0.0

#------------------------------------------------------------------------------


class CameraTest(unittest.TestCase):

    def test_defaults_of_the_camera_settings(self):
        self.assertEqual(camera.DEFAULT_DISTANCE, 30.0)
        self.assertEqual(camera.DEFAULT_PITCH, 45.0)
        self.assertEqual((camera.MIN_PITCH, camera.MAX_PITCH), (20.0, 80.0))
        self.assertEqual((camera.MIN_LIMIT_DISTANCE, camera.MAX_LIMIT_DISTANCE), (3.0, 80.0))
        self.assertEqual((camera.MIN_DISTANCE, camera.MAX_DISTANCE), (4.0, 100.0))
        view = camera.Camera()
        self.assertEqual((view.distance, view.pitch, view.yaw), (30.0, 45.0, 0.0))

    def test_eye_stands_to_the_south_and_above(self):
        view = camera.Camera(target=(10.0, 20.0, 1.0))
        x, y, z = view.eye()
        self.assertAlmostEqual(x, 10.0)
        self.assertAlmostEqual(y, 20.0 - 30.0 * math.cos(math.radians(45.0)))
        self.assertAlmostEqual(z, 1.0 + 30.0 * math.sin(math.radians(45.0)))

    def test_pitch_is_limited(self):
        self.assertEqual(camera.clamp_pitch(-10.0), camera.MIN_PITCH)
        self.assertEqual(camera.clamp_pitch(89.0), camera.MAX_PITCH)
        view = camera.Camera()
        view.pitch_by(-100.0)
        self.assertEqual(view.pitch, camera.MIN_PITCH)
        view.pitch_by(100.0)
        self.assertEqual(view.pitch, camera.MAX_PITCH)

    def test_zoom_is_limited(self):
        self.assertEqual(camera.clamp_distance(1.0), camera.MIN_LIMIT_DISTANCE)
        self.assertEqual(camera.clamp_distance(200.0), camera.MAX_LIMIT_DISTANCE)
        view = camera.Camera(distance=200.0)
        self.assertEqual(view.distance, camera.MAX_LIMIT_DISTANCE)
        view.zoom_by(-1000.0)
        self.assertEqual(view.distance, camera.MIN_LIMIT_DISTANCE)
        view.zoom_by(1000.0)
        self.assertEqual(view.distance, camera.MAX_LIMIT_DISTANCE)

    def test_turn_has_no_limit(self):
        view = camera.Camera()
        view.rotate_by(720.0)
        self.assertEqual(view.yaw, 720.0)
        view.rotate_by(-1000.0)
        self.assertEqual(view.yaw, -280.0)

    def test_pan_moves_the_target_along_the_camera(self):
        view = camera.Camera(target=(0.0, 0.0, 0.0))
        view.pan(2.0, 3.0)
        self.assertAlmostEqual(view.target[0], 2.0)
        self.assertAlmostEqual(view.target[1], 3.0)
        # turned to the east the right of the camera is -y and the forward is +x
        view.yaw = 90.0
        view.pan(2.0, 3.0)
        self.assertAlmostEqual(view.target[0], 5.0)
        self.assertAlmostEqual(view.target[1], 1.0)

    def test_floor_of_the_eye(self):
        """
        40e4c0: the eye stays CameraMinHeight over the highest ground within 2 around it and over the tops of
        the objects near it; the top of an object fades over 10 from its circle.
        """
        land = Land()
        self.assertAlmostEqual(camera.floor_height(land, 0.0, 0.0), camera.MIN_HEIGHT)
        self.assertAlmostEqual(camera.floor_height(land, 8.0, 0.0), 6.0 + camera.MIN_HEIGHT)
        self.assertAlmostEqual(camera.floor_height(land, 7.5, 0.0), camera.MIN_HEIGHT)
        # a tree 12 high with a box of 2 x 2 at -20, 0
        corners = [(x, y, z) for x in (-21.0, -19.0) for y in (-1.0, 1.0) for z in (0.0, 12.0)]
        trees = camera.obstacles([[('trunk', corners)], []])
        self.assertEqual(trees.shape, (1, 4))
        radius = math.sqrt(2.0)
        for got, expected in zip(trees[0], (-20.0, 0.0, radius, 12.0)):
            self.assertAlmostEqual(got, expected)
        self.assertAlmostEqual(camera.floor_height(land, -20.5, 0.0, trees), 12.0 + camera.MIN_HEIGHT)
        self.assertAlmostEqual(camera.floor_height(land, -20.0 - radius - 5.0, 0.0, trees), 6.0 + camera.MIN_HEIGHT)
        self.assertAlmostEqual(camera.floor_height(land, -20.0 - radius - 10.0, 0.0, trees), camera.MIN_HEIGHT)
        # the rise over the objects is not more than CameraMaxHeight
        self.assertAlmostEqual(camera.floor_height(land, -20.5, 0.0, trees, max_height=5.0),
                               5.0 + camera.MIN_HEIGHT)
        self.assertEqual(len(camera.obstacles([])), 0)

    def test_eye_stays_above_the_floor(self):
        view = camera.Camera(target=(0.0, 20.0, 0.0), pitch=camera.MIN_PITCH, distance=camera.MIN_LIMIT_DISTANCE)
        low = view.eye()
        self.assertLess(low[2], camera.MIN_HEIGHT)
        view.terrain = Land()
        eye = view.eye()
        self.assertAlmostEqual(eye[2], camera.MIN_HEIGHT)
        self.assertAlmostEqual(eye[0], low[0])
        self.assertAlmostEqual(eye[1], low[1])
        # a high eye is left alone
        view.zoom_to(camera.DEFAULT_DISTANCE)
        view.pitch = camera.DEFAULT_PITCH
        self.assertGreater(view.eye()[2], camera.MIN_HEIGHT + 10.0)

    def test_keyboard_scroll(self):
        # 40f230 / 40fb30: the velocity comes to ScrollPowerKeyboard by 0.1333 a tick, the move grows with the zoom
        view = camera.Camera(target=(0.0, 0.0, 0.0), yaw=0.0, distance=camera.DEFAULT_DISTANCE)
        self.assertFalse(view.advance_scroll(1.0))
        view.scroll_keys(1, 0)
        view.advance_scroll(1.0)
        self.assertAlmostEqual(view.scroll_velocity[0], 0.5 * camera.SCROLL_SMOOTH)
        for _ in range(200):
            view.advance_scroll(1.0)
        self.assertAlmostEqual(view.scroll_velocity[0], 0.5)
        zoom = (camera.DEFAULT_DISTANCE - camera.MIN_DISTANCE) / (
            (camera.MAX_DISTANCE - camera.MIN_DISTANCE) * camera.SCROLL_ZOOM_SPAN) + camera.SCROLL_ZOOM_BASE
        x = view.target[0]
        view.advance_scroll(1.0)
        self.assertAlmostEqual(view.target[0] - x, 0.5 * zoom)
        self.assertAlmostEqual(view.target[1], 0.0)
        # released: the camera slows down and stands
        view.scroll_keys(0, 0)
        for _ in range(200):
            view.advance_scroll(1.0)
        self.assertEqual(view.scroll_velocity, [0.0, 0.0])
        self.assertFalse(view.advance_scroll(1.0))
        # the forward of a camera turned by 90 degrees is the east
        view.yaw = 90.0
        view.scroll_keys(0, 1)
        self.assertAlmostEqual(view.scroll_target[0], 0.5)
        self.assertAlmostEqual(view.scroll_target[1], 0.0)
