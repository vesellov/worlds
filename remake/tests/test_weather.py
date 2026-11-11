"""
Tests of engine.weather: the kind of the weather of a zone and the particles of the rain, the snow and
the drops of the rain and the snow on the grid around the eye, the particles of the tornado.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import random
import unittest

#------------------------------------------------------------------------------

from engine import weather

#------------------------------------------------------------------------------


class FakeZone(object):

    def __init__(self, kind, is_cave=False):
        self.weather = kind
        self.is_cave = is_cave


class WeatherTest(unittest.TestCase):

    def test_kind_of_a_zone(self):
        self.assertEqual(weather.from_zone(FakeZone('rain')), weather.RAIN)
        self.assertEqual(weather.from_zone(FakeZone('snow')), weather.SNOW)
        self.assertEqual(weather.from_zone(FakeZone('tornado')), weather.TORNADO)
        self.assertEqual(weather.from_zone(FakeZone('rain', is_cave=True)), weather.NONE)
        self.assertEqual(weather.from_zone(FakeZone('unknown')), weather.RAIN)
        self.assertEqual(weather.from_zone(None), weather.NONE)
        self.assertEqual(weather.Weather('unknown', (0.0, 0.0, 0.0)).kind, weather.NONE)

    def test_none_has_no_particles(self):
        value = weather.Weather(weather.NONE, (0.0, 0.0, 0.0), rng=random.Random(1))
        self.assertFalse(value.active)
        self.assertEqual(value.emitters, [])
        value.update((1.0, 2.0, 3.0))
        self.assertEqual(value.emitters, [])

    def test_rain_drops_around_the_eye(self):
        value = weather.Weather(weather.RAIN, (10.0, 20.0, 5.0), rng=random.Random(2))
        self.assertTrue(value.active)
        self.assertEqual(value.emitters, [])
        rain = value.precipitation
        self.assertEqual(rain.texture_name, weather.TEXTURE_NAME)
        eye = (10.0, 20.0, 25.0)
        x, y, z, fall, across = rain.drops(eye, 1000.0)
        # 81 x 81 cells, 0.5 apart, every one shown at the full density
        self.assertEqual(len(x), 81 * 81)
        self.assertTrue(abs(x - 10.0).max() <= 20.0 + 1.0 + 1e-6)
        self.assertTrue(abs(y - 20.0).max() <= 20.0 + 1.0 + 1e-6)
        self.assertTrue(z.min() >= eye[2] - weather.BAND - weather.DROP_SHIFT_Z - 1e-6)
        self.assertTrue(z.max() <= eye[2] + weather.BAND + weather.DROP_SHIFT_Z + 1e-6)

    def test_rain_falls(self):
        rain = weather.Precipitation(weather.RAIN)
        eye = (0.0, 0.0, 25.0)
        z0 = rain.drops(eye, 1000.0)[2]
        z1 = rain.drops(eye, 1001.0)[2]
        # most drops go down by 0.5 in a tick, a few wrap back to the top of the band
        self.assertGreater(((z0 - z1) == 0.5).mean(), 0.9)

    def test_drop_is_a_thin_streak(self):
        rain = weather.Precipitation(weather.RAIN)
        vertices = rain.triangles((0.0, 0.0, 25.0), 1000.0)
        self.assertEqual(vertices.shape, (81 * 81 * 3, 9))
        tri = vertices[:3]
        width = math.hypot(tri[0][0] - tri[1][0], tri[0][1] - tri[1][1])
        self.assertAlmostEqual(width, 2 * weather.DROP_HALF_WIDTH, places=5)
        self.assertAlmostEqual(tri[2][2] - tri[0][2], weather.DROP_LENGTH, places=5)
        # one of the four streaks of the bottom row of rainsnow
        self.assertEqual((tri[0][4], tri[2][4]), (0.75, 1.0))
        self.assertAlmostEqual(tri[1][3] - tri[0][3], 0.25)

    def test_wind_leans_the_rain(self):
        rain = weather.Precipitation(weather.RAIN)
        rain.wind = (0.2, 0.0, 0.0)
        tri = rain.triangles((0.0, 0.0, 25.0), 1000.0)[:3]
        # the top stays behind against the wind
        self.assertLess(tri[2][0], (tri[0][0] + tri[1][0]) * 0.5)

    def test_density(self):
        rain = weather.Precipitation(weather.RAIN, level=0)
        self.assertEqual(len(rain.triangles((0.0, 0.0, 25.0), 1000.0)), 0)
        rain.level = 32
        shown = len(rain.drops((0.0, 0.0, 25.0), 1000.0)[0])
        self.assertTrue(0.35 < shown / (81.0 * 81.0) < 0.65)

    def test_snow(self):
        snow = weather.Weather(weather.SNOW, (0.0, 0.0, 0.0)).precipitation
        vertices = snow.triangles((0.0, 0.0, 25.0), 5000.0)
        self.assertGreater(len(vertices), 0)
        self.assertEqual(vertices[0][4], 0.5)

    def test_tornado_particles(self):
        value = weather.Weather(weather.TORNADO, (0.0, 0.0, 4.0), rng=random.Random(3), ground=lambda x, y: 0.0)
        emitter = value.emitters[0]
        self.assertIsNone(value.precipitation)
        self.assertEqual(len(emitter.particles), weather.COUNTS[weather.TORNADO])
        for _ in range(200):
            value.update((0.0, 0.0, 4.0))
        self.assertEqual(len(emitter.particles), weather.COUNTS[weather.TORNADO])
        for particle in emitter.particles:
            self.assertGreaterEqual(particle.z, 0.0)
            self.assertLessEqual(particle.z, weather.HEIGHT + weather.GROUND_MARGIN + 1e-6)

if __name__ == '__main__':
    unittest.main()
