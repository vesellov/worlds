"""
The rain and the snow of a zone over time, as the server of game.exe runs them (58ccd0, 58dcd0, 4e40b0).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import random
import unittest

#------------------------------------------------------------------------------

from engine import weather

#------------------------------------------------------------------------------


class FixedRandom(random.Random):
    """
    randrange gives the numbers of a list, then always 1.
    """

    def __init__(self, numbers):
        random.Random.__init__(self, 5)
        self.numbers = list(numbers)

    def randrange(self, *args):
        return self.numbers.pop(0) if self.numbers else 1

#------------------------------------------------------------------------------


class ScheduleTest(unittest.TestCase):

    def run_ticks(self, schedule, count):
        strikes = []
        for _ in range(count):
            more, changed = schedule.tick()
            strikes.extend(more)
        return strikes

    def test_zone_opens_dry(self):
        schedule = weather.Schedule(weather.RAIN)
        self.assertEqual(schedule.falling, weather.NONE)
        self.assertEqual(schedule.level(), 0)

    def test_checks_every_64_ticks(self):
        schedule = weather.Schedule(weather.RAIN, rng=FixedRandom([0]))
        self.run_ticks(schedule, weather.CHECK_PHASE - 1)
        self.assertEqual(schedule.falling, weather.NONE)
        strikes, changed = schedule.tick()
        self.assertTrue(changed)
        self.assertEqual(schedule.falling, weather.RAIN)
        # the first strike comes with the check which starts the rain
        self.assertEqual(len(strikes), 1)
        self.assertEqual(schedule.counter, weather.HOLD_CHECKS)

    def test_fade_of_the_rain(self):
        schedule = weather.Schedule(weather.RAIN, rng=FixedRandom([0]))
        self.run_ticks(schedule, weather.CHECK_PHASE)
        self.assertEqual(schedule.level(), 0)
        self.run_ticks(schedule, 10)
        self.assertEqual(schedule.level(), weather.LEVELS // 2)
        self.run_ticks(schedule, 10)
        self.assertEqual(schedule.level(), weather.LEVELS)

    def test_snow_is_slower_and_has_no_lightning(self):
        schedule = weather.Schedule(weather.SNOW, rng=FixedRandom([0]))
        strikes = self.run_ticks(schedule, weather.CHECK_PHASE + 20)
        self.assertEqual(schedule.falling, weather.SNOW)
        self.assertEqual(schedule.level(), weather.LEVELS // 2)
        strikes += self.run_ticks(schedule, weather.CHECK_TICKS * 3)
        self.assertEqual(strikes, [])

    def test_hold_and_stop(self):
        # starts at once, then the counter holds the rain for 10 checks before the stop roll is taken
        schedule = weather.Schedule(weather.RAIN, rng=FixedRandom([0, 0]))
        strikes = self.run_ticks(schedule, weather.CHECK_PHASE + weather.CHECK_TICKS * (weather.HOLD_CHECKS - 1))
        self.assertEqual(schedule.falling, weather.RAIN)
        self.assertEqual(len(strikes), weather.HOLD_CHECKS)
        self.run_ticks(schedule, weather.CHECK_TICKS)
        self.assertEqual(schedule.falling, weather.NONE)
        self.assertEqual(schedule.level(), weather.LEVELS)
        self.run_ticks(schedule, 20)
        self.assertEqual(schedule.level(), 0)

    def test_strike_points(self):
        schedule = weather.Schedule(weather.RAIN, rng=random.Random(3))
        for _ in range(100):
            (sx, sy, sz), (x, y, z) = schedule.strike_points()
            self.assertTrue(2.0 <= x <= 510.0 and 2.0 <= y <= 510.0)
            self.assertTrue(abs(sx - x) <= 2.0 and abs(sy - y) <= 2.0)
            self.assertEqual((sz, z), (35.0, -5.0))

    def test_tornado_and_none_do_not_cycle(self):
        self.assertFalse(weather.Schedule(weather.TORNADO).cycles)
        self.assertFalse(weather.Schedule(weather.NONE).cycles)

    def test_level_hides_the_drops(self):
        w = weather.Weather(weather.RAIN, (50.0, 50.0, 0.0), rng=random.Random(1), level=0)
        self.assertEqual(len(w.precipitation.triangles((50.0, 50.0, 20.0), 100.0)), 0)
        w.set_level(weather.LEVELS)
        self.assertGreater(len(w.precipitation.triangles((50.0, 50.0, 20.0), 100.0)), 0)

if __name__ == '__main__':
    unittest.main()
