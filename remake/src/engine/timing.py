"""
Time of the simulation, the same as the main loop of game.exe (sub_456440).

The world is updated by fixed ticks of 55 ms, the "clock" button of the battle makes it 27 ms.
The time of the animations is counted in frames and one frame lasts one tick: ~18.18 frames per second.
What is left of the real time after the last tick, divided by the tick, is the fraction
the renderer can use to place the objects between two ticks.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

TICK_MS = 55

FAST_TICK_MS = 27

ANIMATION_FPS = 1000.0 / TICK_MS

# CUnitClientSpecific::vf40 (0x535b30) never lets one step of the client be longer than this
MAX_STEP_MS = 500

#------------------------------------------------------------------------------


class TickClock(object):

    def __init__(self, tick_ms=TICK_MS):
        self.tick_ms = tick_ms
        self.accumulator_ms = 0.0
        self.ticks = 0

    @property
    def fraction(self):
        return self.accumulator_ms / self.tick_ms

    def set_fast(self, fast):
        self.tick_ms = FAST_TICK_MS if fast else TICK_MS

    def advance(self, dt_ms):
        """
        How many whole ticks fit into the elapsed time, the rest waits for the next call.
        """
        self.accumulator_ms += min(dt_ms, MAX_STEP_MS)
        count = 0
        while self.accumulator_ms >= self.tick_ms:
            self.accumulator_ms -= self.tick_ms
            self.ticks += 1
            count += 1
        return count

    def frames(self, dt_ms):
        """
        Animation frames in the elapsed real time: a frame is one tick of the world,
        so the "clock" button makes the animations faster too.
        """
        return dt_ms / self.tick_ms
