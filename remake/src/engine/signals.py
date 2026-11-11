"""
The mood signals of the units (CUnitSignal of game.exe 5aea80): a level raised by the logic of a unit plays
once for 20 ticks.

The levels of the game and their voices are the cases 1..9 of 5aea80: the idle sounds 1 and 2, the return
from the suspicion 3 and from the aggression 4, the way to the suspicion 5 and to the aggression 6, the way
to the alarm 7, the alarm 8 and the fight 9. acks.db has no groups for them (the groups 0x2e..0x32 of the
game are not in its records), so the remake reports the levels to Unit.on_signal and the sounds are chosen
by engine.acks.SIGNAL_GROUPS; the state of the signal itself is the same as 5aea80.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

# the levels of the signal (the cases of 5aea80)
IDLE_A = 1
IDLE_B = 2
BACK_FROM_SUSPICION = 3
BACK_FROM_AGGRESSION = 4
TO_SUSPICION = 5
TO_AGGRESSION = 6
TO_ALARM = 7
ALARM = 8
FIGHT = 9

# 5aea80: the signal plays for 0x14 ticks
SIGNAL_TICKS = 20

#------------------------------------------------------------------------------


class Signal(object):
    """
    The signal of one unit: the level waiting to be said, the level playing and the ticks left.
    """

    def __init__(self, ticks=SIGNAL_TICKS):
        self.ticks = int(ticks)
        self.pending = 0
        self.active = 0
        self.left = 0

    def __repr__(self):
        return 'Signal(%d, %d ticks left)' % (self.active, self.left)

    def raise_level(self, level):
        """
        CUnitSignal::Raise: the strongest waiting level wins.
        """
        if level > self.pending:
            self.pending = int(level)
            return True
        return False

    def update(self, ticks=1):
        """
        CUnitSignal::Update: the waiting level starts and plays for the ticks; returns it when it starts,
        0 otherwise.
        """
        started = 0
        if self.pending:
            started = self.active = self.pending
            self.pending = 0
            self.left = self.ticks
            return started
        if self.active:
            self.left -= max(int(ticks), 0)
            if self.left <= 0:
                self.active = 0
                self.left = 0
        return started
