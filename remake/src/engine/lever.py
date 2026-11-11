"""
The levers of a map (CLeverObject): gates, doors, switches, keys. A lever has LEVER_TOTAL_STATE states and
shows its state by the first body proportion of its figure (OBJCOMPLECTION): the phase
state / (states - 1) moves the 8 variants of the .fig, so a gate opens without an animation clip
(52b3f0 sets the phase from the state, CLeverObject::vf06 morphs the figure by 517700).

The switch (52b7a0, SwitchLeverState of the scripts with the state and the time -10):
- a state below 0 takes the next one: round for LEVER_IS_CYCLED (+0x240), not above the last one otherwise;
  any other state is kept within 0..states - 1;
- with a time below 0 the time comes from a table by the name of the model (464750), 10 when it has none
  (the remake always takes 10 ticks);
- the logic has the new phase at once (CFigure::PartPassFlags, then the grid around the lever is built
  again by 5b8b30 / 5b8b80 and the units which stand in it are pushed out by 5ba9d0);
- the view goes from the old phase to the new one: the end is now + time + 1 (+600), the fraction left is
  (end - now) / time (+0x25c), the phase is to - (to - from) * fraction while the fraction is above 0.

EnableLever(o, 0) sets the flag 0x80000 (+8) which forbids to use the lever, EnableLever(o, 1) clears it (52bad0).
GetLeverState reads +0x238.

The use by a unit: LEVER_SCIENCE_STATS_NEW (+0x244) is (the flags, the number of the key, the level).
A lever can be used (vf38, 52bb00) when it is not forbidden and its flags are not 0, so a lever of (0, 0, 0)
is moved only by the scripts. The check 522a80 compares the science of the user with it: the common flags
must not be 0; with the flag 8 the user needs the key of the same number, with the flags 1, 2, 4 a level
not below the level of the lever. The use (vf41) takes the next state and forbids the lever (0x80000).
The remake's hero passes the checks of the flags 1, 2 and 4 and has no keys (no inventory yet).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

# 52b7a0: the time of the switch when the table has nothing for the model
SWITCH_TICKS = 10.0

# 522a80: the flag of a key, the flags of the levels
KEY_FLAG = 8
LEVEL_FLAGS = 7

#------------------------------------------------------------------------------


class Lever(object):

    def __init__(self, state=0, states=2, cycled=False, coefs=(0.0, 0.5, 0.5), is_door=False, science=(0, 0, 0)):
        self.states = states
        self.science = tuple(science)
        self.cycled = cycled
        self.is_door = is_door
        self.enabled = True
        self.state = 0
        self.coefs = list(coefs)
        self.set_state(state)
        # the morph of the view: from the phase, to the phase, the ticks left, the length
        self.phase_from = self.phase
        self.phase_to = self.phase
        self.ticks_left = 0.0
        self.switch_ticks = SWITCH_TICKS
        self.shown_phase = self.phase

    @classmethod
    def from_object(cls, obj):
        """
        The lever of an engine.world.MapObject of the kind LEVER.
        """
        return cls(obj.lever_state, obj.lever_states, obj.lever_cycled, obj.complection, obj.lever_is_door,
                   obj.lever_science)

    def __repr__(self):
        return 'Lever(%d/%d%s)' % (self.state, self.states, '' if self.enabled else ' disabled')

    @property
    def phase(self):
        """
        The phase of the logic: the state over the last state, 0 for a lever of one state.
        """
        if self.states < 2:
            return 0.0
        return float(self.state) / float(self.states - 1)

    def set_state(self, state):
        last = self.states - 1
        if state < 0:
            if self.cycled and self.states > 0:
                state = (self.state + 1) % self.states
            else:
                state = min(self.state + 1, last)
        else:
            state = min(state, last)
        self.state = max(0, state)

    def switch(self, state=-1, ticks=None):
        """
        52b7a0: the new state; the view morphs to it for the ticks (SWITCH_TICKS by default).
        """
        ticks = SWITCH_TICKS if ticks is None else ticks
        self.phase_from = self.shown_phase
        self.set_state(state)
        self.phase_to = self.phase
        if ticks <= 0:
            self.ticks_left = 0.0
            self.shown_phase = self.phase_to
        else:
            self.switch_ticks = ticks
            self.ticks_left = ticks + 1.0
        if _Debug:
            print('%s switched' % self)

    def update(self, ticks):
        """
        CLeverObject::vf06: the phase of the view after the ticks; True while it changes.
        """
        if self.ticks_left <= 0.0:
            self.shown_phase = self.phase_to
            return False
        self.ticks_left -= ticks
        fraction = self.ticks_left / self.switch_ticks
        if fraction > 0.0:
            self.shown_phase = self.phase_to - (self.phase_to - self.phase_from) * fraction
        else:
            self.ticks_left = 0.0
            self.shown_phase = self.phase_to
        return True

    @property
    def usable(self):
        """
        vf38: not forbidden and with flags of the science.
        """
        return self.enabled and self.science[0] != 0

    def allows(self, science):
        """
        522a80: the science (flags, keys, level) of the user against the one of the lever; keys is a collection
        of the numbers of the keys.
        """
        flags, keys, level = science
        common = flags & self.science[0]
        if not common:
            return False
        if common & KEY_FLAG and self.science[1] in keys:
            return True
        if common & LEVEL_FLAGS:
            return self.science[2] <= level
        return False

    def use(self, science):
        """
        vf41: the next state and the lever is forbidden; False when the science does not allow it.
        """
        if not self.usable or not self.allows(science):
            return False
        self.switch()
        self.enabled = False
        return True

    def view_coefs(self):
        """
        The body proportions of the figure as the view shows it now.
        """
        return (self.shown_phase, self.coefs[1], self.coefs[2])

    def logic_coefs(self):
        """
        The body proportions of the logic: the grid of the paths takes them.
        """
        return (self.phase, self.coefs[1], self.coefs[2])
