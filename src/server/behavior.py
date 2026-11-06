"""
Behavior of the neutral units on the server: units which are not controlled by any client.

Every tick before the movement the simulation asks the behavior of the unit what the unit wants to do,
see Simulation.run_behaviors(). The behavior only sets the controls of the unit, the same ones a client sets
for its hero: move, turn, target_direction, and also how fast the unit goes (max_speed) and the action
it plays while moving ('walk' or 'run'). The unit is moved with the usual rules after that, see lib.physics,
so it is stopped by the closed zones, trees, buildings and other units the same way as a hero.
The clients get the controls with the state of the unit and calculate its movement between the ticks.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

#------------------------------------------------------------------------------

# a unit going to a point stops when it is that close to the point, in segments
ARRIVE_DISTANCE = 0.15

# a unit going to a point slows down when it is closer to the point than that, in segments,
# a unit following a moving point keeps behind it instead of stopping and starting again all the time
SLOW_DOWN_DISTANCE = 1.0

# a unit following another unit stops that far from it, in segments
FOLLOW_DISTANCE = 1.0

# a unit running to a capital stops that far from its center, the capital itself is closed
CAPITAL_DISTANCE = 1.5

# the point of UnitWalkInCircle goes around the circle a bit slower than the unit walks
CIRCLE_POINT_SPEED = 0.8

#------------------------------------------------------------------------------


def nearest_shift(a, b, size):
    """
    Shortest way from a to b along one axis of the map, the map is wrapped around.
    """
    d = (b - a) % size
    if d > size / 2.0:
        d -= size
    return d

#------------------------------------------------------------------------------


class UnitBehavior(object):
    """
    Base class of the behaviors. think() is called every tick and sets the controls of the unit,
    it returns the behavior for the next tick: the same one, or another one when this one is finished.
    """

    # what the unit plays while it moves, the simulation chooses the animation, see Simulation.animation_for(),
    # and the speed of the unit follows from that animation, see Simulation.unit_speed()
    action = 'idle'

    def think(self, sim, unit):
        return self

    def stop(self, unit):
        unit.move = 0
        unit.turn = 0
        unit.target_direction = None

    def go(self, unit, dx, dy, speed):
        # turns to (dx, dy) from the unit and moves forward, the unit turns while it moves
        unit.move = 1
        unit.turn = 0
        unit.target_direction = round(math.degrees(math.atan2(dy, dx)) % 360.0, 1)
        unit.max_speed = round(speed, 4)

    def __repr__(self):
        return self.__class__.__name__

#------------------------------------------------------------------------------


class UnitIdle(UnitBehavior):
    """
    The unit stands where it is.
    """

    def think(self, sim, unit):
        self.stop(unit)
        return self

#------------------------------------------------------------------------------


class UnitGoToPoint(UnitBehavior):
    """
    The unit goes to the point (x, y) of the map and stands there, see UnitRunToPoint and UnitWalkToPoint.
    """

    action = 'walk'
    arrive_distance = ARRIVE_DISTANCE

    def __init__(self, x, y, next_behavior=None):
        self.x = x
        self.y = y
        # what the unit does when it came to the point, it stands there by default
        self.next_behavior = next_behavior

    def target(self, sim, unit):
        """
        Where the unit goes now, subclasses move the point.
        """
        return self.x, self.y

    def think(self, sim, unit):
        x, y = self.target(sim, unit)
        dx = nearest_shift(unit.w + unit.shift_w, x, sim.width)
        dy = nearest_shift(unit.h + unit.shift_h, y, sim.height)
        distance = math.sqrt(dx * dx + dy * dy)
        if distance <= self.arrive_distance:
            self.stop(unit)
            return self.arrived(sim, unit)
        # slower near the point, so the unit does not run past it
        speed = sim.unit_speed(unit, self.action)
        k = min(1.0, (distance - self.arrive_distance) / SLOW_DOWN_DISTANCE)
        self.go(unit, dx, dy, speed * max(k, 0.25))
        return self

    def arrived(self, sim, unit):
        return self.next_behavior or UnitIdle()

#------------------------------------------------------------------------------


class UnitRunToPoint(UnitGoToPoint):

    action = 'run'

#------------------------------------------------------------------------------


class UnitWalkToPoint(UnitGoToPoint):

    action = 'walk'

#------------------------------------------------------------------------------


class UnitWalkToUnit(UnitWalkToPoint):
    """
    The unit walks after another unit and stays FOLLOW_DISTANCE from it, it stands when the other unit is gone.
    """

    arrive_distance = FOLLOW_DISTANCE

    def __init__(self, target_id):
        super().__init__(0.0, 0.0)
        self.target_id = target_id

    def target(self, sim, unit):
        o = sim.objects.get(self.target_id)
        if o is None:
            return unit.w + unit.shift_w, unit.h + unit.shift_h
        return o.w + o.shift_w, o.h + o.shift_h

    def think(self, sim, unit):
        if self.target_id not in sim.objects:
            self.stop(unit)
            return UnitIdle()
        return super().think(sim, unit)

    def arrived(self, sim, unit):
        # keeps following when the other unit goes away again
        return self

#------------------------------------------------------------------------------


class UnitWalkInCircle(UnitWalkToPoint):
    """
    The unit walks after a point which goes around the circle with that center and radius.
    """

    def __init__(self, center_x, center_y, radius, angle=0.0, clockwise=False):
        super().__init__(center_x, center_y)
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        # radians
        self.angle = angle
        self.direction = -1.0 if clockwise else 1.0

    def target(self, sim, unit):
        return self.center_x + self.radius * math.cos(self.angle), self.center_y + self.radius * math.sin(self.angle)

    def think(self, sim, unit):
        angle_step = sim.unit_speed(unit, self.action) * CIRCLE_POINT_SPEED / self.radius
        self.angle = (self.angle + angle_step * self.direction) % (2.0 * math.pi)
        return super().think(sim, unit)

    def arrived(self, sim, unit):
        # the point goes further and the unit follows it again
        return self

#------------------------------------------------------------------------------


class UnitRunToCapital(UnitRunToPoint):
    """
    The unit runs to the nearest capital of the biome and stands near it, it stands where it is when there are no capitals.
    """

    arrive_distance = CAPITAL_DISTANCE

    def __init__(self):
        super().__init__(0.0, 0.0)
        self.capital = None

    def think(self, sim, unit):
        if self.capital is None:
            self.capital = sim.nearest_capital(unit.w + unit.shift_w, unit.h + unit.shift_h)
            if self.capital is None:
                self.stop(unit)
                return UnitIdle()
            self.x, self.y = self.capital
        return super().think(sim, unit)
