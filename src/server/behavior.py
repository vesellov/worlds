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

# a unit going along a road turns to the next point of the road when it is that close to the current one, in segments,
# it does not stop at the points of the road. A unit not further than ROAD_PASS_DISTANCE from the point which is
# already ahead of it along the road also turns to the next one, so a fast unit does not go around the point
ROAD_POINT_DISTANCE = 0.5
ROAD_PASS_DISTANCE = 1.5

# a unit going along a road which in STUCK_TICKS ticks came closer to its point by less than STUCK_SHARE of the way
# it was going to make is stuck: in front of a tree, a flag on the road or other units. It goes aside at DETOUR_ANGLE
# degrees from the point for DETOUR_TICKS ticks, to the left and to the right by turns, and goes to the next point
# of the road after STUCK_SKIP tries. Stuck at the last point it stands: other units stand there already
STUCK_TICKS = 40
STUCK_SHARE = 0.2
STUCK_SKIP = 4
DETOUR_ANGLE = 70.0
DETOUR_TICKS = 25

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

#------------------------------------------------------------------------------


class UnitRoadMove(UnitBehavior):
    """
    The unit goes along the road to one of its ends: to the last point of the road when to_end is True,
    to the first one otherwise, or to the point to_index of the road. It goes to the nearest point of the road first,
    then along the road from point to point, see server.roads. When it came there it stands, or does next_behavior.
    """

    action = 'walk'

    def __init__(self, road_id, to_end=True, to_index=None, next_behavior=None):
        self.road_id = road_id
        self.to_index = to_index if to_index is not None else (-1 if to_end else 0)
        self.next_behavior = next_behavior
        # points the unit goes through, see roads.Roads.way(), they are found at the first tick
        self.points = None
        # how the unit came closer to the current point in the last STUCK_TICKS ticks, see is_stuck()
        self.window_distance = None
        self.window_way = 0.0
        self.window_ticks = 0
        self.stuck_count = 0
        self.detour_ticks = 0
        self.detour_side = 1.0

    def think(self, sim, unit):
        x = unit.w + unit.shift_w
        y = unit.h + unit.shift_h
        if self.points is None:
            if not sim.roads or self.road_id not in sim.roads.roads:
                if _Debug:
                    print(f'{unit.name} has no road {self.road_id}')
                return self.arrived(sim, unit)
            self.points = sim.roads.way(self.road_id, x, y, self.to_index)
        # passes the points it came to, it does not stop there
        while len(self.points) > 1 and self.passed(sim, x, y):
            self.next_point()
        last = len(self.points) == 1
        # the unit does not stop between two roads when the next behavior goes along a road too
        passing = last and isinstance(self.next_behavior, UnitRoadMove)
        dx = nearest_shift(x, self.points[0][0], sim.width)
        dy = nearest_shift(y, self.points[0][1], sim.height)
        distance = math.sqrt(dx * dx + dy * dy)
        if last and distance <= (ROAD_POINT_DISTANCE if passing else ARRIVE_DISTANCE):
            return self.arrived(sim, unit)
        speed = sim.unit_speed(unit, self.action)
        if last and not passing:
            # slower near the end of the way, so the unit does not run past it
            speed *= max(0.25, min(1.0, (distance - ARRIVE_DISTANCE) / SLOW_DOWN_DISTANCE))
        if self.is_stuck(distance, speed):
            self.stuck_count += 1
            if _Debug:
                print(f'{unit.name} is stuck {distance:.2f} from the point of road {self.road_id} at {x:.2f},{y:.2f}, {len(self.points)} points left, try {self.stuck_count}')
            if last:
                return self.arrived(sim, unit)
            if self.stuck_count >= STUCK_SKIP:
                self.next_point()
                return self.think(sim, unit)
            self.detour_ticks = DETOUR_TICKS
            self.detour_side = -self.detour_side
        if self.detour_ticks > 0:
            self.detour_ticks -= 1
            angle = math.radians(DETOUR_ANGLE * self.detour_side)
            dx, dy = dx * math.cos(angle) - dy * math.sin(angle), dx * math.sin(angle) + dy * math.cos(angle)
        self.go(unit, dx, dy, speed)
        return self

    def passed(self, sim, x, y):
        """
        The unit came close to the current point, or it is near the point and the point is behind it along the road.
        """
        (px, py), (nx, ny) = self.points[0], self.points[1]
        ux = nearest_shift(px, x, sim.width)
        uy = nearest_shift(py, y, sim.height)
        distance2 = ux * ux + uy * uy
        if distance2 <= ROAD_POINT_DISTANCE * ROAD_POINT_DISTANCE:
            return True
        if distance2 > ROAD_PASS_DISTANCE * ROAD_PASS_DISTANCE:
            return False
        return ux * nearest_shift(px, nx, sim.width) + uy * nearest_shift(py, ny, sim.height) > 0.0

    def is_stuck(self, distance, speed):
        """
        Every STUCK_TICKS ticks checks how much closer to the current point the unit came, speed is how far it wanted to go in this tick.
        """
        if self.window_distance is None:
            self.window_distance = distance
            self.window_way = 0.0
            self.window_ticks = 0
            return False
        self.window_way += speed
        self.window_ticks += 1
        if self.window_ticks < STUCK_TICKS:
            return False
        stuck = self.window_distance - distance < STUCK_SHARE * self.window_way
        self.window_distance = None
        return stuck

    def next_point(self):
        self.points.pop(0)
        self.window_distance = None
        self.stuck_count = 0
        self.detour_ticks = 0

    def arrived(self, sim, unit):
        # the next behavior starts in the same tick, so the unit does not stop between two roads
        return (self.next_behavior or UnitIdle()).think(sim, unit)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.road_id}, {self.to_index})'

#------------------------------------------------------------------------------


class UnitRoadWalk(UnitRoadMove):

    action = 'walk'

#------------------------------------------------------------------------------


class UnitRoadRun(UnitRoadMove):

    action = 'run'

#------------------------------------------------------------------------------


class UnitRoadToCapitalWalk(UnitBehavior):
    """
    The unit walks along the roads to the capital number capital_number of the biome ("i" of buildings.json)
    and stands on the road next to it. It goes to the nearest road first, then along the shortest way through the roads,
    see roads.Roads.find_way(), along every road with UnitRoadWalk. It stands where it is when the roads do not lead there.
    """

    action = 'walk'

    def __init__(self, capital_number):
        self.capital_number = capital_number

    def think(self, sim, unit):
        capital = sim.capitals.get(self.capital_number)
        legs = None
        if capital and sim.roads:
            legs = sim.roads.find_way(unit.w + unit.shift_w, unit.h + unit.shift_h, capital[0], capital[1])
        if not legs:
            if _Debug:
                print(f'{unit.name} found no way along the roads to the capital {self.capital_number}')
            return UnitIdle().think(sim, unit)
        if _Debug:
            print(f'{unit.name} goes to the capital {self.capital_number} along the roads {legs}')
        next_behavior = None
        for road_id, to_index in reversed(legs):
            next_behavior = UnitRoadWalk(road_id, to_index=to_index, next_behavior=next_behavior)
        return next_behavior.think(sim, unit)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.capital_number})'
