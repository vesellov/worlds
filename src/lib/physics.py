"""
Movement rules of the units, shared by the server and the client.

Server moves the units with these functions every tick. Client runs exactly the same functions to calculate where
the unit is between the ticks and during the ticks which were not received yet, so the unit drawn on the client
is always at the same position as on the server. Both know the zones where units can walk, see lib.zone,
so the client knows in advance where its hero stops or slides along a wall. Only the server knows where
the other units are going to be: when a unit is blocked by them, the client finds out from the next state
and corrects the drawn position.
Module is plain Python without any dependencies, so it can be imported by the server and by the client.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

#------------------------------------------------------------------------------

# how many times in one tick a unit which ran into obstacles is pushed out of them before it stops, see slide()
SLIDE_PUSHES = 8

# a unit pushed out of an obstacle stays that far from its border, in segments
SLIDE_GAP = 0.0001

# a unit pushed out of obstacles still moves when the angle between its movement and the direction it looks to
# has a cosine larger than that, a bit backwards too: sliding along a wavy border goes back and forth a little
SLIDE_MIN_COSINE = -0.5

#------------------------------------------------------------------------------


def normalize_angle(a):
    while a > 360.0:
        a -= 360.0
    while a < 0.0:
        a += 360.0
    return a


def normalize_position(w, h, shift_w, shift_h):
    # integer part of the shift moves unit to another segment
    if shift_w > 1.0:
        w += int(shift_w)
        shift_w = float(shift_w) - float(int(shift_w))
    elif shift_w < 0.0:
        w += int(shift_w) - 1
        shift_w = float(shift_w) - float(int(shift_w)) + 1.0
    if shift_h > 1.0:
        h += int(shift_h)
        shift_h = float(shift_h) - float(int(shift_h))
    elif shift_h < 0.0:
        h += int(shift_h) - 1
        shift_h = float(shift_h) - float(int(shift_h)) + 1.0
    return w, h, shift_w, shift_h


def turn(direction, turn_speed, controls_turn, target_direction):
    """
    New direction of the unit after one tick: it turns to target_direction the shorter way, not faster
    than turn_speed degrees, or keeps turning while controls_turn is pressed.
    """
    if target_direction is not None:
        direction_diff = (target_direction - direction) % 360.0
        if direction_diff > 180.0:
            direction_diff -= 360.0
        step = min(turn_speed, abs(direction_diff))
        return normalize_angle(direction + (step if direction_diff > 0.0 else -step))
    if controls_turn:
        return normalize_angle(direction + (turn_speed if controls_turn > 0 else -turn_speed))
    return direction


def accelerate(speed, move, max_speed, acceleration_value, deceleration_value):
    """
    New speed of the unit after one tick: it speeds up while the move button is pressed and slows down
    by deceleration_value every tick after the button was released.
    """
    if move > 0:
        acceleration_up, acceleration_down = acceleration_value, 0.0
    elif move < 0:
        acceleration_up, acceleration_down = -acceleration_value, 0.0
    else:
        acceleration_up, acceleration_down = 0.0, deceleration_value
    speed = speed + acceleration_up
    if acceleration_up >= 0:
        speed = min(speed, max_speed)
    else:
        speed = max(speed, -max_speed)
    if speed > 0:
        speed = max(0.0, speed - acceleration_down)
    elif speed < 0:
        speed = min(0.0, speed + acceleration_down)
    return round(speed, 6)


def step(w, h, shift_w, shift_h, direction, speed, move, controls_turn, target_direction,
         max_speed, acceleration_value, deceleration_value, turn_speed):
    """
    Calculates one tick of the unit movement, returns (w, h, shift_w, shift_h, direction, speed).
    Position is not checked for obstacles here, see move().
    """
    direction = turn(direction, turn_speed, controls_turn, target_direction)
    speed = accelerate(speed, move, max_speed, acceleration_value, deceleration_value)
    if speed:
        w, h, shift_w, shift_h = normalize_position(
            w,
            h,
            shift_w + speed * math.cos(math.radians(direction)),
            shift_h + speed * math.sin(math.radians(direction)),
        )
    return w, h, shift_w, shift_h, direction, speed


def is_blocked(zone, x, y, radius, obstacle=None):
    """
    A unit with that radius in segments can not stand at the position: it is in the closed area of zone,
    see lib.zone.ZoneMap, or obstacle(x, y, heading, from_x, from_y) returns a push out of something else there.
    """
    if zone and not zone.is_open(x, y, radius):
        return True
    return bool(obstacle and obstacle(x, y, None, None, None))


def slide(zone, from_x, from_y, w, h, shift_w, shift_h, heading, radius, obstacle=None):
    """
    Unit wants to move from (from_x, from_y) to the given position. When an obstacle is on the way the unit is pushed
    out of it along its border, so it keeps running along a wall or around a tree, slower because only
    the part of the movement along the obstacle is left. Returns the new position or None when the unit can not move.
    heading is (cos, sin) of the direction the unit moves to, see is_blocked() about zone, radius and obstacle.
    """
    for _ in range(SLIDE_PUSHES):
        x = w + shift_w
        y = h + shift_h
        push = zone.push(x, y, radius, SLIDE_GAP, from_x, from_y) if zone else None
        if push is None and obstacle:
            push = obstacle(x, y, heading, from_x, from_y)
        if push is None:
            move_x = x - from_x
            move_y = y - from_y
            length = math.sqrt(move_x * move_x + move_y * move_y)
            if length < SLIDE_GAP:
                return None
            # obstacles may push the unit back, but it does not run backwards
            if (move_x * heading[0] + move_y * heading[1]) / length < SLIDE_MIN_COSINE:
                return None
            return w, h, shift_w, shift_h
        w, h, shift_w, shift_h = normalize_position(w, h, shift_w + push[0], shift_h + push[1])
    return None


def move(zone, w, h, shift_w, shift_h, direction, speed, move, controls_turn, target_direction,
         max_speed, acceleration_value, deceleration_value, turn_speed, radius, obstacle=None, steer=None):
    """
    One tick of the unit movement with the obstacles, returns (w, h, shift_w, shift_h, direction, speed) as step() does.
    A unit which can not move stops: its speed becomes 0.0. See is_blocked() about zone, radius and obstacle.
    steer is (dx, dy) in segments added to the movement of this tick: the server turns the units aside
    to avoid other moving units, the client does not know it and finds out from the next state.
    """
    new_w, new_h, new_shift_w, new_shift_h, direction, new_speed = step(
        w, h, shift_w, shift_h, direction, speed, move, controls_turn, target_direction,
        max_speed, acceleration_value, deceleration_value, turn_speed,
    )
    if not new_speed:
        return w, h, shift_w, shift_h, direction, new_speed
    if steer:
        new_w, new_h, new_shift_w, new_shift_h = normalize_position(new_w, new_h, new_shift_w + steer[0], new_shift_h + steer[1])
    position = (new_w, new_h, new_shift_w, new_shift_h)
    from_x = w + shift_w
    from_y = h + shift_h
    # a unit which somehow is already inside of a blocked area must be able to walk out of it
    if not is_blocked(zone, from_x, from_y, radius, obstacle):
        # the unit moves backwards when the speed is negative
        sign = 1.0 if new_speed > 0 else -1.0
        heading = (sign * math.cos(math.radians(direction)), sign * math.sin(math.radians(direction)))
        position = slide(zone, from_x, from_y, new_w, new_h, new_shift_w, new_shift_h, heading, radius, obstacle)
    if not position:
        return w, h, shift_w, shift_h, direction, 0.0
    return position[0], position[1], position[2], position[3], direction, new_speed
