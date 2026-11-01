"""
Movement rules of the units, shared by the server and the client.

Server moves the units with these functions every tick. Client runs exactly the same functions to calculate where
the unit is between the ticks and during the ticks which were not received yet, so the unit drawn on the client
is always at the same position as on the server. Only the server knows about obstacles: when a unit is blocked
there, the client finds out from the next state and corrects the drawn position.
Module is plain Python without any dependencies, so it can be imported by the server and by the client.
"""

import math


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
    New direction of the unit after one tick.
    """
    if target_direction is not None:
        direction_diff = target_direction - direction
        if direction_diff > 180.0:
            direction_diff -= 360.0
        if direction_diff < -180.0:
            direction_diff += 360.0
        step = min(turn_speed, abs(direction_diff))
        return normalize_angle(direction + (-step if direction_diff > 0.0 else step))
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
    Position is not checked for obstacles here, that is done by the server.
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
