"""
The part of the body of a unit under the mouse: the ray of the screen against the boxes of the parts
of the figure (engine.figure.FigureModel, eilib.fig.Figure.variant_bounds).

game.exe keeps six cursors of the hit locations (cursor_attack_hd/bd/lh/rh/ll/rl, LoadCursors 5daea0)
and chooses the part under the mouse; its own test is not found. The remake takes the box of a
representative part of every hit location of a unit (skull, torso, the hands, the shins), moves the ray
into the space of the part by the pose of the skeleton (the same matrices as client.scene.UnitView) and
takes the nearest box which the ray crosses. The numbers of the hit locations are the order of
engine.combat: 0 skull, 1 torso, 2 left arm, 3 right arm, 4 left leg, 5 right leg.

The called shot ("cs_head", "cs_lhand"... of config/keyboard.ini) is kept by the client in
app.aim_part; client.main sets it on the unit before the order of the attack, and engine.unit builds
the blow with that location and the aim (Unit.aim_part, combat.make_hit).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import numpy

#------------------------------------------------------------------------------

from engine import quat
from client import gl

#------------------------------------------------------------------------------

# hit location of engine.combat -> the parts of the skeleton which stand for it
PART_NAMES = {
    0: ('hd',),
    1: ('bd', 'hp'),
    2: ('lh3', 'lh2'),
    3: ('rh3', 'rh2'),
    4: ('ll2', 'll3'),
    5: ('rl2', 'rl3'),
}

# the cursors of the hit locations, the order of LoadCursors (5daea0)
PART_CURSORS = {
    0: 'cursor_attack_hd',
    1: 'cursor_attack_bd',
    2: 'cursor_attack_lh',
    3: 'cursor_attack_rh',
    4: 'cursor_attack_ll',
    5: 'cursor_attack_rl',
}

# the names of the cursors of the called shot by the hit location (the "cs_*" of keyboard.ini)
CALLED_SHOT_NAMES = {
    'cs_head': 0,
    'cs_body': 1,
    'cs_lhand': 2,
    'cs_rhand': 3,
    'cs_lleg': 4,
    'cs_rleg': 5,
}

# the cursor of an enemy when no called shot is chosen
WHOLE_UNIT_CURSOR = 'cursor_attack'

#------------------------------------------------------------------------------


def screen_ray(app, x, y):
    """
    (origin, direction) of the ray of the screen point in the world; x and y are the pixels of the window.
    """
    scene = app.scene
    nx = 2.0 * (x - scene.x) / max(scene.width, 1) - 1.0
    ny = 2.0 * (y - scene.y) / max(scene.height, 1) - 1.0
    inverse = numpy.linalg.inv(scene.projection() @ scene.camera.view())
    near = inverse @ numpy.array([nx, ny, -1.0, 1.0])
    far = inverse @ numpy.array([nx, ny, 1.0, 1.0])
    near = near[0:3] / near[3]
    far = far[0:3] / far[3]
    direction = far - near
    length = numpy.linalg.norm(direction)
    if length <= 0.0:
        return near, numpy.array([0.0, 0.0, 1.0])
    return near, direction / length


def ray_box(origin, direction, low, high):
    """
    The distance of the first crossing of the ray and the box, None when the ray misses it.
    """
    t0, t1 = -numpy.inf, numpy.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-9:
            if origin[axis] < low[axis] or origin[axis] > high[axis]:
                return None
            continue
        a = (low[axis] - origin[axis]) / direction[axis]
        b = (high[axis] - origin[axis]) / direction[axis]
        if a > b:
            a, b = b, a
        t0 = max(t0, a)
        t1 = min(t1, b)
        if t0 > t1:
            return None
    if t1 < 0.0:
        return None
    return max(t0, 0.0)


def part_box(unit, part_name):
    """
    (low, high) of the box of the part with the proportions of the unit, None when the model has no part.
    """
    figure = unit.figure
    if part_name not in figure.part_names:
        return None
    part = figure.figure(part_name)
    if part is None or part.center is None:
        return None
    try:
        low, high = part.variant_bounds(figure.coefs)
    except (AttributeError, IndexError, ValueError):
        return None
    return numpy.asarray(low, dtype=numpy.float64), numpy.asarray(high, dtype=numpy.float64)


def part_matrix(unit, part_name, pose):
    """
    The matrix of the part in the world: the same as client.scene.UnitView.update.
    """
    position, rotation = pose[part_name]
    return unit.world_matrix() @ gl.pose_matrix(position, quat.to_matrix(rotation))


def unit_part_at(unit, origin, direction):
    """
    (distance, hit location) of the nearest box of the unit which the ray crosses, None when it misses.
    """
    if unit.figure is None or unit.fighter is None or unit.is_dead:
        return None
    if getattr(unit, 'hidden', False):
        return None
    pose = unit.pose()
    best = None
    for location, names in PART_NAMES.items():
        for name in names:
            if name not in pose:
                continue
            box = part_box(unit, name)
            if box is None:
                continue
            matrix = part_matrix(unit, name, pose)
            inverse = numpy.linalg.inv(matrix)
            local_origin = (inverse @ numpy.append(origin, 1.0))[0:3]
            local_direction = inverse[0:3, 0:3] @ direction
            length = numpy.linalg.norm(local_direction)
            if length <= 0.0:
                continue
            distance = ray_box(local_origin, local_direction / length, box[0], box[1])
            if distance is None:
                continue
            # the distance along the world ray: the local length of the direction scales it
            distance /= length
            if best is None or distance < best[0]:
                best = (distance, location)
    return best


def pick_part(app, pos, units=None):
    """
    (unit, hit location) of the body under the mouse, None when the ray meets no body. Like 565e40/5718b0 of
    game.exe only the figures whose rectangles of the last draw hold the point are tested; the answer is kept
    for the frame (game.exe reads the mouse once a frame, CCursor__Update 5e52e0).
    """
    if app.scene is None or app.terrain is None:
        return None
    key = (tuple(pos), getattr(app, 'frame_index', None), units is None)
    cached = getattr(app, 'pick_cache', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    result = _pick_part(app, pos, units)
    app.pick_cache = (key, result)
    return result


def _pick_part(app, pos, units):
    origin, direction = screen_ray(app, pos[0], pos[1])
    if units is None:
        units_at = getattr(app.scene, 'units_at', None)
        if units_at is not None:
            map_units = set(id(unit) for unit in app.map_units)
            units = [unit for unit in units_at(pos[0], pos[1]) if id(unit) in map_units]
        else:
            units = app.map_units
    best = None
    for unit in units:
        hit = unit_part_at(unit, origin, direction)
        if hit is None:
            continue
        if best is None or hit[0] < best[0]:
            best = (hit[0], unit, hit[1])
    return (best[1], best[2]) if best is not None else None


def cursor_for(app, unit, location):
    """
    The cursor of the hit location under the mouse: the part when a called shot is chosen, the whole unit
    otherwise (the remake's reading, the rule of game.exe is not found).
    """
    if getattr(app, 'aim_part', None) is None:
        return WHOLE_UNIT_CURSOR
    return PART_CURSORS.get(location, WHOLE_UNIT_CURSOR)


def hover_at(app, pos):
    """
    (unit, hit location or None) under the mouse for the information window and the highlight: the body
    under the ray first, then the unit around the point of the ground (a big body or a corpse).
    """
    hit = pick_part(app, pos)
    if hit is not None:
        return hit
    scene = getattr(app, 'scene', None)
    if scene is None or app.terrain is None:
        return None, None
    point = scene.pick_ground(pos[0], pos[1])
    if point is None:
        return None, None
    unit = app.unit_at(point)
    if unit is not None:
        return unit, None
    return app.corpse_at(point), None
