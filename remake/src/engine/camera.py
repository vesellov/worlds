"""
The camera of the player on a map: the target on the ground under the carrier (the party), the distance,
the pitch and the yaw, and the limits of the mouse and of the keyboard.

The numbers are the defaults of [Camera Settings] of HKCU\\Software\\Nival Interactive\\EvilIslands
(FUN_00458da0 @ 458da0); the registry is read in FUN_00459100 @ 459100. The camera itself is updated by
FUN_0040fb30 @ 40fb30, the eye is placed by FUN_0040f8b0 @ 40f8b0. The values of the registry of this
installation are the same as the defaults (CameraDefaultDistanceToCarrier 30, CameraDefaultMinPitch 20,
CameraDefaultMaxPitch 80, CameraMinLimitDistanceToCarrier 3, CameraMaxLimitDistanceToCarrier 80).

The eye is the target moved by the distance along the direction of the yaw and the pitch (the pitch is the
angle above the ground, the yaw 0 looks to the north and the camera stands to the south of the target).
A pitch under the minimum and a distance over the limit are clamped at once; a distance under the minimum
is a spring back to the comfortable minimum (40fb30), the remake clamps it too. The yaw has no limit.

The floor of the eye (40e4c0, called by 40f8b0 after the eye is placed): the eye is not lower than CameraMinHeight
over the highest ground within 2 around it (40d9b0, a step of 1) and over the tops of the objects near it
(56daf0): the top of an object counts in full while the eye is over its circle and fades to nothing 10 away
from it; the rise over the objects is not more than CameraMaxHeight. A lifted eye still looks at the target.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

import numpy as np

#------------------------------------------------------------------------------

# the camera which follows the party, the values of [Camera Settings]
DEFAULT_DISTANCE = 30.0
DEFAULT_PITCH = 45.0
MIN_PITCH = 20.0
MAX_PITCH = 80.0
MIN_LIMIT_DISTANCE = 3.0
MAX_LIMIT_DISTANCE = 80.0
MIN_DISTANCE = 4.0
MAX_DISTANCE = 100.0

# the height of the camera over the obstacles and the offset of the target over the ground (+0xec)
MIN_HEIGHT = 4.0
MAX_HEIGHT = 100.0
DEFAULT_HEIGHT = 30.0
# game.exe looks at the ground of the carrier (40d9b0 averages it over +-2); the remake looks at its chest
TARGET_HEIGHT = 1.0

# the speeds of the mouse and of the keyboard (CameraDefault<...>Speed and Power)
ROTATE_SPEED = 1.0
ROTATE_POWER_MOUSE = 1.0
ROTATE_POWER_KEYBOARD = 0.5
PITCH_SPEED = 1.0
PITCH_POWER_MOUSE = 1.0
PITCH_POWER_KEYBOARD = 0.5
ZOOM_SPEED = 1.0
ZOOM_POWER_MOUSE = 1.0
ZOOM_POWER_KEYBOARD = 0.5
SCROLL_SPEED = 1.0
SCROLL_POWER_MOUSE = 1.0
SCROLL_POWER_KEYBOARD = 0.5

# 40fb30: the velocity of the scroll comes to the one of the keys by 0.1333 a tick (+0xb0, set in 40c8b0)
# and the move is the velocity times (distance - min) / ((max - min) * 1.125) + 0.2 (the min and the max are
# CameraMinDistanceToCarrier and CameraMaxDistanceToCarrier)
SCROLL_SMOOTH = 0.13333334
SCROLL_ZOOM_SPAN = 1.125
SCROLL_ZOOM_BASE = 0.2
SCROLL_STOP = 1e-4

# the bands of the screen which the mouse scrolls by and selects by (CameraBorderScrollArea and others)
BORDER_SCROLL_AREA = 5
FRAME_SELECTION_AREA = 20
TERRAIN_SENSITIVE_AREA = 10.0

# the camera of a briefing
BRIEFING_MIN_DISTANCE = 4.0
BRIEFING_MAX_DISTANCE = 100.0
BRIEFING_MIN_LIMIT_DISTANCE = 3.0
BRIEFING_MAX_LIMIT_DISTANCE = 80.0
BRIEFING_MIN_PITCH = 30.0
BRIEFING_MAX_PITCH = 60.0

# 4113a0: the shake of the camera from a blow or a lightning (strength 5 for a blow, radius 0.25)
SHAKE_STRENGTH = 5.0
SHAKE_DECAY = 0.25
SHAKE_DISTANCE = 30.0
SHAKE_AMPLITUDE = 0.05
SHAKE_FREQUENCY = 1.7

# the degrees of the yaw and of the pitch by a pixel of the mouse (the remake's numbers)
MOUSE_YAW = 0.4
MOUSE_PITCH = 0.3

# 40e4c0: the ground is taken within 2 around the eye by a step of 1; the top of an object fades over 10
FLOOR_GROUND_REACH = 2
FLOOR_OBJECT_FADE = 10.0

# +0xec of the camera (40fb30): the listener of the sounds is this high over the ground under the target
LISTENER_HEIGHT = 2.0

NEAR_CLIP_DISTANCE = 2.0
FAR_CLIP_DISTANCE = 100.0

#------------------------------------------------------------------------------


def clamp_pitch(pitch):
    """
    40fb30: the pitch of the camera stays between CameraDefaultMinPitch and CameraDefaultMaxPitch.
    """
    return min(max(float(pitch), MIN_PITCH), MAX_PITCH)


def clamp_distance(distance):
    """
    40fb30: the zoom stays between the limit distances (3 and 80); the comfortable 4..100 is a spring.
    """
    return min(max(float(distance), MIN_LIMIT_DISTANCE), MAX_LIMIT_DISTANCE)


def obstacles(boxes):
    """
    The objects for floor_height: an array of (x, y, radius, top) from the lists of the corners of the boxes of
    every object (engine.pathing.object_parts); the circle is the one around the box of the whole object.
    """
    rows = []
    for parts in boxes:
        corners = [corner for _, part_corners in parts for corner in part_corners]
        if not corners:
            continue
        points = np.asarray(corners, dtype=np.float64)[:, :3]
        low, high = points.min(axis=0), points.max(axis=0)
        rows.append(((low[0] + high[0]) * 0.5, (low[1] + high[1]) * 0.5,
                     math.hypot(high[0] - low[0], high[1] - low[1]) * 0.5, high[2]))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def floor_height(terrain, x, y, objects=None, min_height=MIN_HEIGHT, max_height=MAX_HEIGHT):
    """
    40e4c0: the lowest height of the eye over the point; objects is the array of obstacles().
    """
    reach = FLOOR_GROUND_REACH
    ground = max(terrain.height_at(x + dx, y + dy)
                 for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1))
    top = 0.0
    if objects is not None and len(objects):
        away = np.hypot(objects[:, 0] - x, objects[:, 1] - y) - objects[:, 2]
        fade = np.clip(1.0 - away / FLOOR_OBJECT_FADE, 0.0, 1.0)
        top = max(0.0, float((objects[:, 3] * fade).max()))
    if top <= ground:
        return ground + min_height
    return min(top - ground, max_height) + ground + min_height


def eye_offset(yaw, pitch, distance):
    """
    The offset of the eye from the target: the yaw in degrees, 0 looks to the north (+y), the camera stands
    to the south; the pitch is the angle above the ground.
    """
    yaw, pitch = math.radians(yaw), math.radians(pitch)
    return np.array([
        -math.sin(yaw) * math.cos(pitch),
        -math.cos(yaw) * math.cos(pitch),
        math.sin(pitch),
    ]) * distance

#------------------------------------------------------------------------------


class Camera(object):
    """
    The camera of the player: the point on the ground which it looks at, the yaw and the pitch in degrees
    and the distance from the target.
    """

    def __init__(self, target=(0.0, 0.0, 1.0), yaw=0.0, pitch=DEFAULT_PITCH, distance=DEFAULT_DISTANCE):
        self.target = np.array(target, dtype=np.float64)
        self.yaw = float(yaw)
        self.pitch = clamp_pitch(pitch)
        self.distance = clamp_distance(distance)
        # 4113a0: the shake (the strength, the ticks left, the whole time and the phase)
        self.shake_strength = 0.0
        self.shake_left = 0.0
        self.shake_time = 1.0
        self.shake_phase = 0.0
        # 40fb30: the velocity of the scroll on the ground (+0x98) and the one of the keys (+0xa4, 40f230)
        self.scroll_velocity = [0.0, 0.0]
        self.scroll_target = [0.0, 0.0]
        # CameraFlipXAxis/CameraFlipYAxis: the mouse turns the camera the other way
        self.reverse_x = False
        self.reverse_y = False
        # 40e4c0: the land (engine.terrain.Terrain) and the obstacles() which the eye stays above, None: no floor
        self.terrain = None
        self.obstacles = None
        self.floor_point = None
        self.floor = 0.0

    def __repr__(self):
        return 'Camera(yaw %g, pitch %g, %.1f away)' % (self.yaw, self.pitch, self.distance)

    def shake(self, strength=SHAKE_STRENGTH, time=SHAKE_DECAY, point=None):
        """
        4113a0: a blow or a lightning shakes the camera; the point checks the distance to the target.
        """
        if point is not None:
            distance = math.sqrt(sum((float(point[i]) - float(self.target[i])) ** 2 for i in range(3)))
            if distance > SHAKE_DISTANCE:
                return False
            strength *= max(0.0, 1.0 - distance / SHAKE_DISTANCE)
        if strength <= 0.0:
            return False
        self.shake_strength = max(self.shake_strength, float(strength))
        self.shake_left = max(self.shake_left, float(time))
        self.shake_time = max(self.shake_left, 1e-6)
        return True

    def advance_shake(self, ticks):
        """
        One step of the shake; ticks may be fractional (the frame is shorter than the tick of the world).
        """
        if self.shake_left <= 0.0:
            self.shake_strength = 0.0
            return
        self.shake_left = max(0.0, self.shake_left - float(ticks))
        self.shake_phase += SHAKE_FREQUENCY * float(ticks)
        if self.shake_left <= 0.0:
            self.shake_strength = 0.0

    def shake_offset(self):
        """
        The offset of the eye by the shake, (right, up, forward) of the camera.
        """
        if self.shake_left <= 0.0 or self.shake_strength <= 0.0:
            return (0.0, 0.0, 0.0)
        decay = self.shake_left / self.shake_time
        amplitude = SHAKE_AMPLITUDE * self.shake_strength * decay
        phase = self.shake_phase
        return (amplitude * math.sin(phase * 2.0), amplitude * math.sin(phase * 3.1 + 1.3),
                amplitude * 0.4 * math.sin(phase * 1.7 + 0.7))

    def eye(self):
        eye = self.target + eye_offset(self.yaw, self.pitch, self.distance)
        if self.terrain is not None:
            point = (float(eye[0]), float(eye[1]))
            if point != self.floor_point:
                self.floor_point = point
                self.floor = floor_height(self.terrain, point[0], point[1], self.obstacles)
            eye[2] = max(eye[2], self.floor)
        right, up, forward = self.shake_offset()
        if right or up or forward:
            yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
            side = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
            vertical = np.array([math.sin(yaw) * math.sin(pitch), math.cos(yaw) * math.sin(pitch),
                                 math.cos(pitch)])
            ahead = eye_offset(self.yaw, self.pitch, 1.0)
            eye = eye + side * right + vertical * up + ahead * forward
        return eye

    def pan(self, right, forward):
        """
        The target moves on the ground by the right and the forward of the camera.
        """
        yaw = math.radians(self.yaw)
        self.target[0] += math.cos(yaw) * right + math.sin(yaw) * forward
        self.target[1] += -math.sin(yaw) * right + math.cos(yaw) * forward

    def scroll_keys(self, right, forward, power=SCROLL_POWER_KEYBOARD):
        """
        40f230: the arrows set the velocity of the scroll: the right and the forward of the camera on the
        ground (-1, 0 or 1) times ScrollPowerKeyboard and ScrollSpeed; (0, 0) when no arrow is held.
        """
        yaw = math.radians(self.yaw)
        speed = float(power) * SCROLL_SPEED
        right, forward = float(right) * speed, float(forward) * speed
        self.scroll_target[0] = math.cos(yaw) * right + math.sin(yaw) * forward
        self.scroll_target[1] = -math.sin(yaw) * right + math.cos(yaw) * forward

    def advance_scroll(self, ticks):
        """
        40fb30: the velocity of the scroll comes to the one of the keys and moves the target; ticks may be
        fractional. False when the camera stands.
        """
        vx, vy = self.scroll_velocity
        tx, ty = self.scroll_target
        k = min(SCROLL_SMOOTH * float(ticks), 1.0)
        vx += (tx - vx) * k
        vy += (ty - vy) * k
        if abs(tx - vx) + abs(ty - vy) < SCROLL_STOP:
            vx, vy = tx, ty
        self.scroll_velocity = [vx, vy]
        if vx == 0.0 and vy == 0.0:
            return False
        zoom = (self.distance - MIN_DISTANCE) / ((MAX_DISTANCE - MIN_DISTANCE) * SCROLL_ZOOM_SPAN) + SCROLL_ZOOM_BASE
        self.target[0] += vx * float(ticks) * zoom
        self.target[1] += vy * float(ticks) * zoom
        return True

    def rotate_by(self, degrees):
        """
        The yaw is not limited, the camera turns around the target.
        """
        self.yaw += float(degrees)

    def drag(self, dx, dy):
        """
        The mouse: RotateSpeed and PitchSpeed times PowerMouse by the pixels of the move; the axes may be
        flipped (CameraFlipXAxis and CameraFlipYAxis, the reverse_x and reverse_y of the options).
        """
        turn = dx * ROTATE_SPEED * ROTATE_POWER_MOUSE * MOUSE_YAW
        tilt = dy * PITCH_SPEED * PITCH_POWER_MOUSE * MOUSE_PITCH
        if self.reverse_x:
            turn = -turn
        if self.reverse_y:
            tilt = -tilt
        self.rotate_by(-turn)
        self.pitch_by(-tilt)

    def pitch_by(self, degrees):
        self.pitch = clamp_pitch(self.pitch + float(degrees))

    def zoom_by(self, distance):
        self.distance = clamp_distance(self.distance + float(distance))

    def zoom_to(self, distance):
        self.distance = clamp_distance(distance)

    def reset(self):
        self.pitch = DEFAULT_PITCH
        self.distance = DEFAULT_DISTANCE
        self.yaw = 0.0
        self.shake_strength = 0.0
        self.shake_left = 0.0
