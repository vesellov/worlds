"""
Quaternions (w, x, y, z) as game.exe uses them.

rotate() is the formula of the bone update (sub_50c230): 2(u.v)u + (s^2 - u.u)v + 2s(u x v),
slerp() repeats Quat::Slerp (sub_49ed20): the shortest arc, plain lerp for very close keys.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

import numpy as np

#------------------------------------------------------------------------------

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])

SLERP_LERP_EPSILON = 1e-4

#------------------------------------------------------------------------------


def multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotate(q, v):
    s = q[0]
    u = q[1:4]
    return 2.0 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2.0 * s * np.cross(u, v)


def slerp(a, b, t):
    cos = float(np.dot(a, b))
    if cos < 0.0:
        b = -b
        cos = -cos
    if 1.0 - cos < SLERP_LERP_EPSILON:
        k0 = 1.0 - t
        k1 = t
    else:
        theta = math.acos(min(cos, 1.0))
        sin = math.sin(theta)
        k0 = math.sin((1.0 - t) * theta) / sin
        k1 = math.sin(t * theta) / sin
    return a * k0 + b * k1


def multiply_many(a, b):
    """
    multiply() of the rows of two (n, 4) arrays.
    """
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), axis=1)


def rotate_many(q, v):
    """
    rotate() of the rows: (n, 4) quaternions and (n, 3) vectors.
    """
    s = q[:, 0:1]
    u = q[:, 1:4]
    uv = np.sum(u * v, axis=1, keepdims=True)
    uu = np.sum(u * u, axis=1, keepdims=True)
    cross = np.stack((u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1],
                      u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2],
                      u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]), axis=1)
    return 2.0 * uv * u + (s * s - uu) * v + 2.0 * s * cross


def slerp_many(a, b, t):
    """
    slerp() of the rows of two (n, 4) arrays by one fraction t.
    """
    cos = np.sum(a * b, axis=1)
    flip = cos < 0.0
    b = np.where(flip[:, None], -b, b)
    cos = np.where(flip, -cos, cos)
    close = 1.0 - cos < SLERP_LERP_EPSILON
    theta = np.arccos(np.minimum(cos, 1.0))
    sin = np.where(close, 1.0, np.sin(theta))
    k0 = np.where(close, 1.0 - t, np.sin((1.0 - t) * theta) / sin)
    k1 = np.where(close, t, np.sin(t * theta) / sin)
    return a * k0[:, None] + b * k1[:, None]


def to_matrices(q):
    """
    to_matrix() of the rows: (n, 3, 3).
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((len(q), 3, 3))
    m[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    m[:, 0, 1] = 2.0 * (x * y - w * z)
    m[:, 0, 2] = 2.0 * (x * z + w * y)
    m[:, 1, 0] = 2.0 * (x * y + w * z)
    m[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    m[:, 1, 2] = 2.0 * (y * z - w * x)
    m[:, 2, 0] = 2.0 * (x * z - w * y)
    m[:, 2, 1] = 2.0 * (y * z + w * x)
    m[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return m


def from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    h = angle * 0.5
    return np.concatenate([[math.cos(h)], axis * math.sin(h)])


def to_matrix(q):
    """
    3x3 rotation matrix which turns column vectors the same way as rotate().
    """
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])
