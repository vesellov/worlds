"""
OpenGL helpers for Kivy: the shader, the matrices of the camera and the textures.

The world keeps the axes of the game: x to the east, y to the north, z up. The matrices are built with numpy
for column vectors and given to Kivy whole, see kivy_matrix(), so nothing depends on the order
in which Kivy multiplies its own matrices.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

import numpy as np

from kivy.graphics.texture import Texture
from kivy.graphics.transformation import Matrix

#------------------------------------------------------------------------------

# tangent of the half of the horizontal angle of view, game.exe 0x740158; the screen is 4:3,
# so the vertical one is 0.75 of it
TAN_HALF_WIDTH = 0.4815746
TAN_HALF_HEIGHT = TAN_HALF_WIDTH * 0.75
FIELD_OF_VIEW = math.degrees(2.0 * math.atan(TAN_HALF_HEIGHT))

# the space of the 3D interface: x to the right, y down, z away from the eye (game.exe 5fc450)
INTERFACE_VIEW = np.diag([1.0, -1.0, -1.0, 1.0])

VERTEX_FORMAT = [
    (b'v_pos', 3, 'float'),
    (b'v_normal', 3, 'float'),
    (b'v_tc0', 2, 'float'),
]

# the point lights of the spells (engine.magic.SpellLight) which the shaders take
POINT_LIGHTS = 8

# while a modal window is up game.exe draws the world and the HUD gray and dark (the menu Esc,
# the window of the difficulty and the window Options): pause_gray blends the color to its
# luminance, pause_dark multiplies the result. The interface props (the columns of the menus,
# the cursor, the modal windows themselves) keep their colors.
PAUSE_SHADER = """
uniform float pause_gray;
uniform float pause_dark;

vec3 paused(vec3 color) {
    float luma = dot(color, vec3(0.299, 0.587, 0.114));
    return mix(color, vec3(luma), pause_gray) * pause_dark;
}
"""

# the light of the map (engine.lighting): the sun and the ambient colors, the fog of the sky color
# goes linearly from fog_start to fog_end by the distance from the eye; the point lights of the spells
# (68d560 for the vertices of the land, 6c3180): point_light = (the point in the eye space, 1 / r^2),
# point_color = (the color, r^2), each one gives the light of the vertex plus its color times 1 - d^2 / r^2,
# full on the side facing it and dot + 1 on the other, up to 1; the largest of them is taken
LIGHT_SHADER = """
uniform vec3 light_dir;
uniform vec3 sun_color;
uniform vec3 ambient_color;
uniform vec3 fog_color;
uniform float fog_start;
uniform float fog_end;

vec3 lit(vec3 color, vec3 normal) {
    return color * (ambient_color + sun_color * max(dot(normalize(normal), light_dir), 0.0));
}

vec3 point_lit(vec3 base, vec3 best, vec3 n, vec3 pos, vec4 light, vec4 color) {
    vec3 d = light.xyz - pos;
    float d2 = dot(d, d);
    if (d2 >= color.w) {
        return best;
    }
    float facing = dot(n, d) / max(sqrt(d2), 0.0001);
    float w = facing > 0.0 ? 1.0 : facing + 1.0;
    return max(best, min(base + color.rgb * ((1.0 - d2 * light.w) * w), vec3(1.0)));
}
""" + ''.join('uniform vec4 point_light%d;\nuniform vec4 point_color%d;\n' % (i, i) for i in range(POINT_LIGHTS)) + """
vec3 lit_at(vec3 color, vec3 normal, vec3 pos) {
    vec3 n = normalize(normal);
    vec3 base = ambient_color + sun_color * max(dot(n, light_dir), 0.0);
    vec3 best = base;
""" + ''.join('    best = point_lit(base, best, n, pos, point_light%d, point_color%d);\n' % (i, i)
              for i in range(POINT_LIGHTS)) + """    return color * best;
}

vec3 fogged(vec3 color, float distance) {
    float fog = clamp((distance - fog_start) / max(fog_end - fog_start, 0.001), 0.0, 1.0);
    return mix(color, fog_color, fog);
}
""" + PAUSE_SHADER

VERTEX_SHADER = """
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3 v_pos;
attribute vec3 v_normal;
attribute vec2 v_tc0;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;
uniform mat4 normal_mat;

varying vec2 tex_coord;
varying vec3 normal;
varying vec3 eye_point;
varying float eye_distance;

void main(void) {
    tex_coord = v_tc0;
    // the normal in the eye space: Kivy leaves normal_mat empty when modelview_mat is set directly
    normal = (modelview_mat * vec4(v_normal, 0.0)).xyz;
    vec4 eye_pos = modelview_mat * vec4(v_pos, 1.0);
    eye_point = eye_pos.xyz;
    eye_distance = length(eye_pos.xyz);
    gl_Position = projection_mat * eye_pos;
}
"""

FRAGMENT_SHADER = """
#ifdef GL_ES
    precision highp float;
#endif

varying vec2 tex_coord;
varying vec3 normal;
varying vec3 eye_point;
varying float eye_distance;

uniform sampler2D texture0;
""" + LIGHT_SHADER + """
void main(void) {
    vec4 color = texture2D(texture0, tex_coord);
    if (color.a < 0.5) {
        discard;
    }
    gl_FragColor = vec4(paused(fogged(lit_at(color.rgb, normal, eye_point), eye_distance)), color.a);
}
"""

# the water of engine.water: the waves of game.exe 6c3e40 move the vertices on the GPU,
# v_wave is (phase of the vertex, wave multiplier of the material, column, row of the sector grid)
WATER_VERTEX_FORMAT = [
    (b'v_pos', 3, 'float'),
    (b'v_normal', 3, 'float'),
    (b'v_tc0', 2, 'float'),
    (b'v_tint', 3, 'float'),
    (b'v_alpha', 1, 'float'),
    (b'v_wave', 4, 'float'),
    (b'v_high', 3, 'float'),
    (b'v_glow', 3, 'float'),
]

WATER_VERTEX_SHADER = """
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3 v_pos;
attribute vec3 v_normal;
attribute vec2 v_tc0;
attribute vec3 v_tint;
attribute float v_alpha;
attribute vec4 v_wave;
attribute vec3 v_high;
attribute vec3 v_glow;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;
uniform mat4 normal_mat;
uniform float wave_time;
uniform float wave_amplitude;
uniform float sway_phase;
uniform float highlight;

varying vec2 tex_coord;
varying vec3 normal;
varying vec3 tint;
varying vec3 shine;
varying vec3 glow;
varying float alpha;
varying vec3 eye_point;
varying float eye_distance;

void main(void) {
    float a = wave_amplitude * v_wave.y;
    float w = sin(v_wave.x + wave_time * v_wave.y);
    vec3 p = v_pos;
    p.x += sin(v_wave.z * 0.78539816 + sway_phase) * a * 3.0;
    p.y += sin(v_wave.w * 0.78539816 + sway_phase) * a * 3.0;
    p.z += w * a * 0.25;
    tex_coord = v_tc0;
    // the normal in the eye space: Kivy leaves normal_mat empty when modelview_mat is set directly
    normal = (modelview_mat * vec4(v_normal, 0.0)).xyz;
    tint = v_tint;
    shine = max(w, 0.0) * highlight * v_wave.y * v_high;
    glow = v_glow;
    alpha = v_alpha;
    vec4 eye_pos = modelview_mat * vec4(p, 1.0);
    eye_point = eye_pos.xyz;
    eye_distance = length(eye_pos.xyz);
    gl_Position = projection_mat * eye_pos;
}
"""

WATER_FRAGMENT_SHADER = """
#ifdef GL_ES
    precision highp float;
#endif

varying vec2 tex_coord;
varying vec3 normal;
varying vec3 tint;
varying vec3 shine;
varying vec3 glow;
varying float alpha;
varying vec3 eye_point;
varying float eye_distance;

uniform sampler2D texture0;
""" + LIGHT_SHADER + """
void main(void) {
    vec4 color = texture2D(texture0, tex_coord);
    vec3 rgb = lit_at(color.rgb * tint, normal, eye_point) + shine + glow;
    gl_FragColor = vec4(paused(fogged(rgb, eye_distance)), alpha);
}
"""

# the flat elements of the battle screen (engine.hud): no light, the texture times the color of the element
HUD_VERTEX_FORMAT = [
    (b'v_pos', 3, 'float'),
    (b'v_tc0', 2, 'float'),
]

HUD_VERTEX_SHADER = """
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3 v_pos;
attribute vec2 v_tc0;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;

varying vec2 tex_coord;

void main(void) {
    tex_coord = v_tc0;
    gl_Position = projection_mat * modelview_mat * vec4(v_pos, 1.0);
}
"""

HUD_FRAGMENT_SHADER = """
#ifdef GL_ES
    precision highp float;
#endif

varying vec2 tex_coord;

uniform sampler2D texture0;
uniform vec4 color;
""" + PAUSE_SHADER + """
void main(void) {
    vec4 rgb = texture2D(texture0, tex_coord) * color;
    gl_FragColor = vec4(paused(rgb.rgb), rgb.a);
}
"""

# the particles (engine.particles): squares facing the camera, their color multiplies the texture,
# the fog of the hour fades the far ones like the rest of the world
PARTICLE_VERTEX_FORMAT = [
    (b'v_pos', 3, 'float'),
    (b'v_tc0', 2, 'float'),
    (b'v_color', 4, 'float'),
]

PARTICLE_VERTEX_SHADER = """
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3 v_pos;
attribute vec2 v_tc0;
attribute vec4 v_color;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;

varying vec2 tex_coord;
varying vec4 tint;
varying float eye_distance;

void main(void) {
    vec4 eye = modelview_mat * vec4(v_pos, 1.0);
    tex_coord = v_tc0;
    tint = v_color;
    eye_distance = length(eye.xyz);
    gl_Position = projection_mat * eye;
}
"""

PARTICLE_FRAGMENT_SHADER = """
#ifdef GL_ES
    precision highp float;
#endif

varying vec2 tex_coord;
varying vec4 tint;
varying float eye_distance;

uniform sampler2D texture0;
uniform float fog_start;
uniform float fog_end;
""" + PAUSE_SHADER + """
void main(void) {
    vec4 color = texture2D(texture0, tex_coord) * tint;
    float fog = clamp((eye_distance - fog_start) / max(fog_end - fog_start, 0.001), 0.0, 1.0);
    color.a *= 1.0 - fog;
    gl_FragColor = vec4(paused(color.rgb), color.a);
}
"""

#------------------------------------------------------------------------------


def perspective(fov_degrees, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_degrees) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def interface_point(x, y, depth):
    """
    The point of the 3D interface seen at (x, y) of the 800 x 600 screen at the depth, game.exe 5fc450
    (the models) and CInterfaceElement::SetScreenPosition 5e6fa0 (the elements).
    """
    return np.array([
        (x * 0.0025 - 1.0) * TAN_HALF_WIDTH * depth,
        (y * 0.0025 - 0.75) * TAN_HALF_WIDTH * depth,
        depth,
    ])


def camera_view(position, rotation_matrix):
    """
    View matrix of a camera of game.exe: it looks along its +z axis and its +y axis goes down on the screen.
    """
    m = np.identity(4)
    axes = np.asarray(rotation_matrix, dtype=np.float64)
    m[0, 0:3] = axes[:, 0]
    m[1, 0:3] = -axes[:, 1]
    m[2, 0:3] = -axes[:, 2]
    m[0:3, 3] = -m[0:3, 0:3] @ np.asarray(position, dtype=np.float64)
    return m


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    true_up = np.cross(side, forward)
    m = np.identity(4)
    m[0, 0:3] = side
    m[1, 0:3] = true_up
    m[2, 0:3] = -forward
    m[0:3, 3] = -m[0:3, 0:3] @ eye
    return m


def pose_matrix(position, rotation_matrix):
    m = np.identity(4)
    m[0:3, 0:3] = rotation_matrix
    m[0:3, 3] = position
    return m


def kivy_matrix(m):
    """
    Kivy keeps the matrices column by column, as OpenGL does.
    """
    k = Matrix()
    k.set(flat=[float(v) for v in np.asarray(m).T.ravel()])
    return k


def make_texture(rgba, mipmap=True, wrap='repeat'):
    height, width = rgba.shape[0:2]
    texture = Texture.create(size=(width, height), colorfmt='rgba', bufferfmt='ubyte', mipmap=mipmap)
    texture.blit_buffer(np.ascontiguousarray(rgba).tobytes(), colorfmt='rgba', bufferfmt='ubyte')
    texture.wrap = wrap
    texture.mag_filter = 'linear'
    texture.min_filter = 'linear_mipmap_linear' if mipmap else 'linear'
    return texture


def white_texture():
    rgba = np.full((4, 4, 4), 255, dtype=np.uint8)
    return make_texture(rgba, mipmap=False)


def checker_texture(size=64):
    cells = (np.indices((size, size)).sum(axis=0) // 8) % 2
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, 0:3] = np.where(cells[:, :, None], 200, 120)
    rgba[:, :, 3] = 255
    return make_texture(rgba)
