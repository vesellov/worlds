"""
The 3D view: the terrain of a map and the units, drawn by Kivy with the shader of client.gl.

The water goes in its own render context after everything else: blended, tested against the depth
of the land but not writing it. The sky color of the hour clears the window and colors the fog.
Over the world, after a clear of the depth, comes the 3D interface: the models (menu columns, heads of the
portraits) and the flat elements of engine.hud, all ordered by their depth like in game.exe.

Every part of a figure is a rigid mesh: its vertices are built once for the body proportions of the unit
and every frame only the matrix of the part changes (the pose of engine.animation.Skeleton).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import time

import numpy as np

from kivy.graphics import RenderContext, Callback, Mesh, PushMatrix, PopMatrix, BindTexture, Color, InstructionGroup, Canvas
from kivy.graphics.context_instructions import MatrixInstruction
from kivy.graphics.transformation import Matrix
from kivy.graphics.opengl import (glEnable, glDisable, glDepthMask, glClear, glBlendFunc, GL_DEPTH_TEST,
                                  GL_DEPTH_BUFFER_BIT, GL_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
from kivy.core.window import Window
from kivy.uix.widget import Widget

#------------------------------------------------------------------------------

from engine import camera as camera_module
from engine import equipment
from engine import figure as figure_module
from engine import lighting
from engine import lightning
from engine import particles
from engine import quat
from engine import water
from engine import woundskin
from client import gl

#------------------------------------------------------------------------------

FIELD_OF_VIEW = gl.FIELD_OF_VIEW

# the light of the models of the 3D interface: from the upper left of the screen
INTERFACE_LIGHT = {
    'light_dir': [-0.4, 0.5, 0.77],
    'sun_color': [0.55, 0.55, 0.55],
    'ambient_color': [0.6, 0.6, 0.6],
    'fog_color': [0.0, 0.0, 0.0],
    'fog_start': 1000.0,
    'fog_end': 2000.0,
}

MAX_MESH_VERTICES = 65535
# a figure whose point is this far (units of the world) out of the screen may still show a part on it
UNIT_SCREEN_MARGIN = 6.0
# the box of rest of a figure grows by this for its rectangle on the screen (the arms and the weapons move)
SCREEN_RECT_SCALE = 1.4

# 5e18e0: the parts of the figure which hold the flying numbers, the first found
HEAD_PARTS = ('hd', 'bd', 'hp')

# the ring of the selection under a unit: its segments and the width of the band by its radius
MARK_SEGMENTS = 40
MARK_INNER = 0.82
MARK_HEIGHT = 0.06
# the time of a tick of the shake of the camera (engine.camera)
SHAKE_TICK_SECONDS = 0.055

# the scale of a hidden object (HideObject of the scripts): its parts shrink into its point
HIDDEN_SCALE = np.diag([0.0, 0.0, 0.0, 1.0])

#------------------------------------------------------------------------------


class Camera(camera_module.Camera):
    """
    The camera of engine.camera with the view matrix of the client.
    """

    def view(self):
        return gl.look_at(self.eye(), self.target)


class FixedCamera(object):
    """
    A camera of camera/*.cam: a position and a rotation, it does not follow anything.
    """

    def __init__(self, position, rotation):
        self.position = np.array(position, dtype=np.float64)
        self.rotation = tuple(rotation)
        self.target = self.position.copy()
        self.distance = 0.0
        self.yaw = 0.0
        self.pitch = 0.0

    def eye(self):
        return self.position

    def view(self):
        return gl.camera_view(self.position, quat.to_matrix(self.rotation))

    def pan(self, right, forward):
        pass


class UnitView(object):

    def __init__(self, unit, canvas, textures):
        """
        textures: texture group of a part (engine.equipment.TEXTURE_BODY / TEXTURE_GEAR) -> Kivy texture.
        """
        self.unit = unit
        self.canvas = canvas
        self.part_matrices = {}
        # the rectangle of the figure on the screen at the last draw (x0, y0, x1, y1 in the pixels of the
        # scene) and its depth, None off the screen: the mouse tests only these (5718b0)
        self.screen_rect = None
        self.screen_depth = 0.0
        # the indices of the parts in the arrays of the pose, for this skeleton
        self.indices_of = None
        self.indices = None
        # the body texture without wounds: its Kivy texture, its key and a function which gives its RGBA
        self.body_texture = textures[equipment.TEXTURE_BODY]
        self.body_key = None
        self.body_rgba = None
        self.body_wrap = 'repeat'
        # engine.woundskin.levels of the drawn texture and the (BindTexture, Mesh) of the parts which use it
        self.wound_levels = (0, ) * woundskin.PARTS_COUNT
        self.body_instructions = []
        # 5e18e0: the part whose middle on the screen holds the flying numbers, the middle of its mesh,
        # its matrix in the world at the last update
        self.head_name = None
        self.head_center = None
        self.head_matrix = None
        self.head_index = None
        figure = unit.figure
        with canvas:
            for part_name in figure.skeleton.visible:
                texture = textures.get(figure.texture_group(part_name)) or textures[equipment.TEXTURE_BODY]
                vertices, indexes = figure.figure(part_name).mesh_arrays(figure.coefs)
                if part_name in HEAD_PARTS and (self.head_name is None or
                                                HEAD_PARTS.index(part_name) < HEAD_PARTS.index(self.head_name)):
                    self.head_name = part_name
                    points = vertices[:, 0:3]
                    self.head_center = (points.min(axis=0) + points.max(axis=0)) * 0.5 if len(points) else np.zeros(3)
                bind = BindTexture(texture=texture, index=0)
                PushMatrix()
                # the context keeps view * matrix of the instruction, see Matrix.multiply() of Kivy
                self.part_matrices[part_name] = MatrixInstruction()
                mesh = Mesh(vertices=vertices.ravel().tolist(), indices=indexes.tolist(), fmt=gl.VERTEX_FORMAT,
                            mode='triangles', texture=texture)
                PopMatrix()
                if texture is self.body_texture:
                    self.body_instructions.append((bind, mesh))

    def set_body_texture(self, texture):
        for bind, mesh in self.body_instructions:
            bind.texture = texture
            mesh.texture = texture

    def head_point(self):
        """
        The middle of the head (5e18e0 takes the rectangle of "hd", else "bd", else "hp") in the world,
        None before the first update.
        """
        if self.head_matrix is None:
            return None
        return (self.head_matrix @ np.append(self.head_center, 1.0))[0:3]

    def hide(self):
        """
        The figure goes away: the matrices of its parts shrink them into a point (a shared canvas keeps the
        instructions, so the view is not removed from the canvas itself).
        """
        hidden_matrix = gl.kivy_matrix(HIDDEN_SCALE)
        for instruction in self.part_matrices.values():
            instruction.matrix = hidden_matrix

    def update(self):
        """
        The matrices of all the parts at once: the pose of the skeleton (Skeleton.pose_arrays) turned into
        4 x 4 matrices and put under the matrix of the unit.
        """
        unit = self.unit
        figure = unit.figure
        skeleton = figure.skeleton
        positions, rotations = skeleton.pose_arrays(unit.player.tracks, unit.player.time)
        base = unit.world_matrix()
        if getattr(unit, 'hidden', False) or getattr(unit, 'unseen', False):
            # a hidden object shrinks into its point; 56a990 does not draw a unit the player does not see
            base = base @ HIDDEN_SCALE
        if self.indices_of is not skeleton:
            self.indices_of = skeleton
            self.indices = np.array([skeleton.index[name] for name in self.part_matrices], dtype=np.int64)
        indices = self.indices
        local = np.zeros((len(indices), 4, 4))
        local[:, 0:3, 0:3] = quat.to_matrices(rotations[indices])
        local[:, 0:3, 3] = positions[indices]
        local[:, 3, 3] = 1.0
        # Kivy keeps the matrices column by column: the transposed rows are its flat lists
        world = base @ local
        if self.head_name is not None:
            if self.head_index is None:
                self.head_index = list(self.part_matrices).index(self.head_name)
            self.head_matrix = world[self.head_index]
        flats = np.transpose(world, (0, 2, 1)).reshape(len(indices), 16).tolist()
        shown_hidden = set()
        if equipment.arrow_in_hand_visible(unit):
            shown_hidden.add(equipment.ARROW_IN_HAND)
        hidden_parts = figure.hidden_parts
        weapon_hidden = equipment.weapon_hidden_parts(unit)
        if weapon_hidden:
            hidden_parts = set(hidden_parts) | weapon_hidden
            shown_hidden = set()
        hidden_matrix = None
        for (part_name, instruction), flat in zip(self.part_matrices.items(), flats):
            if part_name in hidden_parts and part_name not in shown_hidden:
                if hidden_matrix is None:
                    hidden_matrix = gl.kivy_matrix(HIDDEN_SCALE)
                instruction.matrix = hidden_matrix
                continue
            matrix = Matrix()
            matrix.set(flat=flat)
            instruction.matrix = matrix


class PlacedView(object):
    """
    The parts of a figure at their matrices; the scripts can move it (SetCP) or hide it (HideObject).
    """

    # a hidden figure gets this matrix: all its triangles fall into a point
    HIDDEN_MATRIX = np.zeros((4, 4))

    def add_part(self, world_matrix, local_matrix):
        matrix = gl.kivy_matrix(world_matrix @ local_matrix)
        self.local_matrices.append(local_matrix)
        self.part_matrices.append((MatrixInstruction(matrix=matrix), matrix))

    def place(self, world_matrix, hidden=False):
        """
        The figure goes to world_matrix (or nowhere when hidden).
        """
        hidden_matrix = gl.kivy_matrix(self.HIDDEN_MATRIX)
        self.part_matrices = [(instruction, hidden_matrix if hidden else gl.kivy_matrix(world_matrix @ local_matrix))
                              for (instruction, _), local_matrix in zip(self.part_matrices, self.local_matrices)]
        self.update()

    def update(self):
        for instruction, matrix in self.part_matrices:
            instruction.matrix = matrix


class StaticObjectView(PlacedView):
    """
    A figure which does not move: a tree, a building, a stone. Its parts get their matrices once.
    """

    def __init__(self, figure_model, world_matrix, canvas, textures):
        """
        textures: group of figure_model.texture_group() -> Kivy texture, the body texture is a must.
        """
        pose = figure_model.skeleton.pose({}, 0.0)
        # (instruction, matrix): the context keeps view * matrix of an instruction from the time the matrix was set,
        # so the matrices are set again when the camera moves
        self.part_matrices = []
        self.local_matrices = []
        with canvas:
            for part_name in figure_model.skeleton.visible:
                vertices, indexes = figure_model.figure(part_name).mesh_arrays(figure_model.coefs)
                position, rotation = pose[part_name]
                texture = textures.get(figure_model.texture_group(part_name)) or textures[equipment.TEXTURE_BODY]
                BindTexture(texture=texture, index=0)
                PushMatrix()
                self.add_part(world_matrix, gl.pose_matrix(position, quat.to_matrix(rotation)))
                Mesh(vertices=vertices.ravel().tolist(), indices=indexes.tolist(), fmt=gl.VERTEX_FORMAT,
                     mode='triangles', texture=texture)
                PopMatrix()



class LeverView(PlacedView):
    """
    A lever of the map (engine.lever.Lever): a figure which stands still and morphs its meshes when the phase
    of the lever changes (CLeverObject::vf06).
    """

    def __init__(self, figure_model, lever, world_matrix, canvas, textures):
        self.figure_model = figure_model
        self.lever = lever
        self.coefs = lever.view_coefs()
        pose = figure_model.skeleton.pose({}, 0.0)
        self.part_matrices = []
        self.local_matrices = []
        # (part name, Mesh)
        self.meshes = []
        with canvas:
            for part_name in figure_model.skeleton.visible:
                vertices, indexes = figure_model.figure(part_name).mesh_arrays(self.coefs)
                position, rotation = pose[part_name]
                texture = textures.get(figure_model.texture_group(part_name)) or textures[equipment.TEXTURE_BODY]
                BindTexture(texture=texture, index=0)
                PushMatrix()
                self.add_part(world_matrix, gl.pose_matrix(position, quat.to_matrix(rotation)))
                mesh = Mesh(vertices=vertices.ravel().tolist(), indices=indexes.tolist(), fmt=gl.VERTEX_FORMAT,
                            mode='triangles', texture=texture)
                self.meshes.append((part_name, mesh))
                PopMatrix()

    def update(self):
        PlacedView.update(self)
        coefs = self.lever.view_coefs()
        if coefs != self.coefs:
            self.coefs = coefs
            for part_name, mesh in self.meshes:
                vertices, _ = self.figure_model.figure(part_name).mesh_arrays(coefs)
                mesh.vertices = vertices.ravel().tolist()

#------------------------------------------------------------------------------


class SceneWidget(Widget):

    def __init__(self, game_data, **kwargs):
        super(SceneWidget, self).__init__(**kwargs)
        self.game_data = game_data
        self.camera = Camera()
        self.shake_last = None
        self.terrain = None
        self.unit_views = []
        self.static_views = []
        self.water_meshes = []
        self.water_offsets = {}
        self.static_view_key = None
        # LeverView, they are updated every frame
        self.lever_views = []
        self.textures = {}
        # (point, color, radius) of the lights of the spells, set_point_lights
        self.point_lights = []
        # both shaders are given together: setting them one by one links the new one with the default other
        self.canvas = RenderContext(use_parent_projection=False, use_parent_modelview=False, compute_normal_mat=True,
                                    vs=gl.VERTEX_SHADER, fs=gl.FRAGMENT_SHADER)
        if not self.canvas.shader.success:
            raise RuntimeError('shader failed to compile')
        self.water_canvas = RenderContext(use_parent_projection=False, use_parent_modelview=False, compute_normal_mat=True,
                                          vs=gl.WATER_VERTEX_SHADER, fs=gl.WATER_FRAGMENT_SHADER)
        if not self.water_canvas.shader.success:
            raise RuntimeError('water shader failed to compile')
        self.light = lighting.WorldLight(lighting.parse_lights(None)).at(lighting.HOURS / 2)
        self.wind = water.Wind()
        self.wave_clock = water.WaveClock(self.wind)
        # the models of the 3D interface (the column of the escape menu) are drawn over the world
        self.interface_canvas = RenderContext(use_parent_projection=False, use_parent_modelview=False, compute_normal_mat=True,
                                              vs=gl.VERTEX_SHADER, fs=gl.FRAGMENT_SHADER)
        self.interface_views = []
        # the flat elements of the battle screen (engine.hud) in the same space, ordered by the depth with the models
        self.hud_canvas = RenderContext(use_parent_projection=False, use_parent_modelview=False,
                                        vs=gl.HUD_VERTEX_SHADER, fs=gl.HUD_FRAGMENT_SHADER)
        if not self.hud_canvas.shader.success:
            raise RuntimeError('hud shader failed to compile')
        self.hud_layers = {}
        with self.canvas.before:
            Callback(self._enable_depth)
        with self.canvas.after:
            Callback(self._disable_depth_write)
        self.canvas.after.add(self.water_canvas)
        # the particles after the water: tested against the depth, not writing it
        self.particle_canvas = RenderContext(use_parent_projection=False, use_parent_modelview=False,
                                             vs=gl.PARTICLE_VERTEX_SHADER, fs=gl.PARTICLE_FRAGMENT_SHADER)
        if not self.particle_canvas.shader.success:
            raise RuntimeError('particle shader failed to compile')
        # the prints of the steps (engine.footprints) under the rings and the particles
        self.footprint_group = InstructionGroup()
        self.particle_canvas.add(self.footprint_group)
        # the rings of the selection and of the target under the units (G18), under the particles
        self.mark_group = InstructionGroup()
        self.particle_canvas.add(self.mark_group)
        self.particle_group = InstructionGroup()
        self.particle_canvas.add(self.particle_group)
        self.canvas.after.add(self.particle_canvas)
        with self.canvas.after:
            Callback(self._enable_depth_write)
            Callback(self._clear_depth)
        self.canvas.after.add(self.interface_canvas)
        self.canvas.after.add(self.hud_canvas)
        with self.canvas.after:
            Callback(self._disable_depth)
        # the pause filter of the modal windows (0 gray, 1 dark is the open scene); the interface
        # props keep the filter of the open scene, their columns stay colored as in game.exe
        self.pause_filter = None
        for canvas in (self.canvas, self.water_canvas, self.particle_canvas, self.hud_canvas,
                       self.interface_canvas):
            canvas['pause_gray'] = 0.0
            canvas['pause_dark'] = 1.0

    def set_pause_filter(self, gray, dark):
        """
        The world, the water, the particles and the HUD go gray and dark while a modal window is up.
        """
        if (gray, dark) == self.pause_filter:
            return
        self.pause_filter = (gray, dark)
        for canvas in (self.canvas, self.water_canvas, self.particle_canvas, self.hud_canvas):
            canvas['pause_gray'] = float(gray)
            canvas['pause_dark'] = float(dark)
        self._update_clear_color()

    def _enable_depth(self, *args):
        glEnable(GL_DEPTH_TEST)

    def _disable_depth(self, *args):
        glDisable(GL_DEPTH_TEST)

    def _clear_depth(self, *args):
        glClear(GL_DEPTH_BUFFER_BIT)

    def _disable_depth_write(self, *args):
        glDepthMask(False)

    def _enable_depth_write(self, *args):
        glDepthMask(True)

    def _additive_blend(self, *args):
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

    def _alpha_blend(self, *args):
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def set_particles(self, emitters, fraction=1.0, lightnings=(), precipitation=None):
        """
        The squares of the particles (4976b0): between the last tick and the new one by the fraction of the tick
        which went by, the frame of the last tick, facing the camera. The bolts of the lightnings
        (engine.lightning, 467ad0) are bands of their points facing the camera and balls of light.
        precipitation is (engine.weather.Precipitation, time in ticks): the triangles of the rain or the snow
        around the eye (4e4240, 4e4c80), drawn first.
        """
        self.particle_group.clear()
        view = self.camera.view()
        right = np.asarray(view[0, 0:3], dtype=np.float64)
        up = np.asarray(view[1, 0:3], dtype=np.float64)
        toward = np.asarray(view[2, 0:3], dtype=np.float64)
        if precipitation is not None:
            self.add_precipitation(precipitation[0], precipitation[1], right, up)
        # (texture name, additive, wrap) -> quads of 4 corners (x, y, z, u, v) and their color
        batches = {}
        for emitter in emitters:
            if not emitter.particles or not emitter.texture_name:
                continue
            quads = batches.setdefault((emitter.texture_name, emitter.additive, 'clamp_to_edge'), [])
            for p in emitter.particles:
                lx, ly, lz, ls = p.last
                size = ls + (p.size - ls) * fraction
                if size <= 0.0:
                    continue
                color = p.last_color
                alpha = ((color >> 24) & 0xff) / 255.0
                if alpha <= 0.0:
                    continue
                center = (lx + (p.x - lx) * fraction, ly + (p.y - ly) * fraction, lz + (p.z - lz) * fraction)
                rgba = (((color >> 16) & 0xff) / 255.0, ((color >> 8) & 0xff) / 255.0, (color & 0xff) / 255.0, alpha)
                quads.append((self.facing_square(center, size, particles.frame_uv(p.last_frame % 16), right, up),
                              rgba))
        for bolt in lightnings:
            quads = batches.setdefault((bolt.texture_name, True, 'repeat'), [])
            for points, width, along, band in bolt.bands(fraction):
                quads.extend(self.band_quads(points, width, along, band, toward))
            quads = batches.setdefault((bolt.ball_texture_name, True, 'clamp_to_edge'), [])
            frame = int(bolt.ticks)
            for center, size in bolt.balls():
                for point, square_size, square_frame, alpha in lightning.ball_squares(center, size, toward, frame):
                    quads.append((self.facing_square(point, square_size, particles.frame_uv(square_frame), right, up),
                                  (1.0, 1.0, 1.0, alpha / 255.0)))
        for (texture_name, additive, wrap), quads in sorted(batches.items()):
            if not quads:
                continue
            if additive:
                self.particle_group.add(Callback(self._additive_blend))
            texture = self.texture(texture_name, wrap=wrap)
            per_mesh = MAX_MESH_VERTICES // 4
            for start in range(0, len(quads), per_mesh):
                chunk = quads[start:start + per_mesh]
                vertices = []
                indices = []
                for n, (corners, rgba) in enumerate(chunk):
                    for corner in corners:
                        vertices.extend(corner + rgba)
                    k = n * 4
                    indices.extend((k, k + 1, k + 2, k, k + 2, k + 3))
                self.particle_group.add(Mesh(vertices=vertices, indices=indices, fmt=gl.PARTICLE_VERTEX_FORMAT,
                                             mode='triangles', texture=texture))
            if additive:
                self.particle_group.add(Callback(self._alpha_blend))

    def add_precipitation(self, precipitation, time, right, up):
        vertices = precipitation.triangles(self.camera.eye(), time, right=right, up=up)
        if len(vertices) == 0:
            return
        # the drops are a few pixels wide: the mipmaps would average the streak away
        key = precipitation.texture_name + '#flat'
        if key not in self.textures:
            rgba = self.game_data.texture(precipitation.texture_name)
            self.textures[key] = (gl.make_texture(rgba, mipmap=False, wrap='clamp_to_edge') if rgba is not None
                                  else gl.checker_texture())
        texture = self.textures[key]
        per_mesh = (MAX_MESH_VERTICES // 3) * 3
        for start in range(0, len(vertices), per_mesh):
            chunk = vertices[start:start + per_mesh]
            self.particle_group.add(Mesh(vertices=chunk.ravel().tolist(), indices=list(range(len(chunk))),
                                         fmt=gl.PARTICLE_VERTEX_FORMAT, mode='triangles', texture=texture))

    def set_footprints(self, marks):
        """
        The prints of the steps: (x, y, z, angle, u0, v0, u1, v1, size, alpha) of a print of the atlas
        "footprints"; they lie on the land and are drawn under the particles (engine.footprints).
        """
        self.footprint_group.clear()
        if not marks:
            return
        self.footprint_group.add(Callback(self._alpha_blend))
        vertices = []
        indices = []
        for x, y, z, angle, u0, v0, u1, v1, size, alpha in marks:
            if alpha <= 0.0 or size <= 0.0:
                continue
            base = len(vertices) // 9
            ca, sa = math.cos(angle), math.sin(angle)
            half = size * 0.5
            for sx, sy, u, v in ((-1, -1, u0, v0), (1, -1, u1, v0), (1, 1, u1, v1), (-1, 1, u0, v1)):
                px, py = sx * half, sy * half
                vertices.extend((x + ca * px - sa * py, y + sa * px + ca * py, z + MARK_HEIGHT, u, v))
                vertices.extend((1.0, 1.0, 1.0, alpha))
            indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
        if indices:
            self.footprint_group.add(Mesh(vertices=vertices, indices=indices, fmt=gl.PARTICLE_VERTEX_FORMAT,
                                          mode='triangles', texture=self.texture('footprints', wrap='clamp_to_edge')))

    def set_ground_marks(self, marks):
        """
        The flat rings under the units: (x, y, z, radius, (r, g, b, a)) of a unit. They are drawn in the
        layer of the particles, tested against the depth of the land (G18).
        """
        self.mark_group.clear()
        if not marks:
            return
        self.mark_group.add(Callback(self._alpha_blend))
        vertices = []
        indices = []
        white = gl.white_texture()
        for x, y, z, radius, color in marks:
            if radius <= 0.0:
                continue
            base = len(vertices) // 9
            inner = radius * MARK_INNER
            for i in range(MARK_SEGMENTS):
                angle = 2.0 * math.pi * i / MARK_SEGMENTS
                for r in (inner, radius):
                    vertices.extend((x + math.cos(angle) * r, y + math.sin(angle) * r, z + MARK_HEIGHT, 0.5, 0.5))
                    vertices.extend(color)
            for i in range(MARK_SEGMENTS):
                a = base + i * 2
                b = a + 1
                c = base + ((i + 1) % MARK_SEGMENTS) * 2
                d = c + 1
                indices.extend((a, b, d, a, d, c))
        if indices:
            self.mark_group.add(Mesh(vertices=vertices, indices=indices, fmt=gl.PARTICLE_VERTEX_FORMAT,
                                     mode='triangles', texture=white))

    @staticmethod
    def facing_square(center, size, uv, right, up):
        u0, v0, u1, v1 = uv
        c = np.asarray(center, dtype=np.float64)
        corners = []
        for sx, sy, u, v in ((-1, -1, u0, v1), (1, -1, u1, v1), (1, 1, u1, v0), (-1, 1, u0, v0)):
            corner = c + right * (sx * size) + up * (sy * size)
            corners.append((float(corner[0]), float(corner[1]), float(corner[2]), u, v))
        return corners

    @staticmethod
    def band_quads(points, width, along, band, toward):
        """
        467ad0: a band of the width along the points turned to the camera, u goes along the way to the length
        of the texture, v takes one of its four bands (the rows of the image from the bottom).
        """
        count = len(points)
        if count < 2 or width <= 0.0:
            return []
        tangents = np.empty_like(points)
        tangents[1:-1] = points[2:] - points[:-2]
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        sides = np.cross(tangents, toward)
        lengths = np.sqrt((sides * sides).sum(axis=1))
        lengths[lengths <= 0.0] = 1.0
        sides *= (width * 0.5 / lengths)[:, None]
        left = points - sides
        right = points + sides
        v0 = 1.0 - (band + 1) * lightning.BAND_HEIGHT
        v1 = 1.0 - band * lightning.BAND_HEIGHT
        us = [along * i / (count - 1) for i in range(count)]
        white = (1.0, 1.0, 1.0, 1.0)
        quads = []
        for i in range(count - 1):
            corners = (
                (float(left[i][0]), float(left[i][1]), float(left[i][2]), us[i], v1),
                (float(right[i][0]), float(right[i][1]), float(right[i][2]), us[i], v0),
                (float(right[i + 1][0]), float(right[i + 1][1]), float(right[i + 1][2]), us[i + 1], v0),
                (float(left[i + 1][0]), float(left[i + 1][1]), float(left[i + 1][2]), us[i + 1], v1),
            )
            quads.append((corners, white))
        return quads

    def texture(self, texture_name, wrap='repeat'):
        texture_name = (texture_name or '').lower()
        if texture_name not in self.textures:
            rgba = self.game_data.texture(texture_name) if texture_name else None
            self.textures[texture_name] = gl.make_texture(rgba, wrap=wrap) if rgba is not None else gl.checker_texture()
        return self.textures[texture_name]

    def set_terrain(self, terrain):
        self.terrain = terrain
        with self.canvas:
            for (sx, sy) in sorted(terrain.sectors):
                for texture_index, (vertices, indexes) in terrain.sector_meshes(sx, sy).items():
                    texture = self.texture(terrain.texture_name(texture_index), wrap='clamp_to_edge')
                    Mesh(vertices=vertices.ravel().tolist(), indices=indexes.tolist(), fmt=gl.VERTEX_FORMAT,
                         mode='triangles', texture=texture)

    def set_water(self, terrain, wind, seed=0):
        """
        The liquids of all sectors, the waves move with the wind of WORLD_SET.
        """
        self.wind = wind
        self.wave_clock = water.WaveClock(wind)
        phases = water.random_phases(seed)
        white = None
        # (material, Mesh, vertices as built): SetWaterLevel moves the meshes of a material
        self.water_meshes = []
        self.water_offsets = {}
        with self.water_canvas:
            for (sx, sy) in sorted(terrain.sectors):
                meshes = water.sector_meshes(terrain, sx, sy, wind, phases)
                for (texture_index, material_index), (vertices, indexes) in meshes.items():
                    if texture_index is None:
                        white = white or gl.white_texture()
                        texture = white
                    else:
                        texture = self.texture(terrain.texture_name(texture_index), wrap='clamp_to_edge')
                    mesh = Mesh(vertices=vertices.ravel().tolist(), indices=indexes.tolist(),
                                fmt=gl.WATER_VERTEX_FORMAT, mode='triangles', texture=texture)
                    self.water_meshes.append((material_index, mesh, vertices))

    def set_water_levels(self, offsets):
        """
        offsets: material of a liquid -> its height over the level of the .mp (engine.water.WaterLevels).
        game.exe moves the level of the material (CTerrain::SetWaterLevel 6cc7b0) and the vertices of the water
        of every sector (6c30a0).
        """
        for material_index, mesh, vertices in self.water_meshes:
            offset = offsets.get(material_index, 0.0)
            if offset == self.water_offsets.get(material_index, 0.0):
                continue
            moved = vertices.copy()
            moved[:, 2] += offset
            mesh.vertices = moved.ravel().tolist()
        self.water_offsets = dict(offsets)

    def set_light(self, light):
        """
        light: engine.lighting.LightState of the current hour.
        """
        self.light = light
        self._update_clear_color()

    def _update_clear_color(self):
        """
        The sky is the clear color of the window: it takes the pause filter like the world shaders.
        """
        r, g, b = (float(v) for v in self.light.sky_color)
        gray, dark = self.pause_filter or (0.0, 1.0)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        Window.clearcolor = ((r + (luma - r) * gray) * dark,
                             (g + (luma - g) * gray) * dark,
                             (b + (luma - b) * gray) * dark, 1.0)

    def composed_texture(self, key, rgba, wrap='clamp_to_edge'):
        if key not in self.textures:
            self.textures[key] = gl.make_texture(rgba, wrap=wrap)
        return self.textures[key]

    def add_unit(self, unit, texture_name=None, removable=False):
        """
        A unit with equipment gets the body and gear textures built from its items,
        texture_name replaces the body texture. A removable unit keeps its own group for remove_prop().
        """
        composed = unit.figure.composed_textures()
        textures = {group: self.composed_texture(key, rgba) for group, (key, rgba) in composed.items()}
        if texture_name or equipment.TEXTURE_BODY not in textures:
            body_name = (texture_name or unit.figure.default_texture_name() or '').lower()
            textures[equipment.TEXTURE_BODY] = self.texture(body_name)
            body_key, body_rgba = body_name, lambda: self.game_data.texture(body_name) if body_name else None
            body_wrap = 'repeat'
        else:
            body_key, rgba = composed[equipment.TEXTURE_BODY]
            body_rgba = lambda: rgba
            body_wrap = 'clamp_to_edge'
        canvas = self.canvas
        if removable:
            canvas = Canvas()
            self.canvas.add(canvas)
        view = UnitView(unit, canvas, textures)
        view.body_key, view.body_rgba, view.body_wrap = body_key, body_rgba, body_wrap
        view.group = canvas
        self.unit_views.append(view)
        return view

    def update_wounds(self, view):
        """
        537360 (partly): the body texture of the unit with the wounds of its parts (engine.woundskin).
        """
        unit = view.unit
        fighter = getattr(unit, 'fighter', None)
        if fighter is None or not fighter.parts:
            return
        humanoid = getattr(unit, 'humanoid', False)
        gear = unit.figure.equipment
        wound_levels = woundskin.levels(fighter.parts, humanoid, gear.armors if gear is not None else None)
        if wound_levels == view.wound_levels:
            return
        view.wound_levels = wound_levels
        names = [name for name in woundskin.layer_names(unit.figure.model_name, wound_levels, humanoid)
                 if self.game_data.texture(name) is not None]
        if not names:
            view.set_body_texture(view.body_texture)
            return
        base = view.body_rgba()
        if base is None:
            return
        key = '%s#wounds+%s' % (view.body_key, '+'.join(names))
        rgba = self.game_data.cached(('wounded', key), lambda: woundskin.wounded(
            base, [self.game_data.texture(name) for name in names], humanoid))
        view.set_body_texture(self.composed_texture(key, rgba, wrap=view.body_wrap))

    def replace_unit_view(self, unit, texture_name=None):
        """
        The figure of a unit is built again after its equipment changed (the loot of a body): the old view
        is hidden and taken from the list, the new one is removable.
        """
        old_view = next((view for view in self.unit_views if view.unit is unit), None)
        if old_view is not None:
            old_view.hide()
            if old_view.group is not self.canvas:
                self.canvas.remove(old_view.group)
            self.unit_views.remove(old_view)
        return self.add_unit(unit, texture_name=texture_name, removable=True)

    def object_textures(self, texture_name, second_texture_name=None):
        textures = {equipment.TEXTURE_BODY: self.texture(texture_name)}
        if second_texture_name and self.game_data.texture(second_texture_name) is not None:
            textures[figure_module.TEXTURE_SECOND] = self.texture(second_texture_name)
        return textures

    def add_static_object(self, figure_model, world_matrix, texture_name, second_texture_name=None):
        view = StaticObjectView(figure_model, world_matrix, self.canvas, self.object_textures(texture_name, second_texture_name))
        self.static_views.append(view)
        return view

    def add_lever(self, figure_model, lever, world_matrix, texture_name, second_texture_name=None):
        view = LeverView(figure_model, lever, world_matrix, self.canvas,
                         self.object_textures(texture_name, second_texture_name))
        self.lever_views.append(view)
        return view

    def add_prop(self, prop, texture_name, second_texture_name=None, interface=False, texture=None, removable=False,
                 textures=None):
        """
        An animated object (engine.prop.Prop) of the world or of the 3D interface, texture (a Kivy texture)
        replaces the one of texture_name, textures is a whole group -> Kivy texture map which wins over both.
        A removable object of the world keeps its own group for remove_prop().
        """
        canvas = self.canvas
        if removable and not interface:
            canvas = Canvas()
            self.canvas.add(canvas)
        if interface:
            # every model of the interface keeps its own group, so it can go away alone
            canvas = Canvas()
            self.interface_canvas.add(canvas)
        object_textures = self.object_textures(texture_name, second_texture_name)
        if textures:
            object_textures.update(textures)
        if texture is not None:
            object_textures[equipment.TEXTURE_BODY] = texture
        view = UnitView(prop, canvas, object_textures)
        view.group = canvas
        (self.interface_views if interface else self.unit_views).append(view)
        return view

    def remove_prop(self, view):
        if view in self.unit_views:
            self.unit_views.remove(view)
            self.canvas.remove(view.group)

    def remove_interface_prop(self, view):
        if view in self.interface_views:
            self.interface_views.remove(view)
            self.interface_canvas.remove(view.group)

    def show_interface_prop(self, view, visible):
        shown = view.group in self.interface_canvas.children
        if visible and not shown:
            self.interface_canvas.add(view.group)
        elif not visible and shown:
            self.interface_canvas.remove(view.group)

    def clear_interface(self):
        self.interface_canvas.clear()
        self.interface_views = []

    def hud_texture(self, texture_name):
        key = 'hud:%s' % (texture_name or '').lower()
        if key not in self.textures:
            rgba = self.game_data.texture(texture_name) if texture_name else None
            if rgba is None:
                self.textures[key] = gl.white_texture()
            else:
                self.textures[key] = gl.make_texture(rgba, mipmap=False, wrap='clamp_to_edge')
        return self.textures[key]

    def set_hud_layer(self, name, elements):
        """
        Replaces the elements (engine.hud.Element) of a layer, the far ones are drawn first.
        """
        group = self.hud_layers.get(name)
        if group is None:
            group = InstructionGroup()
            self.hud_layers[name] = group
            self.hud_canvas.add(group)
        else:
            # a group emptied in place keeps its old meshes on the screen, a new one in the same place does not
            new_group = InstructionGroup()
            self.hud_canvas.insert(self.hud_canvas.indexof(group), new_group)
            self.hud_canvas.remove(group)
            group = new_group
            self.hud_layers[name] = group
        for element in sorted(elements, key=lambda e: -e.z):
            vertices = []
            for x, y, u, v in element.vertices():
                vertices.extend(gl.interface_point(x, y, element.z).tolist())
                vertices.extend((u, v))
            if not vertices:
                continue
            group.add(Color(*element.color))
            group.add(Mesh(vertices=vertices, indices=list(range(len(vertices) // 5)), fmt=gl.HUD_VERTEX_FORMAT,
                           mode='triangles', texture=self.hud_texture(element.texture_name)))

    def clear_hud(self):
        self.hud_canvas.clear()
        self.hud_layers = {}

    def projection(self):
        aspect = max(self.width, 1) / float(max(self.height, 1))
        return gl.perspective(FIELD_OF_VIEW, aspect, 0.1, 500.0)

    def set_point_lights(self, lights):
        """
        The point lights of the spells: (point, color, radius) in the world; the shaders take the
        gl.POINT_LIGHTS of them nearest to the camera target.
        """
        target = np.asarray(self.camera.target, dtype=np.float64)
        lights = [light for light in lights if light[2] > 0.0]
        lights.sort(key=lambda light: float(np.sum((np.asarray(light[0]) - target) ** 2)))
        self.point_lights = lights[:gl.POINT_LIGHTS]

    def point_light_uniforms(self, view):
        uniforms = {}
        for i in range(gl.POINT_LIGHTS):
            if i < len(self.point_lights):
                point, color, radius = self.point_lights[i]
                eye = view @ np.array((point[0], point[1], point[2], 1.0))
                uniforms['point_light%d' % i] = [float(eye[0]), float(eye[1]), float(eye[2]), 1.0 / (radius * radius)]
                uniforms['point_color%d' % i] = [float(color[0]), float(color[1]), float(color[2]), radius * radius]
            else:
                uniforms['point_light%d' % i] = [0.0, 0.0, 0.0, 0.0]
                uniforms['point_color%d' % i] = [0.0, 0.0, 0.0, 0.0]
        return uniforms

    def set_screen_rect(self, unit_view, view_projection):
        """
        The rectangle of the box of the figure (its size in the pose of rest, turned all around) on the screen.
        """
        unit = unit_view.unit
        if not hasattr(unit, 'x') or not hasattr(unit, 'size') or getattr(unit, 'unseen', False):
            unit_view.screen_rect = None
            return
        size = unit.size()
        if size is None:
            unit_view.screen_rect = None
            return
        radius = SCREEN_RECT_SCALE * 0.5 * max(float(size[0]), float(size[1]))
        height = SCREEN_RECT_SCALE * float(size[2])
        corners = np.array([(unit.x + dx, unit.y + dy, unit.z + dz, 1.0)
                            for dx in (-radius, radius) for dy in (-radius, radius) for dz in (0.0, height)])
        clip = corners @ view_projection.T
        w = clip[:, 3]
        if (w <= 0.05).any():
            unit_view.screen_rect = None
            return
        nx = clip[:, 0] / w
        ny = clip[:, 1] / w
        width, height_px = max(self.width, 1), max(self.height, 1)
        xs = self.x + (nx + 1.0) * 0.5 * width
        ys = self.y + (ny + 1.0) * 0.5 * height_px
        unit_view.screen_rect = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        unit_view.screen_depth = float(w.min())

    def units_at(self, x, y):
        """
        5718b0: the units whose rectangles of the last draw hold the point, the nearest first.
        """
        found = []
        for unit_view in self.unit_views:
            rect = unit_view.screen_rect
            if rect is not None and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                found.append((unit_view.screen_depth, id(unit_view), unit_view.unit))
        found.sort()
        return [unit for depth, key, unit in found]

    @staticmethod
    def on_screen(unit, view_projection, scale_x, scale_y):
        """
        Whether the figure of the unit can be on the screen: its point inside the frustum with a margin of
        UNIT_SCREEN_MARGIN units of the world (scale_x, scale_y: the scales of the projection). The figures out
        of it keep their last matrices.
        """
        if not hasattr(unit, 'x'):
            # a prop of the interface or of a script: few of them, always drawn
            return True
        x, y, z, w = view_projection @ np.array((unit.x, unit.y, unit.z + 1.0, 1.0))
        if w < -UNIT_SCREEN_MARGIN:
            return False
        return abs(x) <= w + UNIT_SCREEN_MARGIN * scale_x and abs(y) <= w + UNIT_SCREEN_MARGIN * scale_y

    def redraw(self):
        now = time.perf_counter()
        if self.shake_last is not None and hasattr(self.camera, 'advance_shake'):
            self.camera.advance_shake((now - self.shake_last) / SHAKE_TICK_SECONDS)
        self.shake_last = now
        view = self.camera.view()
        projection = gl.kivy_matrix(self.projection())
        modelview = gl.kivy_matrix(view)
        # the shaders want the direction to the light in the eye space
        to_light = view[0:3, 0:3] @ -np.asarray(self.light.sun_direction, dtype=np.float64)
        uniforms = {
            'light_dir': [float(v) for v in to_light / np.linalg.norm(to_light)],
            'sun_color': [float(v) for v in self.light.sun_color],
            'ambient_color': [float(v) for v in self.light.ambient_color],
            'fog_color': [float(v) for v in self.light.sky_color],
            'fog_start': float(self.light.fog_start),
            'fog_end': float(self.light.fog_end),
        }
        uniforms.update(self.point_light_uniforms(view))
        self.particle_canvas['projection_mat'] = projection
        self.particle_canvas['modelview_mat'] = modelview
        self.particle_canvas['fog_start'] = float(self.light.fog_start)
        self.particle_canvas['fog_end'] = float(self.light.fog_end)
        for context in (self.canvas, self.water_canvas):
            context['projection_mat'] = projection
            context['modelview_mat'] = modelview
            for name, value in uniforms.items():
                context[name] = value
        self.water_canvas['wave_time'] = float(self.wave_clock.time_phase)
        self.water_canvas['wave_amplitude'] = float(self.wind.amplitude)
        self.water_canvas['sway_phase'] = float(self.wave_clock.sway_phase)
        self.water_canvas['highlight'] = float(self.wind.strength * water.HIGHLIGHT)
        self.interface_canvas['projection_mat'] = projection
        self.interface_canvas['modelview_mat'] = gl.kivy_matrix(gl.INTERFACE_VIEW)
        for name, value in INTERFACE_LIGHT.items():
            self.interface_canvas[name] = value
        self.hud_canvas['projection_mat'] = projection
        self.hud_canvas['modelview_mat'] = gl.kivy_matrix(gl.INTERFACE_VIEW)
        for interface_view in self.interface_views:
            interface_view.update()
        projection_matrix = np.asarray(self.projection())
        view_projection = projection_matrix @ view
        scale_x, scale_y = abs(projection_matrix[0, 0]), abs(projection_matrix[1, 1])
        for unit_view in self.unit_views:
            if self.on_screen(unit_view.unit, view_projection, scale_x, scale_y):
                unit_view.update()
                self.set_screen_rect(unit_view, view_projection)
            else:
                unit_view.screen_rect = None
        view_key = view.tobytes()
        if view_key != self.static_view_key:
            self.static_view_key = view_key
            for static_view in self.static_views:
                static_view.update()
        for lever_view in self.lever_views:
            lever_view.update()

    def screen_point(self, point):
        """
        (x, y) of a point of the world on the screen, None when it is behind the eye.
        """
        clip = self.projection() @ self.camera.view() @ np.array((point[0], point[1], point[2], 1.0))
        if clip[3] <= 0.0:
            return None
        ndc = clip[0:3] / clip[3]
        return (self.x + (ndc[0] + 1.0) * 0.5 * self.width,
                self.y + (1.0 - ndc[1]) * 0.5 * self.height)

    def pick_ground(self, x, y):
        """
        The point of the terrain under the screen position, None when the ray does not hit it.
        """
        if self.terrain is None:
            return None
        nx = 2.0 * (x - self.x) / max(self.width, 1) - 1.0
        ny = 2.0 * (y - self.y) / max(self.height, 1) - 1.0
        inverse = np.linalg.inv(self.projection() @ self.camera.view())
        near = inverse @ np.array([nx, ny, -1.0, 1.0])
        far = inverse @ np.array([nx, ny, 1.0, 1.0])
        near, far = near[0:3] / near[3], far[0:3] / far[3]
        direction = far - near
        length = np.linalg.norm(direction)
        direction /= length
        step = 0.25
        t = 0.0
        while t < min(length, 400.0):
            p = near + direction * t
            if p[2] <= self.terrain.height_at(p[0], p[1]):
                return p
            t += step
        return None
