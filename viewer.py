import sys
import pprint

from kivy.config import Config
Config.set('graphics', 'window_state', 'maximized')

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.resources import resource_find
from kivy.graphics.transformation import Matrix  # @UnresolvedImport
from kivy.graphics.opengl import glGetError, glEnable, glDisable, GL_DEPTH_TEST  # @UnresolvedImport
from kivy.graphics.instructions import InstructionGroup  # @UnresolvedImport
from kivy.graphics.context_instructions import Transform  # @UnresolvedImport
from kivy.graphics import (
    RenderContext, Callback, BindTexture, ChangeState, PushMatrix, PopMatrix, Scale,
    PushState, PopState, Color, Translate, Rotate, Mesh, UpdateNormalMatrix,
)

from scene import SceneData, load_human_female


def ignore_undertouch(func):
    def wrap(self, touch):
        glst = touch.grab_list
        if len(glst) == 0 or (self is glst[ 0 ]()):
            return func(self, touch)
    return wrap


vertex_shader_src = """
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3  v_pos;
attribute vec3  v_normal;
attribute vec2  v_tc0;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;

varying vec2 tex_coord0;
varying vec4 normal_vec;
varying vec4 vertex_pos;

void main (void) {
    vec4 pos = modelview_mat * vec4(v_pos, 1.0);
    vertex_pos = pos;
    normal_vec = vec4(v_normal,0.0);
    gl_Position = projection_mat * pos;
    tex_coord0 = v_tc0;
}
"""

fragment_shader_src = """
#ifdef GL_ES
    precision highp float;
#endif

varying vec4 normal_vec;
varying vec4 vertex_pos;
varying vec2 tex_coord0;

uniform sampler2D texture0;
uniform mat4 normal_mat;
uniform vec4 line_color;

void main (void) {
    gl_FragColor = line_color * texture2D(texture0, tex_coord0);
}
"""


class Renderer(Widget):

    SCALE_FACTOR = 0.05
    MAX_SCALE = 10.0
    MIN_SCALE = 0.1
    ROTATE_SPEED = 1.

    def __init__(self, **kwargs):
        self.angle_x = 0
        self.angle_y = 0
        self.angle_z = 0
        self.coord_x = 0
        self.coord_y = 0
        self.coord_z = 3
        self._touches = []
        self.canvas = RenderContext(compute_normal_mat=True)
        # self.canvas.shader.source = resource_find('data/shader.glsl')
        self.canvas.shader.fs = fragment_shader_src
        self.canvas.shader.vs = vertex_shader_src
        self.container = None
        self.scene = SceneData()
        load_human_female(self.scene)
        for arg in sys.argv[1:]:
            if arg.endswith('.yaml'):
                self.scene.load_yaml_file(resource_find(arg))
            elif arg.endswith('.obj'):
                self.scene.load_obj_file(resource_find(arg))
        self.loaded = set()
        super(Renderer, self).__init__(**kwargs)
        with self.canvas:
            self.cb = Callback(self.setup_gl_context)
            PushMatrix()
            self.setup_scene()
            PopMatrix()
            self.cb = Callback(self.reset_gl_context)
        self.canvas['texture0'] = 1
        Clock.schedule_interval(self.update_glsl, 1 / 60.)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        print(f'loaded objects: {list(self.scene.objects.keys())}')
        for name in self.scene.objects.keys():
            self.add_object(name)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'escape':
            App.get_running_app().stop()
        elif keycode[1] == 'w':
            self.angle_x += 2
        elif keycode[1] == 's':
            self.angle_x -= 2
        elif keycode[1] == 'a':
            self.angle_y += 2
        elif keycode[1] == 'd':
            self.angle_y -= 2
        elif keycode[1] == 'q':
            self.angle_z += 2
        elif keycode[1] == 'e':
            self.angle_z -= 2
        elif keycode[1] == 'g':
            self.coord_x -= 0.5
        elif keycode[1] == 'j':
            self.coord_x += 0.5
        elif keycode[1] == 'y':
            self.coord_y += 0.5
        elif keycode[1] == 'h':
            self.coord_y -= 0.5
        elif keycode[1] == 't':
            self.coord_z -= 0.5
        elif keycode[1] == 'u':
            self.coord_z += 0.5
        elif keycode[1] == 'n':
            if self.container:
                for m in list(self.scene.objects.values()):
                    if m.name not in self.loaded:
                        self.add_object(m.name)
                        break
        elif keycode[1] == 'r':
            if self.container:
                if self.loaded:
                    cur_obj_name = list(self.loaded)[0]
                    self.remove_object(cur_obj_name)
        return True

    @ignore_undertouch
    def on_touch_down(self, touch):
        touch.grab(self)
        self._touches.append(touch)
        if 'button' in touch.profile and touch.button in ('scrollup', 'scrolldown'):
            if touch.button == "scrolldown":
                scale = self.SCALE_FACTOR
            if touch.button == "scrollup":
                scale = -self.SCALE_FACTOR
            xyz = self.global_scale.xyz
            scale = xyz[0] + scale
            if scale < self.MAX_SCALE and scale > self.MIN_SCALE:
                self.global_scale.xyz = (scale, scale, scale)

    @ignore_undertouch
    def on_touch_up(self, touch):
        touch.ungrab(self)
        if touch in self._touches:
            self._touches.remove(touch)

    def define_rotate_angle(self, touch):
        x_angle = (touch.dx / self.width) * 360. * self.ROTATE_SPEED
        y_angle = -1 * (touch.dy / self.height) * 360. * self.ROTATE_SPEED
        return x_angle, y_angle

    @ignore_undertouch
    def on_touch_move(self, touch):
        if touch in self._touches and touch.grab_current == self:
            if len(self._touches) == 1:
            # here do just rotation        
                ax, ay = self.define_rotate_angle(touch)
                self.global_rotate_y.angle += ax
                self.global_rotate_x.angle += ay

    def add_object(self, name):
        if self.container:
            if name in self.scene.objects:
                scene_object = self.scene.objects.get(name)
                scene_object.bone_translate = Translate(
                    scene_object.bone_pos_calculated[0],
                    scene_object.bone_pos_calculated[1],
                    scene_object.bone_pos_calculated[2],
                    group=scene_object.name,
                )
                self.container.add(PushMatrix(group=scene_object.name))
                # self.container.add(scene_object.animation_rotate)
                # self.container.add(scene_object.animation_translate)
                self.container.add(scene_object.bone_translate)
                # TODO: check if we can pass texture data directly to Mesh instruction as a parameter
                self.container.add(BindTexture(source='data/' + scene_object.material['map_Kd'], index=1, group=scene_object.name))
                self.container.add(Mesh(
                    vertices=scene_object.vertices,
                    indices=scene_object.indices,
                    fmt=scene_object.vertex_format,
                    mode='triangles',
                    group=scene_object.name,
                    # texture=<already loaded Texture>,
                ))
                self.container.add(PopMatrix(group=scene_object.name))
                self.loaded.add(name)
                print(f'added {name} on scene with {len(scene_object.vertices)} vertices and {len(scene_object.indices)} indices')

    def remove_object(self, name):
        if self.container:
            if name in self.scene.objects:
                # self.scene.objects[name].shift = None
                # self.scene.objects[name].angle_x = None
                # self.scene.objects[name].angle_y = None
                self.container.remove_group(name)
                self.loaded.remove(name)
                print(f'removed {name} from scene')

    def setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)

    def reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)

    def gl_error(self, text='', kill=True):
        err = glGetError()
        if not err:
            return 
        while err:
            print('## GL ## = ' + text + 'OPENGL Error Code = ' + str(err))
            err = glGetError()
        if kill == True:
            sys.exit(0)

    def update_glsl(self, delta):
        asp = self.width / float(self.height)
        self.gl_error('step1')
        self.canvas['texture0'] = 1
        self.canvas['projection_mat'] = Matrix().view_clip(-asp, asp, -1, 1, 1, 100, 1)
        self.canvas['modelview_mat'] = Matrix().look_at(
            # self.coord_x, self.coord_y, self.coord_z,
            0, 0, 3,
            0, 0, 0,
            0, 1, 0,
        )
        self.canvas['diffuse_light'] = (1.0, 1.0, 0.8)
        self.canvas['ambient_light'] = (0.1, 0.1, 0.1)
        self.gl_error('step2')
        # self.rot_x.angle = self.angle_x
        # self.rot_y.angle = self.angle_y
        # self.rot_z.angle = self.angle_z
        # self.transl.x = self.coord_x
        # self.transl.y = self.coord_y
        # self.transl.z = self.coord_z
        # for i in list(self.scene.objects.keys()):
        #     if self.scene.objects[i].angle_x:
        #         self.scene.objects[i].angle_x.angle += delta * 200
        #     if self.scene.objects[i].angle_y:
        #         self.scene.objects[i].angle_y.angle -= delta * 200

    def setup_scene(self):
        PushMatrix()
        self.global_translate = Translate(0, 0, 0)
        self.global_rotate_x = Rotate(0, 1, 0, 0)
        self.global_rotate_y = Rotate(0, 0, 1, 0)
        self.global_scale = Scale(1.0)
        sz = 1
        ChangeState(line_color=(1., 0., 0., 1.))
        Mesh(
            vertices=[1 * sz, 0, 0, 0, 0, 0],
            indices=[0, 1],
            fmt=[(b'v_pos', 3, 'float'), ],
            mode='lines',
        )
        ChangeState(line_color=(0., 1., 0., 1.))
        Mesh(
            vertices=[0, 1 * sz, 0, 0, 0, 0],
            indices=[0, 1],
            fmt=[(b'v_pos', 3, 'float'), ],
            mode='lines',
        )
        ChangeState(line_color=(0., 0., 1., 1.))
        Mesh(
            vertices=[0, 0, 1 * sz, 0, 0, 0],
            indices=[0, 1],
            fmt=[(b'v_pos', 3, 'float'), ],
            mode='lines',
        )
        ChangeState(line_color=(1., 1., 1., 0.9))
        Mesh(
            vertices=[
                -1 * sz, -1 * sz, -1 * sz,
                -1 * sz, -1 * sz, 1 * sz,
                -1 * sz, 1 * sz, 1 * sz,
                -1 * sz, 1 * sz, -1 * sz,
                1 * sz, -1 * sz, -1 * sz,
                1 * sz, -1 * sz, 1 * sz,
                1 * sz, 1 * sz, 1 * sz,
                1 * sz, 1 * sz, -1 * sz,
            ],
            indices=[0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7],
            fmt=[(b'v_pos', 3, 'float'), ],
            mode='lines',
        )
        ChangeState(line_color=(1.,1.,1.,1.))
        Color(1, 1, 1)
        Rotate(-90, 1, 0, 0)
        # UpdateNormalMatrix()
        self.container = InstructionGroup()
        PopMatrix()


class RendererApp(App):

    def build(self):
        return Renderer()


if __name__ == "__main__":
    RendererApp().run()
