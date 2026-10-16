import os

os.environ['KIVY_WINDOW'] = 'sdl2'
os.environ['KIVY_TEXT'] = 'sdl2'
os.environ['KIVY_IMAGE'] = 'sdl2'
os.environ['KIVY_DPI'] = '192'
os.environ['KIVY_METRICS_DENSITY'] = '2'


# Importing Window creates the GL context; it must exist before any RenderContext/Shader
from kivy.core.window import Window
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import RenderContext, Color, Rectangle
from kivy.vector import Vector

# Basic vertex shader
VERTEX_SHADER = '''
#ifdef GL_ES
    precision highp float;
#endif
attribute vec2     vPosition;
attribute vec2     vTexCoords0;
varying vec2       tex_coord0;
uniform mat4       modelview_mat;
uniform mat4       projection_mat;

void main() {
    tex_coord0 = vTexCoords0;
    gl_Position = projection_mat * modelview_mat * vec4(vPosition, 0.0, 1.0);
}
'''

# Fragment shader that accepts uniform updates from multi-touch
FRAGMENT_SHADER = '''
#ifdef GL_ES
    precision highp float;
#endif
varying vec2 tex_coord0;
uniform vec2 u_center;
uniform float u_zoom;

void main() {
    // Simple custom visualization based on touch metrics
    vec2 st = tex_coord0 - u_center;
    float d = length(st) * u_zoom;
    vec3 color = vec3(smoothstep(0.4, 0.41, d));
    gl_FragColor = vec4(color, 1.0);
}
'''

class MultiTouchShaderWidget(Widget):
    def __init__(self, **kwargs):
        # 1. Substitute the default canvas with a RenderContext
        self.canvas = RenderContext()
        # fs first: the default fs needs frag_color, which our vs doesn't provide
        self.canvas.shader.fs = FRAGMENT_SHADER
        self.canvas.shader.vs = VERTEX_SHADER
        
        super(MultiTouchShaderWidget, self).__init__(**kwargs)
        
        # 2. Initialize shader uniform variables
        self.canvas['u_zoom'] = 1.0
        self.canvas['u_center'] = [0.5, 0.5]
        
        # 3. Track active unique touches
        self.active_touches = {}
        
        with self.canvas:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
            
        self.bind(size=self._update_rect, pos=self._update_rect)

    def _update_rect(self, instance, value):
        self.rect.pos = self.pos
        self.rect.size = self.size
        # Feed modern projection matrices to the RenderContext
        self.canvas['projection_mat'] = Window.render_context['projection_mat']
        self.canvas['modelview_mat'] = Window.render_context['modelview_mat']

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # Grab the touch internally to track its movement path
            touch.grab(self)
            self.active_touches[touch.id] = touch.pos
            
            # If a two-finger touch gesture is initialized, calculate starting distance
            if len(self.active_touches) == 2:
                self.initial_dist = self._get_touch_distance()
            return True
        return super(MultiTouchShaderWidget, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            # Update the stored position for this specific finger
            self.active_touches[touch.id] = touch.pos
            
            if len(self.active_touches) == 2:
                # Handle the two-finger logic
                current_dist = self._get_touch_distance()
                if self.initial_dist > 0:
                    # Calculate scale/zoom factor change relative to initial contact
                    scale_factor = current_dist / self.initial_dist
                    self.canvas['u_zoom'] = float(1.0 / scale_factor)
                
                # Calculate mid-point between the two fingers to adjust shader center point
                positions = list(self.active_touches.values())
                mid_x = (positions[0][0] + positions[1][0]) / 2.0
                mid_y = (positions[0][1] + positions[1][1]) / 2.0
                
                # Normalize coordinates to 0.0 - 1.0 space for fragment shader
                self.canvas['u_center'] = [mid_x / self.width, mid_y / self.height]
                
            return True
        return super(MultiTouchShaderWidget, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            if touch.id in self.active_touches:
                del self.active_touches[touch.id]
            return True
        return super(MultiTouchShaderWidget, self).on_touch_up(touch)

    def _get_touch_distance(self):
        # Extract the positions of the two active touches
        positions = list(self.active_touches.values())
        p1 = Vector(positions[0])
        p2 = Vector(positions[1])
        return p1.distance(p2)

class ShaderApp(App):
    def build(self):
        return MultiTouchShaderWidget()

from kivy.uix.scatter import Scatter

class MyApp(App):
    def build(self):
        root = Scatter(do_rotation=True, do_scale=True, do_translation=True)
        shader_widget = MultiTouchShaderWidget() # Contains RenderContext
        root.add_widget(shader_widget)
        return root

if __name__ == '__main__':
    MyApp().run()
