"""
The flying numbers over the units on the screen (game.exe 5e18e0, engine.flyinghp).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from kivy.graphics import Color, Mesh
from kivy.uix.widget import Widget

#------------------------------------------------------------------------------

from engine import flyinghp
from client import gl

#------------------------------------------------------------------------------

# game.exe draws its numbers in the pixels of the screen of 600 lines
SCREEN_HEIGHT = 600.0

#------------------------------------------------------------------------------


class FlyingNumbersView(Widget):
    """
    Draws engine.flyinghp.FlyingNumbers over the 3D scene, under the interface.
    """

    def __init__(self, game_data, **kwargs):
        Widget.__init__(self, **kwargs)
        self.game_data = game_data
        self.texture = None
        # the client part of every unit: the share of its health which it saw last (52ef90 (partly))
        self.health_parts = {}

    def __repr__(self):
        return 'FlyingNumbersView()'

    def _texture(self):
        if self.texture is None:
            rgba = self.game_data.texture(flyinghp.TEXTURE_NAME)
            self.texture = gl.make_texture(rgba, mipmap=False, wrap='clamp_to_edge') if rgba is not None else False
        return self.texture or None

    def watch(self, units, numbers, now_ms, shown=None):
        """
        52ef90 (partly): a unit whose share of the health fell gives its number with the flags of its blows,
        when shown(unit) allows it.
        """
        seen = {}
        for unit in units:
            fighter = getattr(unit, 'fighter', None)
            if fighter is None or fighter.max_health <= 0.0:
                continue
            part = fighter.health / fighter.max_health
            key = id(unit)
            seen[key] = part
            old = self.health_parts.get(key)
            if old is not None:
                value = flyinghp.lost_health(old, part, fighter.max_health)
                if value is not None and (shown is None or shown(unit)):
                    numbers.add(unit, value, fighter.hit_flags, now_ms)
            fighter.hit_flags = 0
        self.health_parts = seen

    def draw(self, numbers, scene_widget, now_ms, enabled=True):
        """
        5e18e0: the numbers of the units seen on the screen; the option show_flying_hp switches them off.
        """
        numbers.update(now_ms)
        self.canvas.clear()
        texture = self._texture()
        if not enabled or texture is None:
            return
        views = {id(view.unit): view for view in scene_widget.unit_views}
        scale = max(scene_widget.height, 1) / SCREEN_HEIGHT
        top = scene_widget.y + scene_widget.height
        with self.canvas:
            for entry, t in numbers.shown(now_ms):
                view = views.get(id(entry.unit))
                if view is None or view.screen_rect is None or getattr(entry.unit, 'hidden', False):
                    continue
                point = view.head_point()
                if point is None:
                    continue
                center = scene_widget.screen_point(point)
                if center is None:
                    continue
                fighter = getattr(entry.unit, 'fighter', None)
                of_player = fighter is not None and fighter.in_party
                r, g, b = flyinghp.color(entry, of_player)
                Color(r, g, b, 1.0 - t)
                width, height, quads = flyinghp.quads(entry.value, entry.flags)
                # the place in the pixels of the screen of game.exe from the top of the scene
                cx = (center[0] - scene_widget.x) / scale
                cy = (center[1] - scene_widget.y) / scale
                left, upper = flyinghp.place(cx, cy, t, width, height)
                vertices = []
                indices = []
                for x, y, w, h, u0, v_top, u1, v_bottom in quads:
                    x0 = scene_widget.x + (left + x) * scale
                    x1 = x0 + w * scale
                    y1 = top - (upper + y) * scale
                    y0 = y1 - h * scale
                    n = len(vertices) // 4
                    vertices += [x0, y0, u0, v_bottom, x1, y0, u1, v_bottom, x1, y1, u1, v_top, x0, y1, u0, v_top]
                    indices += [n, n + 1, n + 2, n, n + 2, n + 3]
                Mesh(vertices=vertices, indices=indices, mode='triangles', texture=texture)
