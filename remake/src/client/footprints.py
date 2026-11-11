"""
The prints of the steps over the world (G9): client.footprints keeps the field of engine.footprints and
gives its prints to client.scene. They are made for every living unit of the map which walks on the land.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import random

#------------------------------------------------------------------------------

from engine import footprints as footprints_module

#------------------------------------------------------------------------------

class Footprints(object):
    """
    The field of the prints of a map and the layer of the scene which draws it.
    """

    def __init__(self, app):
        self.app = app
        self.map_name = None
        self.field = None
        self.random = random.Random()
        self.ticks_ms = 0.0

    def __repr__(self):
        return 'Footprints(%s, %s)' % (self.map_name, self.field)

    def update(self, dt_ms=55.0):
        app = self.app
        scene = getattr(app, 'scene', None)
        terrain = getattr(app, 'terrain', None)
        if scene is None or terrain is None:
            return
        self.ticks_ms += dt_ms
        if self.ticks_ms < 55.0 and self.field is not None:
            return
        self.ticks_ms = 0.0
        if app.map_name != self.map_name:
            self.map_name = app.map_name
            self.field = footprints_module.FootprintField(terrain.height_at, self.random)
            if hasattr(scene, 'set_footprints'):
                scene.set_footprints([])
        units = ([app.unit] if app.unit is not None else []) + list(app.map_units)
        self.field.update(units)
        marks = []
        for print_ in self.field.prints:
            u0, v0, u1, v1 = footprints_module.cell_uv(print_.cell)
            marks.append((print_.x, print_.y, print_.z, print_.angle, u0, v0, u1, v1,
                          footprints_module.FOOTPRINT_SIZE, print_.alpha))
        scene.set_footprints(marks)

    def clear(self):
        if self.field is not None:
            self.field.clear()
        scene = getattr(self.app, 'scene', None)
        if scene is not None and hasattr(scene, 'set_footprints'):
            scene.set_footprints([])
