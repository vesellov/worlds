"""
The particle sources, the point lights and the sound sources which the scripts make (the lists +0x18 and +0x34
of the script world of game.exe).

CreateParticleSource( id, x, y, z, size, name ) (0xe, 4bd9a0) finds the name in the table 78b120
(engine.particles.SCRIPT_TYPES) and makes the effect 0x4a with that type of particles at the point,
the size goes to +0x15c of the effect (its scale); a source with the same id is removed first, an id below 0
takes the next free number of the world going down (+0x68). The record keeps the ticks of the table:
200 for the types which end, -1 for those which stay.
MoveParticleSource( id, x, y, z, ticks, flag ) (0x10, 4bdbb0) moves the source to the point along a line over
the ticks, SetParticleSourceSize( id, size ) (0xba, 679270) sets the scale again, DeleteParticleSource (0x11, 4bdc90)
removes it. AttachParticles( id, o ) (0xe7, 679240) puts the effect of the source on the object (+0x168, bit 2
of +8): it follows the point of its holder (484c00; the types with +0x13c = 7 take the point "bd" of the figure,
here the middle of the height of the unit) while the emitter keeps it (StartTrans leaves it after 31 ticks).
CreatePointLight( id, x, y, z, radius, r, g, b ) (0xb, 4bd6a0) makes the effect 0x4b: the light of a spell
(engine.magic.SpellLight) with the color / 255; MovePointLight (0xc) and DeletePointLight (0xf, 4bcff0) do the same
as for the particles.

CreateFXSource( id, x, y, z, near, far, wav ) (0x16, 4bd310) keeps a record in the list +0x34 (an id below 0 takes
the next free number going down, +0x6c) and makes the effect 0x48 (CEffectSound) at the point with the distances
+0x24c, +0x250 and the name +0x14c; the effect keeps the loop +0x254 = 1 of its init (vf18), so the sound plays
round until DeleteFXSource( id ) (0x19, 4c6370) removes it (CreateFX, 0x17, sets the loop to 0 and plays it once).
CreateRandomizedFXSource (0x18) makes a record without the effect which plays it now and then; no map uses it.

CreateLightning( id, x0, y0, z0, x1, y1, z1, power ) (0xf, 4bcff0) keeps a record in the list +0x50 (an id below 0
takes the next free number of +0x68 too) and makes the effect 0x4d (CEffectLightning, 67a670) between the two points
of the world with the power, engine.lightning draws it; it stays until DeleteLightning( id ) (0x1a) removes it.

The size is the scale of engine.particles.Emitter as it is: -1 makes the sparks of Teleport fall into the point
(as the spell does at its target). The remake's choices: a record whose ticks ran out stops emitting and goes away
with its last particles; a size set on a source of the size 0 is kept for nothing; the flag of the moves is not read.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import lightning
from engine import magic
from engine import particles

#------------------------------------------------------------------------------

# +0x68 (the particles and the lights) and +0x6c (the sounds) of the script world: the ids given to the sources
# made with an id below 0
FIRST_FREE_ID = -1

#------------------------------------------------------------------------------


class Move(object):
    """
    A line over some ticks: the step of every tick and the ticks which are left.
    """

    def __init__(self, start, end, ticks):
        self.ticks = max(int(round(ticks)), 1)
        self.step = tuple((b - a) / float(self.ticks) for a, b in zip(start, end))

    def advance(self, position):
        self.ticks -= 1
        return tuple(a + d for a, d in zip(position, self.step))


class ParticleSource(object):

    def __init__(self, emitter, size, ticks):
        self.emitter = emitter
        self.position = tuple(emitter.position)
        self.size = size
        self.ticks = ticks
        self.move = None
        # AttachParticles: the unit whose point the source follows
        self.holder = None

    def update(self):
        if self.move is not None:
            self.position = self.move.advance(self.position)
            if self.move.ticks <= 0:
                self.move = None
        holder = self.holder
        if holder is not None and self.emitter.attached and not getattr(holder, 'is_dead', False):
            self.position = (holder.x, holder.y, holder.z + holder.height() * 0.5)
        if self.ticks > 0:
            self.ticks -= 1
            if self.ticks == 0:
                self.emitter.stop()
        return self.emitter.update(self.position)


class PointLight(object):

    def __init__(self, light):
        self.light = light
        self.move = None

    def update(self):
        if self.move is not None:
            self.light.position = self.move.advance(self.light.position)
            if self.move.ticks <= 0:
                self.move = None
        return self.light.update()


class ScriptSources(object):
    """
    The sources of one world; make_emitter(code, position, scale) gives an engine.particles.Emitter
    and raises ValueError for a type which the remake has not.
    """

    def __init__(self, make_emitter=None):
        self.make_emitter = make_emitter or (lambda code, position, scale: particles.Emitter(code, position, scale))
        self.particles = {}
        self.lights = {}
        self.sounds = {}
        # id -> engine.lightning.Lightning
        self.lightnings = {}
        self.free_id = FIRST_FREE_ID
        self.free_sound_id = FIRST_FREE_ID

    def next_id(self, nid):
        nid = int(round(nid))
        if nid < 0:
            nid = self.free_id
            self.free_id -= 1
        return nid

    def create_particles(self, nid, position, size, name):
        """
        False when the name is not in the table or the remake has not its type.
        """
        found = particles.SCRIPT_TYPES.get(name.lower())
        if found is None:
            return False
        code, ticks = found
        nid = self.next_id(nid)
        self.particles.pop(nid, None)
        try:
            emitter = self.make_emitter(code, tuple(position), float(size))
        except ValueError:
            return False
        self.particles[nid] = ParticleSource(emitter, float(size), ticks)
        return True

    def set_particles_size(self, nid, size):
        source = self.particles.get(int(round(nid)))
        if source is None:
            return
        if source.size != 0.0:
            source.emitter.set_scale(float(size) / source.size)
            source.size = float(size)

    def move_particles(self, nid, position, ticks):
        source = self.particles.get(int(round(nid)))
        if source is not None:
            source.move = Move(source.position, position, ticks)

    def attach_particles(self, nid, holder):
        source = self.particles.get(int(round(nid)))
        if source is not None and holder is not None and hasattr(holder, 'height'):
            source.holder = holder
            source.emitter.attached = True

    def delete_particles(self, nid):
        self.particles.pop(int(round(nid)), None)

    def create_light(self, nid, position, radius, color):
        nid = self.next_id(nid)
        rgb = tuple(max(0.0, min(c, 255.0)) / 255.0 for c in color)
        self.lights[nid] = PointLight(magic.SpellLight(position, rgb, radius))

    def move_light(self, nid, position, ticks):
        light = self.lights.get(int(round(nid)))
        if light is not None:
            light.move = Move(light.light.position, position, ticks)

    def delete_light(self, nid):
        self.lights.pop(int(round(nid)), None)

    def create_sound(self, nid, position, near, far, name):
        """
        CreateFXSource: the record of the id takes the new sound.
        """
        nid = int(round(nid))
        if nid < 0:
            nid = self.free_sound_id
            self.free_sound_id -= 1
        self.sounds[nid] = magic.SpellSound(name, position, near, far)
        return nid

    def delete_sound(self, nid):
        self.sounds.pop(int(round(nid)), None)

    def create_lightning(self, nid, start, end, power):
        nid = self.next_id(nid)
        self.lightnings[nid] = lightning.Lightning(tuple(start), tuple(end), float(power))
        return nid

    def delete_lightning(self, nid):
        self.lightnings.pop(int(round(nid)), None)

    def bolts(self):
        return list(self.lightnings.values())

    def sound_loops(self):
        return list(self.sounds.values())

    def update(self):
        """
        One tick of the sources and the lights, the ended ones are removed.
        """
        for nid, source in list(self.particles.items()):
            if not source.update():
                del self.particles[nid]
        for nid, light in list(self.lights.items()):
            if not light.update():
                del self.lights[nid]
        for bolt in self.lightnings.values():
            bolt.update()

    def emitters(self):
        return [source.emitter for source in self.particles.values()]

    def spell_lights(self):
        return [light.light for light in self.lights.values()]
