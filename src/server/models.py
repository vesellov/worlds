"""
Data models of the game world, shared by the server and the client.

Server is the only owner of the models: it creates them, changes them and sends them to the clients over the web socket.
Client never changes the models by itself, it only renders them and sends user input to the server.
Module is plain Python without any dependencies, so it can be imported by the server (no Kivy there) and by the client.
"""

PROTOCOL_VERSION = 1

DEFAULT_PORT = 8280

TICK_RATE = 20

#------------------------------------------------------------------------------

# client -> server
MSG_JOIN = 'join'
MSG_INPUT = 'input'

# server -> client
MSG_WELCOME = 'welcome'
MSG_SPAWN = 'spawn'
MSG_STATE = 'state'
MSG_DESPAWN = 'despawn'


def message(msg_type, payload=None):
    return {'t': msg_type, 'p': payload or {}}

#------------------------------------------------------------------------------


class Model(object):

    # all fields which are transferred over the web socket
    fields = ()
    # fields which can change while the game is running, only those are sent with every state update
    state_fields = ()

    def __init__(self, **kwargs):
        defaults = self.defaults()
        for f in self.fields:
            setattr(self, f, kwargs.get(f, defaults.get(f)))

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join(f"{f}={getattr(self, f)!r}" for f in self.fields)})'

    def defaults(self):
        return {}

    def to_dict(self):
        d = {}
        for f in self.fields:
            v = getattr(self, f)
            d[f] = v.to_dict() if isinstance(v, Model) else v
        return d

    def to_state(self):
        return {f: getattr(self, f) for f in ('id', ) + self.state_fields}

    def update(self, d):
        for f in self.state_fields:
            if f in d:
                setattr(self, f, d[f])

    @classmethod
    def from_dict(cls, d):
        return cls(**{f: d[f] for f in cls.fields if f in d})


class World(Model):
    fields = ('name', 'width', 'height', 'tick_rate', 'protocol')

    def defaults(self):
        return {'tick_rate': TICK_RATE, 'protocol': PROTOCOL_VERSION}


class Map(Model):
    # land segments info, elevation and tiles are loaded from the map files, those are not transferred to clients
    fields = ('width', 'height', 'water_level')


class Team(Model):
    fields = ('name', 'members')

    def defaults(self):
        return {'members': []}


class Template(Model):
    # info about the 3d model and what is required to build and render the 3d model on the client device
    # figure name, coefs, textures, skin, hair, wears, weapon, etc.
    fields = ('model_name', 'skin', 'hair', 'wears', 'weapon', 'texture', 'coefs', 'elevation_correction')

    def defaults(self):
        return {'skin': 0, 'wears': [], 'coefs': [0.5, 0.5, 0.75]}


class Object(Model):
    # all related to position a static object on the map
    # name, template, coordinates, orientation
    kind = 'object'
    fields = ('id', 'kind', 'name', 'template', 'w', 'h', 'shift_w', 'shift_h', 'direction')
    state_fields = ('w', 'h', 'shift_w', 'shift_h', 'direction')

    def defaults(self):
        return {'kind': self.kind, 'shift_w': 0.5, 'shift_h': 0.5, 'direction': 0.0}

    @classmethod
    def from_dict(cls, d):
        o = super().from_dict(d)
        if isinstance(o.template, dict):
            o.template = Template.from_dict(o.template)
        return o


class Unit(Object):
    # speed, acceleration, mass, turn_speed
    kind = 'unit'
    fields = Object.fields + ('speed', 'max_speed', 'acceleration_value', 'turn_speed', 'action', 'animation')
    state_fields = Object.state_fields + ('speed', 'action', 'animation')

    def defaults(self):
        d = super().defaults()
        d.update({'speed': 0.0, 'max_speed': 0.06, 'acceleration_value': 0.004, 'turn_speed': 5.0, 'action': 'idle'})
        return d


class Hero(Unit):
    # unit controlled by one of the connected players
    kind = 'hero'
    fields = Unit.fields + ('owner', )


class Monster(Unit):
    kind = 'monster'


class Building(Object):
    kind = 'building'


class Controls(Model):
    # what the player currently wants his hero to do, client sends it every time the user input changes
    # move: 1 forward, -1 backward, 0 stop
    # turn: 1 left, -1 right, 0 no turn
    # target_direction: direction (in degrees) the hero must turn to, set by the joystick, None if not used
    fields = ('move', 'turn', 'target_direction')

    def defaults(self):
        return {'move': 0, 'turn': 0}


KINDS = {c.kind: c for c in (Object, Unit, Hero, Monster, Building)}


def object_from_dict(d):
    return KINDS.get(d.get('kind'), Object).from_dict(d)
