"""
Data models of the game world, shared by the server and the client.

Server is the only owner of the models: it creates them, changes them and sends them to the clients over the web socket.
Client never changes the models by itself, it only renders them and sends user input to the server.
Module is plain Python without any dependencies, so it can be imported by the server (no Kivy there) and by the client.
"""

import re
from typing import List, Optional

PROTOCOL_VERSION = 4

# game data (models and textures) is published in several stores: http://p2p-alice.ai/store/<store_id>/
# 1 is made of the eng2001 files, 2 of the astral2006 files. Every template says which store its data comes from,
# models of the world (buildings, plants) are taken from the default store
DEFAULT_STORE_ID = 2

DEFAULT_PORT = 8280

TICK_RATE = 20

#------------------------------------------------------------------------------

# client -> server
MSG_JOIN = 'join'
# {biome_id, sha256}: client wants to enter the biome, sha256 of the biome archive it already has or None
MSG_ENTER = 'enter'
MSG_INPUT = 'input'

# server -> client
# {name, protocol, biomes: [Biome]}: sent after "join", the client enters the first biome of the list
MSG_SERVER_INFO = 'server_info'
# {biome_id, sha256, size, chunks}: answer to "enter", "chunks" messages "biome_data" follow if the client
# does not have that archive yet, then "welcome" and "spawn" of the objects of that biome
MSG_BIOME = 'biome'
# {biome_id, index, data}: one piece of the biome archive, data is base64 encoded
MSG_BIOME_DATA = 'biome_data'
MSG_WELCOME = 'welcome'
MSG_SPAWN = 'spawn'
MSG_STATE = 'state'
MSG_DESPAWN = 'despawn'

# size of one piece of the biome archive before base64 encoding, bytes
BIOME_CHUNK_SIZE = 256 * 1024


def message(msg_type, payload=None):
    return {'t': msg_type, 'p': payload or {}}


def safe_name(name):
    # names received from the network are used as folder names, only letters, digits, "-" and "_" stay
    name = re.sub(r'[^A-Za-z0-9_-]', '_', str(name))
    return name or '_'


def location_id(server_name, biome_id):
    # files of the biome are kept on the client in biomes/<location_id>/
    return f'{safe_name(server_name)}_{safe_name(biome_id)}'

#------------------------------------------------------------------------------


class Model(object):

    # all fields which are transferred over the web socket
    data_fields = ()
    # fields which can change while the game is running, only those are sent with every state update
    state_fields = ()

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join(f"{f}={getattr(self, f)!r}" for f in self.data_fields)})'

    def to_data(self):
        d = {}
        for f in self.data_fields:
            v = getattr(self, f)
            d[f] = v.to_data() if isinstance(v, Model) else v
        return d

    def to_state(self):
        return {f: getattr(self, f) for f in ('id', ) + self.state_fields}

    def update(self, d):
        for f in self.state_fields:
            if f in d:
                setattr(self, f, d[f])

    @classmethod
    def from_data(cls, d):
        return cls(**{f: d[f] for f in cls.data_fields if f in d})


class World(Model):
    data_fields = ('name', 'width', 'height', 'tick_rate', 'protocol')

    def __init__(self, name: str, width: int, height: int, tick_rate: int = TICK_RATE, protocol: int = PROTOCOL_VERSION):
        self.name = name
        self.width = width
        self.height = height
        self.tick_rate = tick_rate
        self.protocol = protocol


class Biome(Model):
    # one world of the server: its land, buildings and plants are in the archive the client downloads before entering
    data_fields = ('id', 'name', 'sha256', 'size')

    def __init__(self, id: str, name: Optional[str] = None, sha256: Optional[str] = None, size: int = 0):
        self.id = id
        self.name = name or id
        self.sha256 = sha256
        self.size = size


class Map(Model):
    data_fields = ('width', 'height', 'water_level')

    def __init__(self, width: int, height: int, water_level: float):
        self.width = width
        self.height = height
        self.water_level = water_level


class Team(Model):
    data_fields = ('name', 'members')

    def __init__(self, name: str, members: Optional[List[int]] = None):
        self.name = name
        self.members = members if members is not None else []


class Template(Model):
    # info about the 3d model and what is required to build and render the 3d model on the client device
    # figure name, coefs, textures, skin, hair, wears, weapon, etc.
    # store_id: which store the model and its textures are downloaded from, see DEFAULT_STORE_ID
    data_fields = ('model_name', 'skin', 'hair', 'wears', 'weapon', 'texture', 'coefs', 'elevation_correction', 'store_id')

    def __init__(
        self,
        model_name: str,
        skin: int = 0,
        hair: Optional[int] = None,
        wears: Optional[List[str]] = None,
        weapon: Optional[str] = None,
        texture: Optional[str] = None,
        coefs: Optional[List[float]] = None,
        elevation_correction: Optional[float] = None,
        store_id: int = DEFAULT_STORE_ID,
    ):
        self.model_name = model_name
        self.skin = skin
        self.hair = hair
        self.wears = wears if wears is not None else []
        self.weapon = weapon
        self.texture = texture
        self.coefs = coefs if coefs is not None else [0.5, 0.5, 0.75]
        self.elevation_correction = elevation_correction
        self.store_id = store_id


class Object(Model):
    # all related to position a static object on the map
    # name, template, coordinates, orientation
    kind = 'object'
    data_fields = ('id', 'kind', 'name', 'template', 'w', 'h', 'shift_w', 'shift_h', 'direction')
    state_fields = ('w', 'h', 'shift_w', 'shift_h', 'direction')

    def __init__(
        self,
        id: int,
        name: str,
        template: Template,
        w: int,
        h: int,
        kind: Optional[str] = None,
        shift_w: float = 0.5,
        shift_h: float = 0.5,
        direction: float = 0.0,
    ):
        self.id = id
        self.name = name
        # received from the network as a dict
        self.template = Template.from_data(template) if isinstance(template, dict) else template
        self.w = w
        self.h = h
        self.kind = kind or type(self).kind
        self.shift_w = shift_w
        self.shift_h = shift_h
        self.direction = direction


class Unit(Object):
    # speed, acceleration, mass, turn_speed
    # move, turn, target_direction: controls the unit is moved with now, so the clients can calculate its movement
    # between the ticks with the same rules as the server, see lib.physics
    # input_seq, input_tick: the last Controls applied by the server and the tick it was applied at,
    # input_slack: how many ticks before that tick the Controls arrived, negative if it was late
    kind = 'unit'
    data_fields = Object.data_fields + (
        'speed', 'max_speed', 'acceleration_value', 'turn_speed', 'deceleration_value', 'action', 'animation',
        'move', 'turn', 'target_direction', 'input_seq', 'input_tick', 'input_slack',
    )
    state_fields = Object.state_fields + (
        'speed', 'action', 'animation', 'move', 'turn', 'target_direction', 'input_seq', 'input_tick', 'input_slack',
    )

    def __init__(
        self,
        id: int,
        name: str,
        template: Template,
        w: int,
        h: int,
        kind: Optional[str] = None,
        shift_w: float = 0.5,
        shift_h: float = 0.5,
        direction: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 0.06,
        acceleration_value: float = 0.004,
        turn_speed: float = 5.0,
        deceleration_value: float = 0.02,
        action: str = 'idle',
        animation: Optional[str] = None,
        move: int = 0,
        turn: int = 0,
        target_direction: Optional[float] = None,
        input_seq: int = 0,
        input_tick: int = 0,
        input_slack: int = 0,
    ):
        super().__init__(id=id, name=name, template=template, w=w, h=h, kind=kind, shift_w=shift_w, shift_h=shift_h, direction=direction)
        self.speed = speed
        self.max_speed = max_speed
        self.acceleration_value = acceleration_value
        self.turn_speed = turn_speed
        self.deceleration_value = deceleration_value
        self.action = action
        self.animation = animation
        self.move = move
        self.turn = turn
        self.target_direction = target_direction
        self.input_seq = input_seq
        self.input_tick = input_tick
        self.input_slack = input_slack


class Hero(Unit):
    # unit controlled by one of the connected players
    kind = 'hero'
    data_fields = Unit.data_fields + ('owner', )

    def __init__(
        self,
        id: int,
        name: str,
        template: Template,
        w: int,
        h: int,
        kind: Optional[str] = None,
        shift_w: float = 0.5,
        shift_h: float = 0.5,
        direction: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 0.06,
        acceleration_value: float = 0.004,
        turn_speed: float = 5.0,
        deceleration_value: float = 0.02,
        action: str = 'idle',
        animation: Optional[str] = None,
        move: int = 0,
        turn: int = 0,
        target_direction: Optional[float] = None,
        input_seq: int = 0,
        input_tick: int = 0,
        input_slack: int = 0,
        owner: Optional[str] = None,
    ):
        super().__init__(
            id=id, name=name, template=template, w=w, h=h, kind=kind, shift_w=shift_w, shift_h=shift_h, direction=direction,
            speed=speed, max_speed=max_speed, acceleration_value=acceleration_value, turn_speed=turn_speed,
            deceleration_value=deceleration_value, action=action, animation=animation, move=move, turn=turn,
            target_direction=target_direction, input_seq=input_seq, input_tick=input_tick, input_slack=input_slack,
        )
        self.owner = owner


class Monster(Unit):
    kind = 'monster'


class Building(Object):
    kind = 'building'


class Controls(Model):
    # what the player currently wants his hero to do, client sends it every time the user input changes
    # move: 1 forward, -1 backward, 0 stop
    # turn: 1 left, -1 right, 0 no turn
    # target_direction: direction (in degrees) the hero must turn to, set by the joystick, None if not used
    # tick: server tick the controls must be applied at, so the client knows in advance how the hero will move
    # seq: number of the controls sent by the client, server reports the last applied one in Unit.input_seq
    data_fields = ('move', 'turn', 'target_direction', 'tick', 'seq')

    def __init__(self, move: int = 0, turn: int = 0, target_direction: Optional[float] = None, tick: int = 0, seq: int = 0):
        self.move = move
        self.turn = turn
        self.target_direction = target_direction
        self.tick = tick
        self.seq = seq


KINDS = {c.kind: c for c in (Object, Unit, Hero, Monster, Building)}


def object_from_dict(d):
    return KINDS.get(d.get('kind'), Object).from_data(d)
