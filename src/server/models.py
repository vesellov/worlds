

class World:
    #name, size, map, rules
    pass


class Map:
    pass


class Team:
    pass


class Template:
    # info about the 3d model and what is required to build and render the 3d model on the client device
    # figure name, coefs, textures, skin, hair, wears, weapon, etc.
    pass


class Object:
    # all related to position a static object on the map
    # name, template, coordinates, orientation
    pass


class Unit(Object):
    # speed, acceleration, mass, turn_speed 
    pass


class Hero(Unit):
    pass


class Monster(Unit):
    pass


class Building(Object):
    pass

