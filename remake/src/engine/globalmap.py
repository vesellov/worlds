"""
The map of an allod where the party travels between the zones (the global map of game.exe 64cbd0 and the
sounds of buttons/Globalmap of sfx.res).

A zone of map.txt has a "#position" of three numbers: the place of its marker over the texture "<allod>mapg".
A zone of the type "edge" has no .mpr: it is a road, and its exits are the zones where the party can go.
The remake opens this map when the party walks into an exit of a road (client.zoneexit): the markers of
the zones of the allod and of their roads are drawn over the texture of the allod, the roads out of the
current zone are the places to click. "#passtime" of an exit is the time of the way in hours.

The positions of the zones of the type "game" are often (0, 0, 0) in map.txt; the remake takes the middle
of the positions of the roads which lead to such a zone. The tests cover the positions, the places of the
markers and the roads out of a zone.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import questmap

#------------------------------------------------------------------------------

# the share of the map texture which the markers keep free at its sides
MARGIN = 0.10

# the types of the zones of map.txt: a map of the party, a base of the global map and a road
GAME_TYPE = 'game'
BRIEF_TYPE = 'brief'
EDGE_TYPE = 'edge'

#------------------------------------------------------------------------------


class TravelOption(object):
    """
    A road out of a zone of the global map: the exit of the zone, its zone and the hours of the way.
    """

    def __init__(self, zone, zone_exit, point):
        self.zone = zone
        self.exit = zone_exit
        self.point = point

    def __repr__(self):
        return 'TravelOption(%s:%d)' % (self.zone.name if self.zone is not None else None,
                                        self.exit.number if self.exit is not None else 0)

    @property
    def has_map(self):
        return self.zone is not None and bool(self.zone.mpr_name)

    @property
    def pass_time(self):
        return self.exit.pass_time if self.exit is not None else 0.0

#------------------------------------------------------------------------------


class GlobalMap(object):
    """
    The zones of an allod, their places on its texture and the roads out of one zone.
    """

    def __init__(self, game_data, allod, edge_zone):
        self.game_data = game_data
        self.allod = allod
        self.edge_zone = edge_zone
        self.zones = [zone for zone in game_data.zones() if zone.allod == allod]
        self.by_name = {zone.name.lower(): zone for zone in self.zones}
        self.positions = self.zone_positions()
        self.bounds = bounds(self.positions.values())
        self.options = self.travel_options(edge_zone)

    def __repr__(self):
        return 'GlobalMap(%s, %d zones, %d roads)' % (self.allod, len(self.zones), len(self.options))

    @classmethod
    def from_game(cls, game_data, edge_zone):
        allod = edge_zone.allod if edge_zone is not None else None
        return cls(game_data, allod, edge_zone)

    def zone_of(self, name):
        return self.by_name.get((name or '').lower())

    def zone_positions(self):
        """
        zone name -> (x, y) of map.txt; the zones at (0, 0) take the middle of the zones of their roads.
        """
        text = self.game_data.text('map.txt') or ''
        quests, points = questmap.parse_map_txt(text)
        positions = {}
        for point in points:
            if point.allod != self.allod or len(point.position or ()) < 2:
                continue
            positions[point.name.lower()] = (float(point.position[0]), float(point.position[1]))
        neighbours = {}
        for zone in self.zones:
            for zone_exit in zone.exits:
                other = (zone_exit.neighbour_name or '').lower()
                if not other:
                    continue
                neighbours.setdefault(zone.name.lower(), set()).add(other)
                neighbours.setdefault(other, set()).add(zone.name.lower())
        known = [(name, x, y) for name, (x, y) in positions.items() if (x, y) != (0.0, 0.0)]
        for zone in self.zones:
            name = zone.name.lower()
            x, y = positions.get(name, (0.0, 0.0))
            if (x, y) == (0.0, 0.0):
                found = [(px, py) for other, px, py in known if other in neighbours.get(name, ())]
                if found:
                    x = sum(point[0] for point in found) / len(found)
                    y = sum(point[1] for point in found) / len(found)
            positions[name] = (x, y)
        return positions

    def travel_options(self, zone):
        """
        The exits of a zone of the roads whose neighbour zone is known.
        """
        if zone is None:
            return []
        result = []
        for zone_exit in zone.exits:
            other = self.zone_of(zone_exit.neighbour_name)
            if other is None:
                continue
            result.append(TravelOption(other, zone_exit, self.positions.get(other.name.lower(), (0.0, 0.0))))
        return result

    def point(self, zone):
        return self.positions.get((zone.name if zone is not None else '').lower())

    def marker(self, zone):
        """
        The place of a marker in the map texture: u and v in 0..1 (0, 0 at its left top).
        """
        point = self.point(zone)
        if point is None:
            return None
        return project(point[0], point[1], self.bounds)

#------------------------------------------------------------------------------


def bounds(points):
    """
    (min x, min y, max x, max y) of the points; at least one unit wide and high.
    """
    points = list(points)
    if not points:
        return (-1.0, -1.0, 1.0, 1.0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 - x0 < 1.0:
        x0, x1 = x0 - 0.5, x1 + 0.5
    if y1 - y0 < 1.0:
        y0, y1 = y0 - 0.5, y1 + 0.5
    return (x0, y0, x1, y1)


def project(x, y, box):
    """
    The u, v of a point in 0..1 inside the bounds, with the margin at the sides; y goes up.
    """
    x0, y0, x1, y1 = box
    u = MARGIN + (x - x0) / (x1 - x0) * (1.0 - 2.0 * MARGIN)
    v = MARGIN + (y1 - y) / (y1 - y0) * (1.0 - 2.0 * MARGIN)
    return (u, v)
