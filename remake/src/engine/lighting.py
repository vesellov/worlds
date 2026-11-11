"""
The light of a map by the time of the day, as game.exe does it.

config/Lights[Cave]<allod>.ini (563930) has the sections [sunlight], [ambient] and [sky] with keys "timeNN" = r, g, b.
The hour is the two characters after "time", the colors are divided by 255, the hours which are not given
are filled linearly between the given ones, the hours after the last given one keep it.

Every frame (566360) for the hour t of the world (WORLD_SET WS_TIME of the .mob at the start):
- the three colors are interpolated between the hour floor(t) and (floor(t) + 1) % 24;
- the ambient is 0.8 * sunlight when the ambient of the hour is black;
- the sky color clears the screen and colors the fog;
- the sun stands at the elevation 30 + 30 * (1 - |t - 12| / 8) degrees in the day (4..20 h), at night
  it is 30 + 45 * (distance to 4 h or 20 h) / 4, the azimuth is t / 12 * pi;
  the light goes along (cos az * cos el, sin az * cos el, -sin el);
- the fog starts at FogDayStartDistance (90) from 6 to 18 h, at FogNightStartDistance (50) from 22 to 2 h
  and changes linearly in the 4 hours between them.
In a cave (the zone has "#sky cave") the colors are taken from the hour 0, the light goes along (0.5, 0.5, -0.7071)
and the fog starts at the day distance.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

import numpy as np

#------------------------------------------------------------------------------

HOURS = 24

SECTIONS = ('sunlight', 'ambient', 'sky')

# game.exe 0x73cca8
AMBIENT_FROM_SUN = 0.8

# the defaults of the settings, game.exe 45xxxx (FogNightStartDistance, FogDayStartDistance, FarClipDistance)
FOG_NIGHT_START = 50.0
FOG_DAY_START = 90.0
FAR_CLIP = 100.0

CAVE_LIGHT_DIRECTION = (0.5, 0.5, -0.70710677)

DEFAULT_LIGHTS = {
    'sunlight': {0: (1.0, 1.0, 1.0)},
    'ambient': {0: (0.5, 0.5, 0.5)},
    'sky': {0: (0.2, 0.6, 0.8)},
}

#------------------------------------------------------------------------------


class LightState(object):

    def __init__(self, sun_color, ambient_color, sky_color, sun_direction, fog_start, fog_end):
        self.sun_color = sun_color
        self.ambient_color = ambient_color
        self.sky_color = sky_color
        # the direction the light goes, from the sun to the ground
        self.sun_direction = sun_direction
        self.fog_start = fog_start
        self.fog_end = fog_end

    def __repr__(self):
        return 'LightState(sun=%s ambient=%s sky=%s dir=%s fog=%.0f)' % (
            np.round(self.sun_color, 2), np.round(self.ambient_color, 2), np.round(self.sky_color, 2),
            np.round(self.sun_direction, 2), self.fog_start)


class WorldLight(object):

    def __init__(self, tables, cave=False):
        """
        tables: section name -> (24, 3) colors of the hours.
        """
        self.tables = tables
        self.cave = cave

    def at(self, hour):
        if self.cave:
            hour = 0.0
        hour = float(hour) % HOURS
        index = int(math.floor(hour))
        fraction = hour - index
        following = (index + 1) % HOURS
        colors = {}
        for section in SECTIONS:
            table = self.tables[section]
            colors[section] = table[index] + (table[following] - table[index]) * fraction
        ambient = colors['ambient']
        if not np.any(self.tables['ambient'][index]):
            ambient = colors['sunlight'] * AMBIENT_FROM_SUN
        if self.cave:
            return LightState(colors['sunlight'], ambient, colors['sky'], np.array(CAVE_LIGHT_DIRECTION), FOG_DAY_START, FAR_CLIP)
        return LightState(colors['sunlight'], ambient, colors['sky'], sun_direction(hour), fog_start(hour), FAR_CLIP)

#------------------------------------------------------------------------------


def parse_lights(text):
    """
    Section name -> (24, 3) float colors from the text of a Lights*.ini.
    """
    given = {section: {} for section in SECTIONS}
    section = None
    for raw_line in (text or '').splitlines():
        line = raw_line.split(';', 1)[0].strip()
        if not line:
            continue
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1].strip().lower()
            continue
        if section not in given or '=' not in line:
            continue
        key, value = (part.strip() for part in line.split('=', 1))
        if not key.lower().startswith('time'):
            continue
        try:
            hour = int(key[4:6])
            rgb = tuple(int(v) / 255.0 for v in value.split(','))[0:3]
        except ValueError:
            continue
        if len(rgb) == 3:
            given[section][hour] = rgb
    tables = {}
    for section in SECTIONS:
        tables[section] = fill_hours(given[section] or DEFAULT_LIGHTS[section])
    return tables


def fill_hours(given):
    """
    (24, 3) colors: linear between the given hours like game.exe 563930, the first given hour from 0.
    """
    table = np.zeros((HOURS, 3))
    previous = 0
    for hour in sorted(given):
        if not 0 <= hour < HOURS:
            continue
        table[hour] = given[hour]
        steps = hour - previous
        for k in range(previous + 1, hour):
            table[k] = table[previous] + (table[hour] - table[previous]) * (k - previous) / float(steps)
        previous = hour
    for k in range(previous + 1, HOURS):
        table[k] = table[previous]
    return table


def sun_elevation_degrees(hour):
    if hour < 4.0:
        part = (4.0 - hour) / 4.0
        return part * 45.0 + 30.0
    if hour <= 20.0:
        distance = abs(hour - (20.0 + 4.0) * 0.5) / (20.0 - 4.0)
        return ((1.0 - (distance + distance)) + 1.0) * 30.0
    part = (hour - 20.0) / (24.0 - 20.0)
    return part * 45.0 + 30.0


def sun_direction(hour):
    elevation = math.radians(sun_elevation_degrees(hour))
    azimuth = hour / 12.0 * math.pi
    return np.array([
        math.cos(azimuth) * math.cos(elevation),
        math.sin(azimuth) * math.cos(elevation),
        -math.sin(elevation),
    ])


def fog_start(hour, day=FOG_DAY_START, night=FOG_NIGHT_START):
    if 6.0 <= hour <= 18.0:
        return day
    if hour >= 22.0 or hour <= 2.0:
        return night
    if hour >= 6.0:
        return day - (day - night) * ((hour - 18.0) / 4.0)
    return (day - night) * ((hour - 2.0) / 4.0) + night


def for_map(game_data, map_name):
    """
    WorldLight of a map: its zone in map.txt chooses the island and the cave lights.
    """
    zone = game_data.zone_of_map(map_name)
    cave = zone is not None and zone.is_cave
    file_name = zone.lights_file_name() if zone is not None else 'lightsgipat.ini'
    text = game_data.config_text(file_name)
    if text is None:
        if _Debug:
            print('no %s, default lights' % file_name)
    return WorldLight(parse_lights(text), cave=cave)
