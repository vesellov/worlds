"""
The weather of the client: the rain, the snow and the tornado of a zone (engine.weather), the loop of
their sound and, in the main menu, the storm of the night with the bolts of engine.lightning.

engine.weather.Precipitation gives the triangles of the drops of the rain and the snow around the eye of the
camera (CWeather, 4e4240/4e4c80), the tornado is particles of the type RainSnow; the view gives them the wind of
the map and client.scene.set_particles draws them with the particles of the spells. The sound is a loop of
sfx.res: "nature/rain/circle.wav" for the rain, "nature/Tornado/circle.wav" for the tornado; the
thunder of the storm of the menu is "nature/thunder/<n>.wav".

On a map the rain and the snow come and go as engine.weather.Schedule says (the server of game.exe, 58ccd0):
the zone opens dry, the density of the drops and the volume of the loop turn in and out (4e40b0), and while it
rains a lightning falls at a random point of the map on every check of the server.

The storm of the main menu follows CMainMenuWindow__Update (632ad0): every 100 ms a random number is
taken and a lightning falls when it is 4 modulo 41. game.exe turns the weather on at 0 h and off at 7 h;
the remake opens the menu at the local hour of game.exe (engine.menus.main_menu_hour) and keeps the storm on.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import random

#------------------------------------------------------------------------------

from engine import lightning
from engine import particles
from engine import weather as weather_module
from engine import world

#------------------------------------------------------------------------------

# the loops of the weather (sfx.res)
RAIN_SOUND = 'nature/rain/circle.wav'
TORNADO_SOUND = 'nature/Tornado/circle.wav'
THUNDER_SOUNDS = ('nature/thunder/1.wav', 'nature/thunder/2.wav', 'nature/thunder/3.wav',
                  'nature/thunder/4.wav')
LOOP_VOLUME = 0.55

# the storm (632ad0): every 100 ms, a number modulo 41 equal to 4 gives a lightning
STORM_PERIOD_MS = 100.0
STORM_CHANCE = 41
STORM_ROLL = 4
STORM_RADIUS = 34.0
STORM_HEIGHT = 30.0
STORM_TICKS = 12
# the tick of the world is 55 ms, so the check goes once in two ticks
STORM_TICKS_PERIOD = 2
# the thunder of a lightning of the rain is heard only near the camera
THUNDER_RADIUS = 80.0

# the campfire of the main menu (the torch of the map)
CAMPFIRE_SCALE = 0.7

#------------------------------------------------------------------------------


class WeatherView(object):
    """
    The weather over the scene of the map or of the main menu.
    """

    def __init__(self, app):
        self.app = app
        self.weather = None
        self.kind = weather_module.NONE
        # engine.weather.Schedule of the rain or the snow of the zone, None in the menu
        self.schedule = None
        self.level = weather_module.LEVELS
        self.storm = False
        self.random = random.Random()
        self.ticks = 0
        self.storm_ticks = 0
        self.storm_bolts = []
        self.sound = None
        self.sound_name = None
        self.campfire = None

    def __repr__(self):
        return 'WeatherView(%s, %d bolts)' % (self.kind, len(self.storm_bolts))

    @property
    def is_open(self):
        return self.weather is not None

    def open_map(self, map_name):
        """
        The weather of the zone of the map (#weather of map.txt, none in a cave).
        """
        zone = self.app.game_data.zone_of_map(map_name) if map_name else None
        kind = weather_module.from_zone(zone)
        schedule = weather_module.Schedule(kind, rng=self.random)
        self.start(kind, level=schedule.level() if schedule.cycles else weather_module.LEVELS)
        if schedule.cycles and self.weather is not None:
            self.schedule = schedule

    def open_menu(self):
        """
        The night of the main menu: the rain with the campfire of the torch of the map and the storm.
        """
        self.start(weather_module.RAIN)
        self.storm = True
        self.add_campfire()

    def start(self, kind, level=weather_module.LEVELS):
        self.close()
        app = self.app
        if app.terrain is None or app.scene is None:
            return
        target = tuple(float(v) for v in app.scene.camera.target)
        self.weather = weather_module.Weather(kind, target, rng=self.random,
                                              ground=app.terrain.height_at, level=level)
        self.kind = self.weather.kind
        self.level = level
        self.ticks = 0
        self.storm_ticks = 0
        self.storm_bolts = []
        self.set_wind()
        self.start_sound()

    def set_wind(self):
        """
        The wind of the map (WS_WIND_DIR and WS_WIND_STR) turns the rain and the snow.
        """
        app = self.app
        mob_name = app.map_name
        if mob_name is None:
            return
        root = app.game_data.map_mob(mob_name)
        if root is None:
            return
        settings = world.world_settings(root)
        strength = max(settings.wind_strength, 0.0)
        if self.weather.precipitation is not None:
            direction = tuple(settings.wind_direction) + (0.0, 0.0, 0.0)
            self.weather.precipitation.wind = (direction[0] * strength, direction[1] * strength,
                                               direction[2] * strength)
        for emitter in self.weather.emitters:
            emitter.wind = (settings.wind_direction[0], settings.wind_direction[1],
                            max(settings.wind_strength, 0.0), 0.0)

    def add_campfire(self):
        """
        The campfire of the menu: the fire smoke at the first torch of the map.
        """
        app = self.app
        mob_name = app.map_name
        root = app.game_data.map_mob(mob_name) if mob_name else None
        if root is None:
            return
        for obj in world.map_objects(root):
            if obj.kind != 'TORCH':
                continue
            x, y, z = obj.position
            ground = app.terrain.height_at(x, y)
            self.campfire = particles.Emitter(particles.FIRE_SMOKE, (x, y, ground + z + 0.5),
                                              scale=CAMPFIRE_SCALE, rng=self.random,
                                              ground=app.terrain.height_at)
            return

    def start_sound(self):
        self.stop_sound()
        app = self.app
        audio = getattr(app, 'audio', None)
        if audio is None or not audio.enabled:
            return
        if self.level <= 0:
            return
        if self.kind == weather_module.RAIN:
            name = RAIN_SOUND
        elif self.kind == weather_module.TORNADO:
            name = TORNADO_SOUND
        else:
            return
        sound = audio.load(name)
        if sound is None:
            return
        sound.loop = True
        sound.volume = self.sound_volume()
        sound.play()
        self.sound = sound
        self.sound_name = name

    def sound_volume(self):
        """
        4e40b0: the volume of the loop follows the density of the drops.
        """
        audio = self.app.audio
        return audio.volume_sfx * LOOP_VOLUME * self.level / float(weather_module.LEVELS)

    def stop_sound(self):
        if self.sound is not None:
            self.sound.stop()
            self.sound.loop = False
        self.sound = None
        self.sound_name = None

    def tick(self):
        """
        One tick of the world: the rain of the zone comes and goes, the particles follow the camera,
        the campfire burns, the storm throws bolts.
        """
        if self.weather is None:
            return
        app = self.app
        target = tuple(float(v) for v in app.scene.camera.target)
        if self.schedule is not None:
            self.tick_schedule(target)
        self.weather.update(target)
        if self.campfire is not None:
            self.campfire.update()
        self.ticks += 1
        self.tick_bolts()
        if self.storm:
            self.tick_storm(target)

    def tick_schedule(self, target):
        strikes, changed = self.schedule.tick()
        level = self.schedule.level()
        if level != self.level:
            self.level = level
            self.weather.set_level(level)
            if level <= 0:
                self.stop_sound()
            elif self.sound is None:
                self.start_sound()
            else:
                self.sound.volume = self.sound_volume()
        for start, end in strikes:
            self.strike(start, end, target)

    def strike(self, start, end, target):
        """
        58ccd0: the lightning of the rain at a point of the map, from above the land down under it.
        """
        app = self.app
        sx, sy, sz = start
        ex, ey, ez = end
        bolt = lightning.Lightning((sx, sy, app.terrain.height_at(sx, sy) + sz),
                                   (ex, ey, app.terrain.height_at(ex, ey) + ez), 0.0,
                                   rng=self.random, time=self.ticks)
        self.storm_bolts.append((bolt, STORM_TICKS))
        if math.hypot(ex - target[0], ey - target[1]) > THUNDER_RADIUS:
            return
        audio = getattr(app, 'audio', None)
        if audio is not None and audio.enabled:
            audio.play_interface(self.random.choice(THUNDER_SOUNDS))

    def tick_bolts(self):
        alive = []
        for bolt, left in self.storm_bolts:
            bolt.update(1.0)
            left -= 1
            if left > 0:
                alive.append((bolt, left))
        self.storm_bolts = alive

    def tick_storm(self, target):
        """
        632ad0: every 100 ms a random number modulo 41 equal to 4 strikes a lightning near the camera.
        """
        self.storm_ticks += 1
        if self.storm_ticks < STORM_TICKS_PERIOD:
            return
        self.storm_ticks = 0
        if self.random.randrange(STORM_CHANCE) != STORM_ROLL:
            return
        app = self.app
        angle = self.random.random() * 2.0 * math.pi
        radius = STORM_RADIUS * (0.3 + 0.7 * self.random.random())
        x = target[0] + math.cos(angle) * radius
        y = target[1] + math.sin(angle) * radius
        ground = app.terrain.height_at(x, y)
        bolt = lightning.Lightning((x, y, ground + STORM_HEIGHT), (x, y, ground), 0.0,
                                   rng=self.random, time=self.ticks)
        self.storm_bolts.append((bolt, STORM_TICKS))
        audio = getattr(app, 'audio', None)
        if audio is not None and audio.enabled:
            audio.play_interface(self.random.choice(THUNDER_SOUNDS))

    def particles(self):
        """
        The emitters for client.scene.set_particles.
        """
        emitters = []
        if self.weather is not None:
            emitters.extend(self.weather.emitters)
        if self.campfire is not None:
            emitters.append(self.campfire)
        return emitters

    def precipitation(self, fraction):
        """
        The drops of the rain or the snow for client.scene.set_particles and their time in ticks.
        """
        if self.weather is None or self.weather.precipitation is None or self.level <= 0:
            return None
        return self.weather.precipitation, self.ticks + fraction

    def bolts(self):
        return [bolt for bolt, left in self.storm_bolts]

    def close(self):
        self.stop_sound()
        self.weather = None
        self.kind = weather_module.NONE
        self.schedule = None
        self.level = weather_module.LEVELS
        self.storm = False
        self.storm_bolts = []
        self.campfire = None
