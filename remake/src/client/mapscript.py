"""
The scripts of a map in the client: engine.script.ScriptWorld over the objects of the map with a host
which knows the units, the props and the other objects of the .mob by their NID and name.

The string events of the scripts are taken by the interface of game.exe (5f8070) by the first word:
"say" and "say_block" make a unit speak, "tutorial" opens the tutorial window, "briefing" a briefing,
"autosave" saves the game, "endofgame" ends it; the rest ("#output 150" of the editor) is skipped.
A tutorial goes to the tutorial window (client.tutorial); "say <phrase> <unit>" plays the phrase of the scenario
of the unit (acks.db).
The things which the scripts give to the player 0 go to the party of the app (engine.party.Party).
HideObject sets hidden of a unit or a prop (the view shrinks it into its point), RemoveUnitFromServer
also takes the unit from the map and from engine.ai. The other objects of the map (MapFigure: the static ones
and the levers) can be hidden, moved by SetCP and removed; then the grid of the paths is built again
and SetCP pushes the units out of the new place.

PlayerSee (0xa5) gives the units of the player and the units they see; the remake takes the hero and
the units seen by the sight of engine.ai (5cb850 by the sight alone).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from eilib import idb
from eilib import script as script_lang
from engine import ai
from engine import magic
from engine import pathing
from engine import script
from engine import trap as trap_module

#------------------------------------------------------------------------------

TUTORIAL_EVENT = 'tutorial'

BRIEFING_EVENT = 'briefing'

AUTOSAVE_EVENT = 'autosave'

SAY_EVENTS = ('say', 'say_block')

# the name of the hero in the say events ("say_block R1 Hero")
HERO_NAME = 'Hero'

#------------------------------------------------------------------------------


class MapFigure(object):
    """
    A figure of the map without a unit (engine.world.MapObject): its place, its view (client.scene.PlacedView)
    and its parts for the grid of the paths. position is x, y and the height over the land, like OBJPOSITION.
    """

    def __init__(self, map_object, figure_model, world_matrix):
        self.map_object = map_object
        self.figure_model = figure_model
        self.world_matrix = world_matrix
        self.position = map_object.position
        self.name = map_object.name
        self.view = None
        # RemoveObjectFromServer: not drawn and not on the grid
        self.removed = False
        self._hidden = False
        self._parts = None

    def __repr__(self):
        return 'MapFigure(%s %s)' % (self.name, self.map_object.template)

    @property
    def hidden(self):
        return self._hidden

    @hidden.setter
    def hidden(self, value):
        self._hidden = bool(value)
        self.show()

    def show(self):
        if self.view is not None:
            self.view.place(self.world_matrix, self._hidden or self.removed)

    def move(self, x, y, z, land_z):
        """
        5c11d0: the figure stands at x, y with its bottom z over the land (land_z under it), its turn stays.
        """
        self.position = (x, y, z)
        self.world_matrix = self.world_matrix.copy()
        self.world_matrix[0:3, 3] = (x, y, land_z + z)
        self._parts = None
        self.show()

    def remove(self):
        self.removed = True
        self.show()

    def parts(self):
        """
        engine.pathing.object_parts of the figure, none when it is removed.
        """
        if self.removed:
            return []
        if self._parts is None:
            self._parts = pathing.object_parts(self.figure_model, self.world_matrix)
        return self._parts


class LeverObject(MapFigure):
    """
    A lever of the map in the client: the engine.lever.Lever, its figure and its place.
    """

    def __init__(self, map_object, lever, figure_model, world_matrix):
        MapFigure.__init__(self, map_object, figure_model, world_matrix)
        self.lever = lever

    def __repr__(self):
        return 'LeverObject(%s %s)' % (self.name, self.lever)

    def parts(self):
        """
        engine.pathing.object_parts of the figure in the state of the logic.
        """
        if self.removed:
            return []
        self.figure_model.coefs = self.lever.logic_coefs()
        return pathing.object_parts(self.figure_model, self.world_matrix)

#------------------------------------------------------------------------------


class ClientHost(script.Host):

    def __init__(self, app):
        self.app = app
        # the names of the .mob files which AddMob loaded (+0x128 of the world)
        self.added_mobs = set()
        # NID of the .mob -> a unit, a prop or an engine.world.MapObject
        self.objects_by_nid = {}
        # OBJNAME -> the same
        self.objects_by_name = {}
        # the object -> its NID
        self.nids = {}

    def add_object(self, nid, name, obj):
        if nid is None:
            return
        self.objects_by_nid[nid] = obj
        self.nids[id(obj)] = nid
        if name and name not in self.objects_by_name:
            self.objects_by_name[name] = obj

    def object_by_id(self, nid):
        return self.objects_by_nid.get(nid)

    def object_by_name(self, name):
        return self.objects_by_name.get(name)

    def object_id(self, obj):
        return self.nids.get(id(obj), 0)

    def position(self, obj):
        if hasattr(obj, 'x'):
            return (obj.x, obj.y, getattr(obj, 'z', 0.0))
        position = getattr(obj, 'position', None)
        if position is not None:
            return tuple(position)
        return None

    def unit_of_player(self, player, index):
        """
        4b56c0 case 0xb4: the units of the player (+0xa0 of the player, the party with the hero first for the
        player 0, the units of a map in their order for the others) by the number.
        """
        if player == ai.PARTY_PLAYER:
            units = ([self.app.unit] if self.app.unit is not None else []) + list(self.app.companions)
        else:
            units = [u for u in self.app.map_units if self.app.ai.player_of(u) == player]
        if 0 <= index < len(units):
            return units[index]
        return None

    def leader(self):
        """
        4b56c0 case 0xe4: the unit which leads the party (41a020), the hero of the remake.
        """
        return self.app.unit

    def seen_by_player(self, player):
        """
        4b5b00 (0xa5): every unit of the player and the units it sees (the lists +0xa94 of its units,
        engine.ai.Controller.party_seen).
        """
        hero = self.app.unit
        if player != 0 or hero is None or hero.is_dead:
            return ()
        controller = self.app.ai
        result = [hero]
        for u, seen in controller.party_seen.items():
            for other in [u] + list(seen):
                if other not in result and not other.is_dead:
                    result.append(other)
        return tuple(result)

    def add_mob(self, name, variables):
        """
        595ab0: a .mob is loaded once; its program, None when it has no scripts or was loaded already.
        """
        key = name.lower()
        if key in self.added_mobs:
            return None
        self.added_mobs.add(key)
        root = self.app.add_mob(name)
        try:
            return program_of(root, variables)
        except script_lang.ScriptError as error:
            self.app.show_message('the scripts of %s: %s' % (name, error))
            return None

    def leave_to_zone(self, player, zone_name, exit_index):
        if player == 0:
            self.app.leave_to_zone(zone_name, exit_index)

    def play_music(self, name):
        self.app.music.system.play_script(name)
        return True

    def cast_spell(self, text, origin, target, point):
        app = self.app
        spell = trap_module.parse_spell(app.game_data.spells(), text)
        if spell is None or app.terrain is None:
            return False
        if target is not None and getattr(target, 'fighter', None) is None:
            return True
        ground = app.terrain.height_at
        caster = trap_module.Caster((origin[0], origin[1], ground(*origin)), land_height=ground)
        caster.on_shoot = app.add_arrow
        caster.on_spell_effect = app.spell_effects.append
        if target is None:
            target = magic.Spot(point[0], point[1], ground(*point))
        caster.spread(spell, target, app.living_units())
        return True

    def play_fx(self, position, near, far, sound_name):
        self.app.audio.play_sound(magic.SpellSound(sound_name, position, near, far))

    def world_time(self):
        return self.app.hour

    def set_world_time(self, hour, runs):
        """
        The remake keeps the hour still in both cases: the rate of the time of game.exe (79be4c) is not known.
        """
        self.app.set_hour(hour)

    def is_enemy(self, obj, player):
        if not hasattr(obj, 'brain'):
            return False
        return self.app.ai.diplomacy.at_war(player, self.app.ai.player_of(obj))

    def play_animation(self, obj, action_name):
        figure = getattr(obj, 'figure', None)
        if figure is None:
            return False
        # a body does not take the animations of the script (the dead unit takes no orders)
        if getattr(obj, 'is_dead', False):
            return False
        action = figure.database.by_name(action_name)
        if action is None:
            return False
        obj.play_action(action)
        return True

    def hide_object(self, obj, hidden):
        if not hasattr(obj, 'hidden'):
            return False
        obj.hidden = hidden
        return True

    def lever_switched(self, obj):
        """
        5b8b30 / 5b8b80: the remake builds the whole grid again, the units in the closed points go out (5ba9d0).
        """
        if self.app.terrain is not None:
            self.app.build_path_grid()
            self.app.push_units_out()

    def place_object(self, obj, x, y, z, push):
        if not isinstance(obj, MapFigure) or self.app.terrain is None:
            return False
        obj.move(x, y, z, self.app.terrain.height_at(x, y))
        self.app.build_path_grid()
        if push:
            self.app.push_units_out()
        return True

    def remove_object(self, obj):
        """
        5935c0: a unit leaves the map, the remake hides it and takes it from the world and its logic;
        a figure of the map goes away from the view and from the grid of the paths.
        """
        if isinstance(obj, MapFigure):
            obj.remove()
            if self.app.terrain is not None:
                self.app.build_path_grid()
            return True
        if not hasattr(obj, 'hidden'):
            return False
        obj.hidden = True
        app = self.app
        if obj in app.map_units:
            app.map_units.remove(obj)
        if getattr(obj, 'brain', None) is not None and obj.brain in app.ai.brains:
            app.ai.brains.remove(obj.brain)
        return True

    def party(self, player):
        if player == 0:
            return self.app.party
        return None

    def roster(self, player):
        if player == 0:
            return self.app.roster
        return None

    def mercs_number(self, player):
        """
        591740: the living units of the party in the zone less the hero.
        """
        if player != 0:
            return 0
        return max(len(self.app.zone_party()) - 1, 0)

    def fix_items(self):
        return self.app.fix_items()

    def party_units(self, player):
        """
        The living units of the party in the zone, the hero first.
        """
        if player != 0:
            return []
        return self.app.zone_party()

    def recalc_mercenaries(self):
        self.app.recalc_mercenaries()

    def complete_merc_briefing(self, variable):
        self.app.complete_merc_briefing(variable)

    def play_movie(self, name):
        return self.app.play_movie(name)

    def redeploy_party(self, player):
        return player == 0 and self.app.redeploy_party()

    def string_event(self, player, text):
        words = text.split()
        if not words:
            return
        if _Debug:
            print('script event %r' % text)
        if words[0] == TUTORIAL_EVENT and len(words) > 1:
            self.app.tutorial.show(words[1])
        elif words[0] == BRIEFING_EVENT and len(words) > 1:
            self.app.show_briefing(words[1])
        elif words[0] in SAY_EVENTS and len(words) > 2:
            self.say(words[1], words[2])
        elif words[0] == AUTOSAVE_EVENT:
            self.app.autosave()

    def speaker(self, name):
        """
        The unit of a say event: game.exe finds it by the id 461360 makes of the name, the remake by OBJNAME,
        "Hero" is the hero.
        """
        obj = self.objects_by_name.get(name)
        if obj is None and name == HERO_NAME:
            obj = self.app.unit
        return obj if hasattr(obj, 'prototype_name') else None

    def say(self, phrase_name, unit_name):
        """
        "say <phrase> <unit>": the unit says the phrase of the scenario of its prototype in acks.db (6d8a00)
        and the text "say <phrase>" of texts.res goes to the messages as "<unit>: <text>" (636700);
        the English texts are empty, the phrases are only heard.
        """
        unit = self.speaker(unit_name)
        if unit is None or unit.prototype_name is None:
            return
        acks = self.app.game_data.acks().get(unit.prototype_name)
        phrase = acks.phrase(idb.ACKS_SCENARIO, phrase_name) if acks is not None else None
        if phrase is None:
            return
        self.app.audio.play_sound(magic.SpellSound(phrase.sound_name, (unit.x, unit.y, unit.z)))
        said = (self.app.game_data.text('say %s' % phrase_name) or '').strip()
        if said:
            self.app.show_message('%s: %s' % (unit_name, said))

#------------------------------------------------------------------------------


def program_of(mob_root, outer_variables=None):
    """
    eilib.script.Program of the SS_TEXT of the .mob, None when the map has no scripts; outer_variables are
    the globals of the map for a .mob of AddMob.
    """
    if mob_root is None:
        return None
    for node in mob_root.walk('SS_TEXT'):
        if node.value:
            return script_lang.parse(node.value, outer_variables)
    return None
