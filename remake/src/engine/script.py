"""
The runtime of the scripts of a map (eilib.script), as the script world of game.exe runs them.

Every tick of the server (58dcd0, 55 ms) calls 4bf000 which:

    1. runs the threads (4b6760) of the list +0x160 and removes those which return 0;
    2. goes over the instances of the scripts (list +0x144): for each branch "if ... then" of the script
       4bb520 computes the conditions (4b41e0) one by one and stops at the first which is 0; when all
       are nonzero 4bc420 starts the commands of the branch as a thread. As soon as the instance is
       killed (KillScript) the next branches are not checked and the instance is removed.

A thread [code, pc, runs, sleep start, kill flag] (4b6760) adds 1 to its runs, returns 0 when pc is at
the end, otherwise runs the commands from pc on: a blocking command keeps pc and the thread returns 1
to go on at the next tick, after the last command it returns 1 too and so it is removed one tick later.
4bc420 runs the new thread at once and puts it into the list only when it returned 1; the copy in the list
has no kill flag, so KillScript works only before the first blocking command.

The blocking commands: Sleep(n) (7) waits n runs of the thread: it takes the runs at its first try and
goes on when the runs grew by n; WaitSegment (9) waits for the runs to reach n; SleepUntil(c) (0xde)
waits for a nonzero c, the arguments are computed again at each try; SleepUntilIdle(o) (0xb9) waits
while the unit has a command of the AI.

A call of a declared script (command 8, 4c7c90) makes a new instance with the values of the arguments
(the groups are copied) at the head of the list +0x144: called from a thread it is checked at the same
tick, called from a branch only at the next one.

For(x, g) (0x6b) puts the members of the group into x from the last to the first and runs its block
once for each, at once, as a thread of its own: a blocking command there ends the block. Every(x, g, c)
(0x68) and Any (0x69) compute c for the members from the first on, Every of an empty group is 1, Any 0.
A group (4b2ef0) holds every object once and no NULL; game.exe keeps them sorted by the address,
the remake in the order they were added. Random(n) (0x5f) is rand32 * n / 2^32. The distances (0x53, 0x54) are on the ground (x, y),
1e30 for a NULL object; GetX and GetY of NULL are 0.

The orders of the units go to their logic (engine.ai.Controller, the object +0x240 of a unit):
MoveToPoint (5d7ae0) and RotateTo (5d7b30) set the state 1 which 5d53b0 keeps while the unit goes or turns,
Follow (5d7e40) the state 6, Idle (5d4df0 + 5d7790) clears the behaviours and the state; Walk and Run
(555270 with 2 and 3) choose the way of the orders; UMClear (5d4df0) removes the behaviours, UMSentry
(4ba360, weight 50: it walks to its point when the logic is free) and UMGuard add one, UMStandard
the standard set; SetCP (0x8e, 0xad) puts the object at the point at once (5c11d0) and a unit drops its
orders (554450); KillUnit (0x35) kills, InflictDamage (0x44) gives a blow of the lightning to the torso which
always hits; PlayMusic plays a track (engine.music) for a player with a party; CastSpellPoint and CastSpellUnit cast without a caster from a point, GetFutureX / GetFutureY give
the place of a unit after some ticks; ActivateTrap (0x95) turns a magic trap (engine.trap) on or off; UMPlayer (0x31) gives the behaviour of a unit of the player.
UMAggression (0x2d) and UMRevenge (0x2e) add the behaviour 73d260 which attacks the enemies which the unit sees or
only the one who hit it, UMFear( o, f ) (0xc7) the caution of 5d2850 (a shooter keeps away), UMFollow( o, leader )
(0xca) the behaviour 73d1c4 which follows the leader when nothing outweighs it.
SleepUntilIdle waits while the unit goes or turns.
The levers (engine.lever): GetLeverState, SwitchLeverState, EnableLever and SetScience.
The things of a player (engine.party.Party of Host.party): GiveItem, GiveQuestItem, EraseQuestItem, HaveItem,
GiveMoney, GetMoney and QuestComplete; a player without a party takes nothing and has nothing. QuestComplete also
shares the experience of the quest among the living units of the party in the zone (engine.experience).
GiveSkill, GiveStrength, GiveDexterity and GiveIntelligence change engine.experience.HeroStats of a hero.
The parties between the zones (engine.roster.Roster of Host.roster): CreateParty, AddUnitToParty, RemoveUnitFromParty,
SetCurrentParty, CopyStats, CopyItems, CopyLoot, AddLoot, GetMercsNumber and RedeployParty; the things of the current
party are the items of engine.party.Party, CopyLoot, AddLoot and SetCurrentParty move them between the two.

The diplomacy (0x37, 0x60): SetDiplomacy( a, b, v ) gives the player a the relation to b by the sign of v
(above 0 the same side, 0 neutral, below 0 war, 4554e0), GetDiplomacy answers 1, 0 or -1 the same way.
The alarms: InvokeAlarm( n, x, y ) (0x43) raises the alarm n at the point by force, IsAlarm, AlarmTime,
AlarmPosX and AlarmPosY read the table 7affdc. UnitSee( o ) (0x5d) is the list of the units which the logic
of the unit sees, GroupSee the same of a group; PlayerSee adds the units of the player themselves.
UnitInSquare( o, x0, y0, x1, y1 ) (0xbc) checks the place of the object inside the rect with the borders.
The areas (0xc3..0xc6): AddRectToArea( n, x0, y0, x1, y1 ) and AddRoundToArea( n, x, y, r ) add a shape
to the area n, IsInArea( n, x, y ) is 1 inside any of them (a circle: d^2 < r^2), DeleteArea clears it.
The time of the world (0xb5..0xb7): GetWorldTime is (ticks * rate + start) mod 24 in hours,
FixWorldTime( h ) stops the time at h, RunWorldTime( h ) sets it and lets it run.

AddMob( name ) (0xa0, 595ab0) loads the objects of another .mob into the map once (596030, the names are kept
in +0x128); its scripts join the world: their global variables are those of the map when the names are the same
(zone19alarm1 fills the group a1 of zone19), its WorldScript starts, and a call goes to the script of the same
.mob first (the editor gives the same names like VCheck#3#1 in every .mob).

The particle sources and the point lights of CreateParticleSource, CreatePointLight and the rest are
engine.sources.ScriptSources of the world (ScriptWorld.sources), CreateFX plays a sound through the host,
CreateFXSource keeps one which plays round until DeleteFXSource.
PlayMovie( name ) (0xc1) asks the host to play data/Movies/<name> (5db3f0), the world waits while it plays.

The functions of the units and the world come from a host (Host): the client gives the objects
of the map, the rest of the commands which are not made yet are counted in ScriptWorld.missing.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import random

#------------------------------------------------------------------------------

from eilib import script as script_lang
from engine import ai
from engine import combat
from engine import experience
from engine import lever as lever_module
from engine import particles
from engine import sources as sources_module
from engine import trap as trap_module
from engine import water

#------------------------------------------------------------------------------

# 0x53 / 0x54: the distance of NULL
NO_DISTANCE = 1e30

# 592090: the script which the end of a briefing calls, and the value of the variable of a briefing which was seen
BRIEFING_COMPLETE_SCRIPT = '#OnBriefingComplete'
BRIEFING_SEEN = 2.0

# InflictDamage (0x44): the packet has only the part 1 of the damage of the type 5 (the lightning)
INFLICT_DAMAGE_TYPE = 5

# GetWorldTime: fmod by 24 (73d108)
HOURS = 24.0

DEFAULTS = {
    script_lang.FLOAT: 0.0,
    script_lang.STRING: '',
    script_lang.OBJECT: None,
    script_lang.GROUP: (),
}

#------------------------------------------------------------------------------


class Host(object):
    """
    The world the scripts see; the client overrides it.
    """

    def object_by_id(self, nid):
        """
        The object with the NID of the .mob, None when there is none.
        """
        return None

    def object_by_name(self, name):
        return None

    def object_id(self, obj):
        return 0

    def is_dead(self, obj):
        return bool(getattr(obj, 'is_dead', False))

    def position(self, obj):
        """
        (x, y, z) of the object, None when it has no place.
        """
        if obj is None or not hasattr(obj, 'x'):
            return None
        return (obj.x, obj.y, getattr(obj, 'z', 0.0))

    def unit_of_player(self, player, index):
        """
        GetUnitOfPlayer( player, index ) (4b56c0 case 0xb4): the unit number index of the list of the player,
        None when there is none.
        """
        return None

    def leader(self):
        """
        GetLeader (4b56c0 case 0xe4): the leading unit of the party, None when there is none.
        """
        return None

    def seen_by_player(self, player):
        """
        PlayerSee( player ): the units the units of the player see.
        """
        return ()

    def is_enemy(self, obj, player):
        return False

    def play_animation(self, obj, action_name):
        """
        PlayAnimation( unit, action ): False when it is not made.
        """
        return False

    def hide_object(self, obj, hidden):
        """
        HideObject( o, f ) (0xc2): the flag 0x10000 of the object; False when it is not made.
        """
        return False

    def place_object(self, obj, x, y, z, push):
        """
        SetCP / SetCPFast of an object which is not a unit: 5c11d0 puts it at x, y and z over the land;
        push (SetCP, not SetCPFast) moves the units out of its new place (5ba9d0). False when it is not made.
        """
        return False

    def lever_switched(self, obj):
        """
        SwitchLeverState changed the lever of the object: the grid of the paths around it is built again.
        """

    def remove_object(self, obj):
        """
        RemoveUnitFromServer / RemoveObjectFromServer (0x89, 0x96, 5935c0); False when it is not made.
        """
        return False

    def string_event(self, player, text):
        """
        SendStringEvent( player, text ), for example "#output 150".
        """

    def event(self, player, number):
        """
        SendEvent( player, number ).
        """

    def party(self, player):
        """
        engine.party.Party of the player (594180), None when the player has none.
        """
        return None

    def roster(self, player):
        """
        engine.roster.Roster of the parties of the player, None when the player has none.
        """
        return None

    def mercs_number(self, player):
        """
        GetMercsNumber (591740): the living units of the party of the player in the zone without the hero.
        """
        return 0

    def recalc_mercenaries(self):
        """
        RecalcMercBriefings (0xd2, 591800): the topics of the mercenaries of the parties.
        """

    def complete_merc_briefing(self, variable):
        """
        591de0: the mercenary of a finished briefing joins the party or leaves it.
        """

    def fix_items(self):
        """
        FixItems (0xdf): every item of the world (the list +0x54 of CWorldServer) gets its durability back,
        +0x20 = +0x24 (the camp repairs the items of a unit the same way, 6111e0); the client does it.
        """
        return

    def party_units(self, player):
        """
        The living units of the party of the player in the zone, the hero first (5911c0 shares the experience
        of a quest among them).
        """
        return []

    def play_movie(self, name):
        """
        PlayMovie( name ) (0xc1): data/Movies/<name> over the screen (5db3f0), the game waits; False when it is
        not played.
        """
        return False

    def redeploy_party(self, player):
        """
        RedeployParty (0xd3): the current party of the player comes into the zone again; False when it is not made.
        """
        return False

    def add_mob(self, name, variables):
        """
        AddMob: puts the objects of the .mob into the map and gives the eilib.script.Program of its scripts
        parsed with the globals (name -> type) of the world, None when it has none.
        """
        return None

    def leave_to_zone(self, player, zone_name, exit_index):
        """
        LeaveToZone( player, zone, exit ) (0x8a, 66f2d0): the party goes to the zone of map.txt and comes in
        by its exit, an index of the list of the exits from 0.
        """

    def make_emitter(self, code, position, scale):
        """
        The engine.particles.Emitter of a source of CreateParticleSource, ValueError for a type which is not made.
        """
        return particles.Emitter(code, position, scale)

    def play_music(self, name):
        """
        PlayMusic: the track plays once in the zone (engine.music); False when it is not made.
        """
        return False

    def cast_spell(self, text, origin, target, point):
        """
        A spell of the scripts from the point origin (x, y) at the unit target or the point (x, y);
        False when it is not made.
        """
        return False

    def play_fx(self, position, near, far, sound_name):
        """
        CreateFX( x, y, z, near, far, wav ) (0x17, 4bd260): the sound of sfx.res at the point.
        """

    def world_time(self):
        """
        The hour of the world, 0..24.
        """
        return 0.0

    def set_world_time(self, hour, runs):
        """
        FixWorldTime (runs False) and RunWorldTime (runs True).
        """

#------------------------------------------------------------------------------


class Instance(object):
    """
    A copy of a script waiting for its conditions (an element of +0x144).
    """

    def __init__(self, script, values, program=None):
        self.script = script
        self.values = values
        self.killed = False
        # the eilib.script.Program of the .mob of the script
        self.program = program

    def __repr__(self):
        return 'Instance(%s%s)' % (self.script.name, ' killed' if self.killed else '')


class Thread(object):
    """
    The commands which run (an element of +0x160): pc, runs and the start of a Sleep as in 4b6760.
    """

    def __init__(self, commands, values, instance=None, program=None):
        self.commands = commands
        self.values = values
        self.instance = instance
        self.program = program
        self.pc = 0
        self.runs = 0
        self.sleep_start = -1

    def __repr__(self):
        return 'Thread(%d/%d)' % (self.pc, len(self.commands))

#------------------------------------------------------------------------------


class ScriptWorld(object):

    def __init__(self, program, host=None, game_vars=None, rng=None, controller=None):
        self.program = program
        self.host = host or Host()
        # GSSetVar / GSGetVar: the variables of the game which live through the zones
        self.game_vars = game_vars if game_vars is not None else {}
        self.rng = rng or random.Random()
        # engine.ai.Controller which takes the orders of the units
        self.controller = controller or ai.Controller()
        # name -> type of the globals, those of the .mob files of AddMob too
        self.variables = dict(program.variables)
        self.globals = {name: _default(kind) for name, kind in program.variables.items()}
        self.instances = []
        self.threads = []
        # the name of a command or a function which is not made yet -> how many times it was called
        self.missing = {}
        self.ticks = 0
        # AddRectToArea / AddRoundToArea: number -> [(x0, y0, x1, y1) or (x, y, r^2)]
        self.areas = {}
        # engine.sources: the particle sources and the point lights of the scripts, the client runs them
        self.sources = sources_module.ScriptSources(self.host.make_emitter)
        # the levels of the liquids (CWorld +0x70); the client ticks them after the scripts, like 58dcd0
        self.water_levels = water.WaterLevels()
        self.functions = {
            'IsEqual': lambda a, b: _bool(a == b),
            'IsLess': lambda a, b: _bool(a < b),
            'IsGreater': lambda a, b: _bool(a > b),
            'IsEqualString': lambda a, b: _bool(a == b),
            'Add': lambda a, b: a + b,
            'Sub': lambda a, b: a - b,
            'Mul': lambda a, b: a * b,
            'Div': self.divide,
            'Not': lambda a: _bool(a == 0.0),
            'Random': lambda n: self.rng.getrandbits(32) * n / 4294967296.0,
            'GetObject': lambda nid: self.host.object_by_id(int(nid)),
            'GetObjectByID': self.object_by_id_text,
            'GetObjectByName': lambda name: self.host.object_by_name(name),
            'GetObjectID': lambda obj: float(self.host.object_id(obj)) if obj is not None else 0.0,
            'GroupSize': lambda group: float(len(group)),
            'GroupHas': lambda group, obj: _bool(obj is not None and obj in group),
            'GroupAdd': lambda a, b: _group_add(a, b),
            'GroupSub': lambda a, b: tuple(o for o in a if o not in b),
            'GroupCross': lambda a, b: tuple(o for o in a if o in b),
            'GSGetVar': lambda player, name: float(self.game_vars.get(name, 0.0)),
            'IsDead': lambda obj: _bool(obj is not None and self.host.is_dead(obj)),
            # the flags +8 of the object: 0x100000 looted (NULL counts as looted), 0x800000 blocked
            'WasLooted': lambda obj: _bool(obj is None or getattr(obj, 'looted', False)),
            'IsUnitBlocked': lambda obj: _bool(obj is not None and getattr(obj, 'blocked', False)),
            'IsAlive': lambda obj: _bool(obj is not None and not self.host.is_dead(obj)),
            'GetX': lambda obj: self.coordinate(obj, 0),
            'GetY': lambda obj: self.coordinate(obj, 1),
            'GetZ': lambda obj: self.coordinate(obj, 2),
            'DistanceUnitUnit': self.distance_unit_unit,
            'DistanceUnitPoint': self.distance_unit_point,
            'GetUnitOfPlayer': lambda player, index: self.host.unit_of_player(int(player), int(round(index))),
            'GetLeader': lambda: self.host.leader(),
            'PlayerSee': lambda player: tuple(self.host.seen_by_player(int(player))),
            'IsEnemy': lambda obj, player: _bool(obj is not None and self.host.is_enemy(obj, int(player))),
            'GetLeverState': self.lever_state,
            'GetDiplomacy': self.get_diplomacy,
            'IsAlarm': lambda n: _bool(int(n) in self.controller.alarms),
            'AlarmTime': lambda n: float(self.controller.alarms.get(int(n), (0, (0.0, 0.0)))[0]),
            'AlarmPosX': lambda n: float(self.controller.alarms.get(int(n), (0, (0.0, 0.0)))[1][0]),
            'AlarmPosY': lambda n: float(self.controller.alarms.get(int(n), (0, (0.0, 0.0)))[1][1]),
            'UnitSee': self.unit_see,
            'GroupSee': self.group_see,
            'IsPlayerInDanger': lambda player: _bool(self.controller.in_danger(int(player))),
            'UnitInSquare': self.unit_in_square,
            'IsInArea': self.is_in_area,
            'GetFutureX': lambda obj, ticks: self.future(obj, ticks, 0),
            'GetFutureY': lambda obj, ticks: self.future(obj, ticks, 1),
            'GetWorldTime': lambda: float(self.host.world_time()) % HOURS,
            'HaveItem': lambda player, number: _bool(self.party_call(player, 'has_quest_item', int(number))),
            'GetMoney': lambda player: float(getattr(self.host.party(int(player)), 'money', 0)),
            'GetMercsNumber': lambda player: float(self.host.mercs_number(int(player))),
            'RecalcMercBriefings': self.host.recalc_mercenaries,
        }
        self.commands = {
            'GSSetVar': self.gs_set_var,
            'GSSetVarMax': self.gs_set_var_max,
            'GSDelVar': lambda player, name: self.game_vars.pop(name, None),
            'SendStringEvent': lambda player, text: self.host.string_event(int(player), text),
            'SendEvent': lambda player, number: self.host.event(int(player), int(number)),
            'ConsoleFloat': self.console,
            'ConsoleString': self.console,
            'PlayAnimation': self.play_animation,
            'MoveToPoint': self.unit_order(self.controller.order_move),
            'RotateTo': self.unit_order(self.controller.order_turn),
            'Follow': self.unit_order(self.controller.order_follow),
            'Walk': self.unit_order(lambda u: self.controller.set_gait(u, False)),
            'Run': self.unit_order(lambda u: self.controller.set_gait(u, True)),
            'ResetTarget': self.unit_order(self.controller.reset_target),
            'UMClear': self.unit_order(self.controller.clear_behaviours),
            'UMPlayer': self.unit_order(self.controller.player_behaviour),
            'Idle': self.unit_order(self.controller.stop_orders),
            'UMStandard': self.unit_order(self.controller.standard_behaviours),
            'UMAggression': self.unit_order(self.controller.aggression),
            'UMRevenge': self.unit_order(self.controller.revenge),
            'UMFear': self.unit_order(lambda u, f: self.controller.fear(u, f != 0.0)),
            'UMFollow': self.unit_order(self.controller.follow_behaviour),
            'UMSentry': self.unit_order(self.controller.sentry),
            'Sentry': self.unit_order(self.controller.sentry),
            'UMGuard': self.unit_order(self.controller.guard),
            'Guard': self.unit_order(self.controller.guard),
            'SwitchLeverState': self.switch_lever,
            'EnableLever': self.enable_lever,
            'SetScience': set_science,
            'SetDiplomacy': self.set_diplomacy,
            'InvokeAlarm': lambda n, x, y: self.controller.raise_alarm(int(n), (x, y), force=True),
            # CScriptArea::AddRect (4b1ed0) and AddRound (4b2070) add a record of the shape to the area
            'AddRectToArea': lambda n, x0, y0, x1, y1: self.areas.setdefault(int(n), []).append((x0, y0, x1, y1)),
            'AddRoundToArea': lambda n, x, y, r: self.areas.setdefault(int(n), []).append((x, y, r * r)),
            'DeleteArea': lambda n: self.areas.pop(int(n), None),
            'FixWorldTime': lambda hour: self.host.set_world_time(hour, False),
            'RunWorldTime': lambda hour: self.host.set_world_time(hour, True),
            'LeaveToZone': lambda player, zone_name, index: self.host.leave_to_zone(int(player), zone_name, int(index)),
            'AddMob': lambda name: self.add_program(self.host.add_mob(name, self.variables)),
            'GiveItem': lambda player, name: self.party_call(player, 'give_item', name),
            'GiveQuestItem': lambda player, name: self.party_call(player, 'give_item', name),
            'EraseQuestItem': lambda player, number: self.party_call(player, 'erase_quest_item', int(number)),
            'GiveMoney': lambda player, amount: self.party_call(player, 'give_money', amount),
            'QuestComplete': self.quest_complete,
            'CreateParty': lambda player, name: self.roster_call(player, 'create_party', name),
            'AddUnitToParty': lambda player, path, prototype_name: self.roster_call(player, 'add_unit', path,
                                                                                    prototype_name),
            'RemoveUnitFromParty': lambda player, path: self.roster_call(player, 'remove_unit', path),
            'CopyStats': lambda player, source, dest: self.roster_call(player, 'copy_stats', source, dest),
            'CopyItems': lambda player, source, dest: self.roster_call(player, 'copy_items', source, dest),
            'CopyLoot': lambda player, source, dest: self.move_loot(player, 'copy_loot', source, dest),
            'AddLoot': lambda player, source, dest: self.move_loot(player, 'copy_loot', source, dest, True),
            'SetCurrentParty': lambda player, name: self.move_loot(player, 'set_current', name),
            'RedeployParty': self.redeploy_party,
            'CreateParticleSource': self.create_particles,
            'SetParticleSourceSize': self.sources.set_particles_size,
            'MoveParticleSource': lambda nid, x, y, z, ticks, flag: self.sources.move_particles(nid, (x, y, z), ticks),
            'DeleteParticleSource': self.sources.delete_particles,
            'AttachParticles': self.sources.attach_particles,
            'PlayMovie': self.play_movie,
            'CreatePointLight': lambda nid, x, y, z, radius, r, g, b: self.sources.create_light(nid, (x, y, z), radius,
                                                                                               (r, g, b)),
            'MovePointLight': lambda nid, x, y, z, ticks, flag: self.sources.move_light(nid, (x, y, z), ticks),
            'DeletePointLight': self.sources.delete_light,
            'CreateFXSource': lambda nid, x, y, z, near, far, sound_name: self.sources.create_sound(
                nid, (x, y, z), near, far, sound_name),
            'DeleteFXSource': self.sources.delete_sound,
            'CreateLightning': lambda nid, x0, y0, z0, x1, y1, z1, power: self.sources.create_lightning(
                nid, (x0, y0, z0), (x1, y1, z1), power),
            'DeleteLightning': self.sources.delete_lightning,
            'CreateFX': lambda x, y, z, near, far, sound_name: self.host.play_fx((x, y, z), near, far, sound_name),
            'SetCP': lambda obj, x, y, z: self.set_place(obj, x, y, z, True),
            'BlockUnit': self.block_unit,
            'GiveSkill': self.give_skill,
            'FixItems': self.host.fix_items,
            'GiveStrength': lambda obj, amount: self.give_attribute(obj, 'give_strength', amount),
            'GiveDexterity': lambda obj, amount: self.give_attribute(obj, 'give_dexterity', amount),
            'GiveIntelligence': lambda obj, amount: self.give_attribute(obj, 'give_intelligence', amount),
            'SetWaterLevel': lambda water_index, level, ticks: self.water_levels.set_level(
                int(round(water_index)), level, int(round(ticks))),
            'RemoveQuestItem': self.remove_quest_item,
            'SetCPFast': lambda obj, x, y, z: self.set_place(obj, x, y, z, False),
            'KillUnit': self.unit_order(kill_unit),
            'InflictDamage': self.unit_order(inflict_damage),
            'ActivateTrap': activate_trap,
            'PlayMusic': self.play_music,
            'CastSpellPoint': lambda name, x0, y0, x1, y1: self.cast_spell('CastSpellPoint', name, (x0, y0), None,
                                                                           (x1, y1)),
            'CastSpellUnit': lambda name, x0, y0, obj: self.cast_spell('CastSpellUnit', name, (x0, y0), obj, None),
            'HideObject': lambda obj, hidden: self.host_command('HideObject', self.host.hide_object, obj, hidden != 0.0),
            'RemoveUnitFromServer': lambda obj: self.host_command('RemoveUnitFromServer', self.host.remove_object, obj),
            'RemoveObjectFromServer': lambda obj: self.host_command('RemoveObjectFromServer', self.host.remove_object,
                                                                    obj),
        }

    def __repr__(self):
        return 'ScriptWorld(%d instances, %d threads)' % (len(self.instances), len(self.threads))

    def start(self):
        """
        Starts WorldScript as a thread.
        """
        self.start_thread(self.program.world_commands, {}, program=self.program)

    def add_program(self, program):
        """
        AddMob: the scripts of another .mob join the world, their WorldScript starts at once.
        """
        if program is None:
            return
        for name, kind in program.variables.items():
            if name not in self.globals:
                self.variables[name] = kind
                self.globals[name] = _default(kind)
        self.start_thread(program.world_commands, {}, program=program)

    def tick(self):
        """
        4bf000: the threads, then the instances of the scripts.
        """
        self.ticks += 1
        for thread in list(self.threads):
            if not self.run(thread):
                self.threads.remove(thread)
        for instance in list(self.instances):
            for branch in instance.script.branches:
                self.check(instance, branch)
                if instance.killed:
                    break
            if instance.killed:
                self.instances.remove(instance)

    def check(self, instance, branch):
        """
        4bb520: all the conditions nonzero start the commands of the branch.
        """
        for condition in branch.conditions:
            if self.evaluate(condition, instance.values) == 0.0:
                return
        self.start_thread(branch.commands, dict(instance.values), instance, instance.program)

    def start_thread(self, commands, values, instance=None, program=None):
        """
        4bc420: the thread runs at once, it is kept when it is not done, without the kill flag.
        """
        thread = Thread(commands, values, instance, program)
        if self.run(thread):
            thread.instance = None
            self.threads.insert(0, thread)
        return thread

    def briefing_variable(self, name):
        """
        608410: the event "briefing <name>" shows the briefing of the variable of the game "b.<owner>.<name>"
        (split by the dots, the third part compared without the case); the scripts set it to 1 beforehand,
        for example GSSetVar( 0, "b.elder.s71_1", 1 ) on bz18h. None when there is no such variable.
        """
        for variable in self.game_vars:
            parts = variable.split('.')
            if len(parts) == 3 and parts[2].lower() == name.lower():
                return variable
        return None

    def complete_briefing(self, player, variable):
        """
        592090 at the end of a briefing runs "WorldScript( GSSetVar( player, variable, 2 )
        #OnBriefingComplete( player, variable ) )"; 591de0 before it makes a mercenary join the party or
        leave it, 591800 recalculates the topics of the mercenaries.
        """
        self.gs_set_var(player, variable, BRIEFING_SEEN)
        self.host.complete_merc_briefing(variable)
        if BRIEFING_COMPLETE_SCRIPT in self.program.scripts:
            self.call_script(BRIEFING_COMPLETE_SCRIPT, [float(player), variable])
        self.host.recalc_mercenaries()

    def call_script(self, name, args, program=None):
        """
        Command 8: a new instance of a declared script at the head of the list; the script of the .mob
        of the caller comes first. The arguments go to the parameters by their types (SetScriptArgs 4bb6a0).
        """
        program = program or self.program
        if name not in program.scripts:
            program = self.program
        script = program.scripts.get(name)
        if script is None:
            self.note_missing(name)
            return None
        values = {}
        for (param_name, kind), value in zip(script.params, args):
            values[param_name] = tuple(value) if kind == script_lang.GROUP else value
        instance = Instance(script, values, program)
        self.instances.insert(0, instance)
        if _Debug:
            print('script %s(%s)' % (name, ', '.join(map(repr, args))))
        return instance

    def run(self, thread):
        """
        4b6760: True while the thread has to run at the next tick.
        """
        thread.runs += 1
        if thread.pc >= len(thread.commands):
            return False
        while thread.pc < len(thread.commands):
            if not self.execute(thread, thread.commands[thread.pc]):
                return True
            thread.pc += 1
        return True

    def execute(self, thread, command):
        """
        One command; False when it blocks the thread.
        """
        values = thread.values
        if isinstance(command, script_lang.Assign):
            self.assign(command.name, self.evaluate(command.value, values), values)
            return True
        name = command.name
        if name == 'For':
            self.run_for(thread, command)
            return True
        args = [self.evaluate(a, values) for a in command.args]
        if command.function is None:
            self.call_script(name, args, thread.program)
            return True
        if name == 'Sleep':
            if thread.sleep_start < 0:
                thread.sleep_start = thread.runs
            if thread.runs - thread.sleep_start < int(round(args[0])):
                return False
            thread.sleep_start = -1
            return True
        if name == 'WaitSegment':
            return thread.runs >= int(round(args[0]))
        if name == 'SleepUntil':
            return args[0] != 0.0
        if name == 'SleepUntilIdle':
            return not (is_unit(args[0]) and ai.is_busy(args[0]))
        if name == 'KillScript':
            if thread.instance is None:
                if _Debug:
                    print('KillScript command not allowed here')
            else:
                thread.instance.killed = True
            return True
        if name == 'AddObject':
            group_name = command.args[0].name
            obj = args[1]
            if obj is not None and obj not in args[0]:
                self.assign(group_name, tuple(args[0]) + (obj, ), values)
            return True
        handler = self.commands.get(name)
        if handler is None:
            self.note_missing(name)
            return True
        handler(*args)
        return True

    def run_for(self, thread, command):
        """
        0x6b: the block once for each member, from the last one, a blocking command ends it.
        """
        variable = command.args[0].name
        group = self.evaluate(command.args[1], thread.values)
        for obj in reversed(group):
            self.assign(variable, obj, thread.values)
            inner = Thread(command.block, thread.values, thread.instance, thread.program)
            self.run(inner)

    def evaluate(self, expression, values):
        if isinstance(expression, script_lang.Number):
            return expression.value
        if isinstance(expression, script_lang.String):
            return expression.value
        if isinstance(expression, script_lang.Variable):
            return self.value(expression.name, values)
        name = expression.name
        if name in ('Every', 'Any'):
            return self.every_any(expression, values)
        args = [self.evaluate(a, values) for a in expression.args]
        function = self.functions.get(name)
        if function is None:
            self.note_missing(name)
            return _default(expression.function.returns)
        return function(*args)

    def every_any(self, expression, values):
        """
        0x68 / 0x69: the condition for the members from the first one.
        """
        every = expression.name == 'Every'
        variable = expression.args[0].name
        for obj in self.evaluate(expression.args[1], values):
            self.assign(variable, obj, values)
            result = self.evaluate(expression.args[2], values)
            if every and result == 0.0:
                return 0.0
            if not every and result != 0.0:
                return 1.0
        return 1.0 if every else 0.0

    def value(self, name, values):
        if name in values:
            return values[name]
        return self.globals.get(name)

    def assign(self, name, value, values):
        if name in values:
            values[name] = value
        else:
            self.globals[name] = value

    def play_movie(self, name):
        if not self.host.play_movie(name):
            self.note_missing('PlayMovie %s' % name)

    def note_missing(self, name):
        if name not in self.missing:
            if _Debug:
                print('script: %s is not made yet' % name)
        self.missing[name] = self.missing.get(name, 0) + 1

    def object_by_id_text(self, text):
        try:
            nid = int(text)
        except ValueError:
            return None
        return self.host.object_by_id(nid)

    def divide(self, a, b):
        if b == 0.0:
            return 0.0
        return a / b

    def coordinate(self, obj, index):
        position = self.host.position(obj) if obj is not None else None
        if position is None:
            return 0.0
        return float(position[index])

    def distance_unit_unit(self, a, b):
        pa = self.host.position(a) if a is not None else None
        pb = self.host.position(b) if b is not None else None
        if pa is None or pb is None:
            return NO_DISTANCE
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    def distance_unit_point(self, obj, x, y):
        position = self.host.position(obj) if obj is not None else None
        if position is None:
            return NO_DISTANCE
        return math.hypot(position[0] - x, position[1] - y)

    def gs_set_var(self, player, name, value):
        self.game_vars[name] = value

    def gs_set_var_max(self, player, name, value):
        self.game_vars[name] = max(self.game_vars.get(name, 0.0), value)

    def play_animation(self, obj, action_name):
        if obj is None:
            return
        if not self.host.play_animation(obj, action_name):
            self.note_missing('PlayAnimation')

    def unit_order(self, order):
        """
        A command whose first argument is a unit: NULL or another object does nothing, as 4b2530 gives none.
        """
        def _command(obj, *args):
            if is_unit(obj) and not obj.is_dead:
                order(obj, *args)
        return _command

    def lever_state(self, obj):
        """
        GetLeverState (0x90): +0x238 of a lever, 0 for another object.
        """
        lever = lever_of(obj)
        return float(lever.state) if lever is not None else 0.0

    def switch_lever(self, obj, state):
        """
        SwitchLeverState (0x91): 52b7a0 with the state and the time -10.
        """
        lever = lever_of(obj)
        if lever is None:
            return
        lever.switch(int(state))
        self.host.lever_switched(obj)

    def enable_lever(self, obj, enabled):
        """
        EnableLever (0xcc, 52bad0): the flag 0x80000 forbids the use.
        """
        lever = lever_of(obj)
        if lever is not None:
            lever.enabled = enabled != 0.0

    def set_diplomacy(self, player, other, value):
        relation = ai.SAME_SIDE if value > 0.0 else (ai.NEUTRAL if value == 0.0 else ai.WAR)
        self.controller.diplomacy.set_relation(int(player), int(other), relation)

    def get_diplomacy(self, player, other):
        relation = self.controller.diplomacy.relation(int(player), int(other))
        return {ai.SAME_SIDE: 1.0, ai.WAR: -1.0}.get(relation, 0.0)

    def unit_see(self, obj):
        if not is_unit(obj):
            return ()
        return tuple(self.controller.seen_by(obj))

    def group_see(self, group):
        result = ()
        for obj in group:
            result = _group_add(result, self.unit_see(obj))
        return result

    def unit_in_square(self, obj, x0, y0, x1, y1):
        position = self.host.position(obj) if obj is not None else None
        if position is None:
            return 0.0
        return _bool(x0 <= position[0] <= x1 and y0 <= position[1] <= y1)

    def is_in_area(self, number, x, y):
        for shape in self.areas.get(int(number), ()):
            if len(shape) == 3:
                if (x - shape[0]) ** 2 + (y - shape[1]) ** 2 < shape[2]:
                    return 1.0
            elif shape[0] <= x <= shape[2] and shape[1] <= y <= shape[3]:
                return 1.0
        return 0.0

    def party_call(self, player, method_name, *args):
        """
        A method of the party of the player, None when the player has no party.
        """
        party = self.host.party(int(player))
        if party is None:
            return None
        return getattr(party, method_name)(*args)

    def roster_call(self, player, method_name, *args):
        """
        A method of the parties of the player (engine.roster.Roster), None when the player has none.
        """
        roster = self.host.roster(int(player))
        if roster is None:
            return None
        return getattr(roster, method_name)(*args)

    def move_loot(self, player, method_name, *args):
        """
        A method of the parties which changes the things of the parties: the items of engine.party.Party
        are those of the current party before and after it.
        """
        roster = self.host.roster(int(player))
        if roster is None:
            return None
        things = self.host.party(int(player))
        if things is not None:
            roster.current().loot = list(things.items)
        result = getattr(roster, method_name)(*args)
        if things is not None:
            things.items = list(roster.current().loot)
        return result

    def redeploy_party(self, player):
        if self.host.roster(int(player)) is None:
            return
        if not self.host.redeploy_party(int(player)):
            self.note_missing('RedeployParty')

    def quest_complete(self, player, name):
        """
        QuestComplete (0x9f): 4c9c90 finds the quest of quests.qdb, an unknown name does nothing.
        """
        party = self.host.party(int(player))
        if party is None:
            return
        quest = party.complete_quest(name)
        if quest is None:
            self.note_missing('QuestComplete %s' % name)
            return
        units = [u for u in self.host.party_units(int(player)) if getattr(u, 'hero_stats', None) is not None]
        if quest.experience:
            experience.reward_units(units, quest.experience)

    def give_skill(self, obj, name, amount):
        """
        GiveSkill( unit, "skill", value ) (0xcd): only a hero or a mercenary (the class 0x39) learns; the experience
        grows by the price of the levels (528ab0, 528da0 for the science) and the health and the mana are counted
        again. The names: science, melee, archery, backstab, elemental, sense, astral, stealth, awareness.
        """
        stats = getattr(obj, 'hero_stats', None) if is_unit(obj) else None
        index = experience.SKILL_CODES.get(str(name).lower())
        if stats is None or index is None:
            self.note_missing('GiveSkill %s' % name)
            return
        stats.give_skill(index, int(amount))
        experience.update_fighter(obj.fighter, stats)

    def give_attribute(self, obj, method_name, amount):
        """
        GiveStrength (0xce, 529090), GiveDexterity (0xcf, 529220), GiveIntelligence (0xd0, 529400).
        """
        stats = getattr(obj, 'hero_stats', None) if is_unit(obj) else None
        if stats is None:
            return
        getattr(stats, method_name)(int(amount))
        experience.update_fighter(obj.fighter, stats)

    def set_place(self, obj, x, y, z, push):
        """
        SetCP (0x8e) and SetCPFast (0xad): 5c11d0 puts the object at the point, a unit also drops its orders (554450).
        Another object stands at the height z over the land, like the OBJPOSITION of the .mob; SetCP then pushes
        the units out of the circle of twice its radius (5ba9d0), SetCPFast does not.
        """
        if obj is None:
            return
        if is_unit(obj):
            obj.teleport_to(x, y)
            if obj.brain is not None:
                obj.brain.order = ai.ORDER_NONE
            return
        if not self.host.place_object(obj, x, y, z, push):
            self.note_missing('SetCP of an object')

    def block_unit(self, obj, blocked):
        """
        BlockUnit( unit, f ) (0x32): the flag 0x800000 of a unit which IsUnitBlocked (0xe6) reads.
        """
        if is_unit(obj):
            obj.blocked = blocked != 0.0

    def remove_quest_item(self, obj, number):
        """
        RemoveQuestItem( unit, number ) (0xae): the first item of the unit (+0x4a0) with the number +0x4c
        goes away (5313b0).
        """
        if not is_unit(obj):
            return
        for index, item in enumerate(obj.quest_items):
            if item.number == int(number):
                del obj.quest_items[index]
                return

    def play_music(self, player, name):
        """
        PlayMusic( player, name ) (0x1d): the track of data/stream for a player with a party (594180).
        """
        if self.host.party(int(player)) is None:
            return
        if not self.host.play_music(name):
            self.note_missing('PlayMusic')

    def future(self, obj, ticks, axis):
        """
        GetFutureX / GetFutureY (0xbf, 0xc0): the place of the object after the ticks, 0 for NULL.
        """
        if obj is None:
            return 0.0
        if hasattr(obj, 'future_position'):
            return float(obj.future_position(ticks)[axis])
        return self.coordinate(obj, axis)

    def cast_spell(self, name, text, origin, obj, point):
        """
        CastSpellPoint( spell, x0, y0, x1, y1 ) (0xa8) and CastSpellUnit( spell, x0, y0, o ) (0xa9): the spell
        "name{modifier;...}" (688ff0) goes from the point x0, y0 without a caster (CWorld::SpreadSpell).
        """
        if obj is None and point is None:
            return
        if not self.host.cast_spell(text, origin, obj, point):
            self.note_missing('%s %s' % (name, text))

    def create_particles(self, nid, x, y, z, size, name):
        if not self.sources.create_particles(nid, (x, y, z), size, name):
            self.note_missing('CreateParticleSource %s' % name)

    def host_command(self, name, method, *args):
        if args[0] is None:
            return
        if not method(*args):
            self.note_missing(name)

    def console(self, value):
        if _Debug:
            print('script console: %r' % (value, ))

#------------------------------------------------------------------------------


def lever_of(obj):
    """
    The engine.lever.Lever of an object of the scripts, None when it is not a lever.
    """
    if isinstance(obj, lever_module.Lever):
        return obj
    lever = getattr(obj, 'lever', None)
    return lever if isinstance(lever, lever_module.Lever) else None


def is_unit(obj):
    return obj is not None and hasattr(obj, 'brain') and hasattr(obj, 'go_to')


def set_science(obj, key, level, *flags):
    """
    SetScience( o, key, level, f1, f2, f4, f8 ) (0xac, 52bb20): the science of a lever (+0x244): the flags 1, 2, 4, 8
    of the arguments which are not 0, the number of the key and the level; all 0 leave the lever to the scripts.
    """
    lever = lever_of(obj)
    if lever is None:
        return
    bits = 0
    for i, f in enumerate(flags[:4]):
        if f != 0.0:
            bits |= 1 << i
    lever.science = (bits, int(key), int(level))


def activate_trap(obj, on):
    """
    ActivateTrap( o, f ) (0x95): the flag 0x80000 of the object, a magic trap casts only with it (engine.trap).
    """
    if isinstance(obj, trap_module.MagicTrap):
        obj.active = on != 0.0


def inflict_damage(unit, damage):
    """
    InflictDamage( o, n ) (0x44): a blow packet (the type 5, the roll n, the torso, the attack 100) goes to vf20 of
    the unit (553f40) without an attacker: the roll of the attack (553df0) is then FLT_MAX, the blow always hits
    and the unit does not answer it.
    """
    parts = [0.0] * combat.DAMAGE_TYPES_COUNT
    parts[INFLICT_DAMAGE_TYPE] = 1.0
    hit = combat.Hit(parts, float(damage), location=combat.TORSO, backstab=0.0, attack=combat.SURE_ATTACK)
    unit.take_spell_blow(hit, None, None)


def kill_unit(unit):
    """
    KillUnit (0x35): every part of the body is lost and the unit dies.
    """
    fighter = unit.fighter
    if fighter is not None:
        for part in fighter.parts:
            part.health = -1.0
    unit.die()


def _default(kind):
    return DEFAULTS.get(kind)


def _bool(value):
    return 1.0 if value else 0.0


def _group_add(a, b):
    result = list(a)
    for obj in b:
        if obj is not None and obj not in result:
            result.append(obj)
    return tuple(result)
