"""
The units of a map fight back: whom they hate, whom they see, when they attack, call for help and go home.

game.exe keeps a logic object for every unit of the computer (5d6630 builds it from UNIT_LOGIC of the .mob)
with a list of behaviours, each one gives a weight every time the logic thinks and the heaviest one runs;
the attack (5d2850, vf04 5d2960) weighs 15000 when it has a target, the guard of the point (73d128) and the others
less. Only the parts of it which make a unit fight back are taken here, in a simpler form:

The diplomacy (DIPLOMATION_FOF of the .mob, 32 x 32 dwords, 481f00 copies them): 0 for the same side,
1 for the neutral, 2 for the war; the player of the remake (the party) is the player 0, the units of a map
carry theirs in OBJ_PLAYER. On zone1 the player 0 is at war with the monsters (3, 4, 6, 7, 8), neutral
to the people (2) and the pigs (5).

The senses (5ca5a0 (partly), UNIT_STATS 29..34 and 35..40, prototype tags 25 and 26). A unit of the computer sees
an enemy when the distance is not more than its sight times the detection of the other, the light (+0x904) and
the visibility (+0x900), the other stands inside the angle of the sight (UNIT_STATS 10, the cosine of its half) and
the ground does not hide it (5bc3e0 (partly): game.exe also dims the sight by the objects on the way). An enemy
ahead and nearer than the peripheral radius (UNIT_STATS 11) is seen too, and a living one nearer than the life
sense times its detection, anywhere around. 5caa00 (partly): an enemy which was seen once stays in the list of
the logic (+0x114) until it dies, only the units of a player lose the others from the sight. 5cb850 counts how
near a unit of a player is to be noticed by the sight, the life sense, the smell and the hearing, for its interface.

The threat (5cbbe0): the logic remembers one place with a weight, which fades by a step every tick (5cb700):
1000 and -2 for the place of the one who hit the unit (5d1e90) or of the target which got away alive (5d76a0),
900 and -3 for a sound (5cc9d0 (partly): the remake takes the noise of a moving enemy the unit hears, game.exe
a list of delayed sounds). The death of the target clears it (5d2710). The behaviour of the suspicion (73d1f8)
weighs as the threat (5d1e90) against 50 of the guard and of the patrol, the running behaviour gets 100 more
(5d53b0): the unit raises the signal 5, goes to the place and looks around there (5d21a0, 5d1f80 (partly)), then
the signal 3 (5d2310).

The target (5cb670): the nearest living enemy of the list of the logic. A unit which is hit remembers the attacker
(5d2960 takes it at +0x30 of the logic with the priority 1000) and calls for help (5d5990): every unit within
the radius of help whose player is in the mask of the friends (7c2bb4) of the caller gets the message 0x1a.

The record UNIT_LOGIC (5d6630):
- UNIT_LOGIC_MODEL: 1 guards UNIT_LOGIC_GUARD_PT (4ba1c0), 2 patrols by the points GUARD_PT (5d0f60), 3 and 5
  have other behaviours (4ba360, a guard of a point of 7affe4); a unit of the model 0 or 4 gets no behaviour
  of the fight at all, it does not fight back;
- UNIT_LOGIC_GUARD_R, the radius of the guard (below 0: GuardRadius of [Logic] of ai.reg, 5d0b70);
- UNIT_LOGIC_HELP, the radius of the call for help (when it equals CallForHelpRadiusDefault,
  CallForHelpRadius is taken);
- UNIT_LOGIC_AGRESSION_MODE: 0 gives the attack (5d2850) and three more behaviours (4ba710, 4ba610, 4ba920),
  1, 2 and 3 other sets of them.

The patrol (model 2, behaviour 5d0f60, vtable 73ff40): 5d6630 puts every GUARD_PT at the middle of its cell
of the grid (0.5 map units) and turns every ACTION_PT of it into (the angle to ACTION_PT_LOOK_PT, the speed of
the turn, the wait, a flag); when UNIT_LOGIC_CYCLIC is 0 the points are added once more backwards, so the unit
goes there and back. The patrol weighs 50 (5d16b0), the attack outweighs it; without points it turns into
the guard of the place of the unit. When the logic is free, 5d1700 takes the next point (after the last one
the first) and walks to it; at the point (nearer than 1, 5d7520) every action turns the unit to its angle
(5546b0) and then waits ACTION_PT_WAIT_SEG ticks of the server (7affb4), then the logic is free again.

A shooter (54b6c0: a bow or a crossbow, or UNIT_STATS 15 above 1 without a weapon item) whose health is above
0.6 runs from the nearest enemy nearer than RangedAttackDist (RangedRunDist while it runs) to a point 5 farther
from it, moved by up to 3 at random, when the path there is found (5d2960 weighs it 15000).

The alarms (593fc0, 5d5940): a unit of a map has 5 records UNIT_LOGIC, one for each alarm 0..4 of the map.
The condition UNIT_LOGIC_ALARM_CONDITION arms the alarm of a unit: 0 when it sees an enemy, 1 when it is hit,
2 when it sees an enemy or begins to fight (5d53b0, 5d7e90); after the delay +0x44 of the record (not kept
in the .mob, so the next thought) a unit whose UNIT_LOGIC_NALARM is not 0 raises that alarm at its place.
An alarm is raised once (the script InvokeAlarm of 4b6760 can force it, the alarm 0 forced clears them all);
then every unit of the map, of any side, takes its record of that alarm when UNIT_LOGIC_USE is set in it
(always for the alarm 0) and says "to Alarm!" (the signal 7 of 5aea80, the voices are not here).
The scripts read the alarms by IsAlarm, AlarmTime, AlarmPosX and AlarmPosY.

The remake's choices, not read from game.exe yet: the logic thinks every THINK_TICKS ticks; a unit gives up
a target which is farther from its guard point than the guard radius plus its sight (it has to see the target again
to attack it) and, after the suspicion, walks back to the point, as it does when it stands farther than the guard
radius from it (a patrol goes to its next point instead);
the threats of UMFear( o, 1 ) are not here.

What the player sees (668c10, 5caa00 (partly), 5ca5a0 (partly)): every unit of the party keeps the units it saw
(+0xa94 of its server, the same list of its logic +0x980 + 0x114) with the rules of the sight above, but it looks
at all the units, not only at the enemies, and loses a living one again when the other is farther than its sight
and its life sense, unless the other attacks it (5caa00 with the factor 1). The server sends the union of the lists
to the client of the player (+0x50d8), which marks those units (0x400000 of +8); in the single game (79be5c) the
client draws (56a990), sounds (534b60), shows on the minimap (643d80) and gives the flying numbers (5117f0) only
of the marked units. The remake keeps the dead units seen (their figures stay on the map).

The behaviours of the scripts (engine.script): UMAggression and UMRevenge (73d260 with +0xc 0 or 1) set attacks
and revenge, UMFear (5d2850) sets wary, UMFollow (73d1c4) the companion; see Controller.aggression and the others.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import random

#------------------------------------------------------------------------------

from engine import magic
from engine import signals
from engine import unit as unit_module

#------------------------------------------------------------------------------

PLAYERS_COUNT = 32

SAME_SIDE = 0
NEUTRAL = 1
WAR = 2

# the player of the party
PARTY_PLAYER = 0

SIGHT = 0
# the senses of stats.UnitStats (5cb850): the hearing of the moving units
LIFE = 2
HEARING = 3

# 5bc3e0 (partly): the step of the ray of the sight, a cell of the grid of the server
SIGHT_STEP = 0.5

# 5d1e90, 5d76a0: the weight of the place of an enemy which hit the unit or got away, and its step a tick
THREAT_ENEMY = 1000
THREAT_ENEMY_STEP = -2
# 5cc9d0: of a sound
THREAT_SOUND = 900
THREAT_SOUND_STEP = -3
# 5d0a80, 5d16b0: the weight of the guard and of the patrol; 5d53b0 adds 100 to the behaviour which runs
GUARD_WEIGHT = 50
RUNNING_WEIGHT = 100
# 5d1f80 (partly): the suspicion loses 5 a tick while the unit walks farther than 5 from the place or finds no way
SUSPICION_FAR = 5.0
SUSPICION_FAR_STEP = 5

LOGIC_SECTION = 'Logic'

# 5d6630: the models of UNIT_LOGIC without the behaviours of the fight
PEACEFUL_MODELS = (0, 4)
GUARD_MODEL = 1
PATROL_MODEL = 2

# UNIT_LOGIC_ALARM_CONDITION (+0x40 of the record, +0x3c of the logic): 0 arms the alarm when the unit sees
# an enemy (5d53b0), 1 when it is hit (5d7e90), 2 when it sees an enemy or begins to fight
ALARM_ON_SIGHT = 0
ALARM_ON_HIT = 1
ALARM_ON_FIGHT = 2

# 593fc0: the alarms 0..4 of a map
ALARMS_COUNT = 5

# 5d6630: the points of a patrol are put at the middle of their cells
PATROL_CELL = 0.5
# 73fef0: the square of the distance at which a patrol has come to its point
PATROL_ARRIVE_DISTANCE = 1.0

# [Logic] of ai.reg as game.exe has them when the registry gives none (5ce0f0 keeps the old values)
GUARD_RADIUS = 10.0
CALL_FOR_HELP_RADIUS = 7.0
CALL_FOR_HELP_RADIUS_DEFAULT = 7.0
RANGED_ATTACK_DIST = 7.0
RANGED_RUN_DIST = 4.0
# PlayerCallForHelp of [Logic] (7c540c): a unit of the player helps its fighting friends this near (5d3f50)
PLAYER_CALL_FOR_HELP = 16.0

# 5d2960: a shooter runs only while its health is above this part (73cde0), 5 away (73bf34) from the enemy,
# the point moved by up to 3 (rand32 * 73ffc8 - 73baa4) along both axes
RANGED_HEALTH_PART = 0.6
RANGED_RUN_STEP = 5.0
RANGED_JITTER = 3.0

THINK_TICKS = 5

# GuardStayTime of [Logic]: the stay of a guard at a point it walked to, in the units of 5d5b40
GUARD_STAY_TIME = 10.0
# 5d0be0: rand() % 1001 == 1 raises the idle signal 1
GUARD_IDLE_SIGNAL_CHANCE = 1001
# 5d0be0 -> 5d7f80: the walk of a free guard to a random point of its circle
WANDER_TRIES = 10
# 73c54c: a point whose ground is steeper (the biggest difference of the heights of the cells of 0.5 around it,
# 5bc2a0) is not taken
WANDER_MAX_SLOPE = 0.3
WANDER_SLOPE_STEP = 0.5
# 73ce18: no point within this many radii of the unit from a friend
WANDER_FRIEND_RADII = 2.5
# 5d7f80: a point without a path makes the logic wait min(10, distance / 3) calls (73d9a4)
WANDER_PAUSE_PER_DISTANCE = 1.0 / 3.0
WANDER_MAX_PAUSE = 10
# 5d0be0: after this many points not found in its circle the guard looks around itself
WANDER_FAILS = 100
# 5d5b40: 1..4 looks at the point, a look waits stay * 15 * (0.25..1) ticks, 40 without the stay;
# one time of 11 an animal (not "unhu...") has one long look
LOOK_TICKS_PER_STAY = 15.0
LOOK_DEFAULT_TICKS = 40.0
LOOK_MIN_PART = 0.25
LOOK_LONG_CHANCE = 11
LOOK_LONG_SKIP_PREFIX = 'unhu'

# the state +4 of the logic: free, a move or a turn of a script (5d7ae0, 5d7b30), the follow (5d7e40)
ORDER_NONE = 0
ORDER_MOVE = 1
ORDER_FOLLOW = 6

# 5d1b50 / 5d1b70: the sentry walks to its point when the logic is free; the remake's arrive distance
SENTRY_RADIUS = 1.0

# the remake's choice: a follower keeps this far from its leader
FOLLOW_DISTANCE = 2.0

# a turn is over when the yaw is this near to the target yaw
TURN_DONE = 0.05

#------------------------------------------------------------------------------


class Diplomacy(object):

    def __init__(self, relations=None):
        """
        relations: PLAYERS_COUNT * PLAYERS_COUNT values by rows, None for the war of the party with everybody else.
        """
        self.relations = list(relations) if relations else None

    @classmethod
    def from_mob(cls, mob_root):
        if mob_root is not None:
            for node in mob_root.walk('DIPLOMATION_FOF'):
                if node.value:
                    return cls(node.value)
        return cls()

    def relation(self, player, other):
        """
        CDiplomacy::Get (455630): 0 the same side, 1 neutral, 2 war.
        """
        if player == other:
            return SAME_SIDE
        if self.relations is None:
            return WAR if PARTY_PLAYER in (player, other) else SAME_SIDE
        if not (0 <= player < PLAYERS_COUNT and 0 <= other < PLAYERS_COUNT):
            return NEUTRAL
        return self.relations[player * PLAYERS_COUNT + other]

    def set_relation(self, player, other, relation):
        """
        SetDiplomacy (4554e0): the relation of the player to the other one, the other way is not changed.
        """
        if not (0 <= player < PLAYERS_COUNT and 0 <= other < PLAYERS_COUNT):
            return
        if self.relations is None:
            self.relations = [self.relation(p, o) for p in range(PLAYERS_COUNT) for o in range(PLAYERS_COUNT)]
        self.relations[player * PLAYERS_COUNT + other] = relation

    def at_war(self, player, other):
        return self.relation(player, other) == WAR

    def friends(self, player, other):
        return self.relation(player, other) == SAME_SIDE


class Settings(object):
    """
    The section [Logic] of ai.reg (read by 5ce0f0 into 7c53ec..7c5418).
    """

    def __init__(self, guard_radius=GUARD_RADIUS, call_for_help_radius=CALL_FOR_HELP_RADIUS,
                 call_for_help_radius_default=CALL_FOR_HELP_RADIUS_DEFAULT, ranged_attack_dist=RANGED_ATTACK_DIST,
                 ranged_run_dist=RANGED_RUN_DIST, guard_stay_time=GUARD_STAY_TIME,
                 player_call_for_help=PLAYER_CALL_FOR_HELP):
        self.player_call_for_help = player_call_for_help
        self.guard_radius = guard_radius
        self.guard_stay_time = guard_stay_time
        self.call_for_help_radius = call_for_help_radius
        self.call_for_help_radius_default = call_for_help_radius_default
        self.ranged_attack_dist = ranged_attack_dist
        self.ranged_run_dist = ranged_run_dist

    @classmethod
    def from_registry(cls, registry):
        logic = (registry or {}).get(LOGIC_SECTION) or {}
        return cls(guard_radius=float(logic.get('GuardRadius', GUARD_RADIUS)),
                   call_for_help_radius=float(logic.get('CallForHelpRadius', CALL_FOR_HELP_RADIUS)),
                   call_for_help_radius_default=float(logic.get('CallForHelpRadiusDefault',
                                                                CALL_FOR_HELP_RADIUS_DEFAULT)),
                   ranged_attack_dist=float(logic.get('RangedAttackDist', RANGED_ATTACK_DIST)),
                   ranged_run_dist=float(logic.get('RangedRunDist', RANGED_RUN_DIST)),
                   guard_stay_time=float(logic.get('GuardStayTime', GUARD_STAY_TIME)),
                   player_call_for_help=float(logic.get('PlayerCallForHelp', PLAYER_CALL_FOR_HELP)))


class UnitLogic(object):
    """
    The record UNIT_LOGIC of a unit of a map.
    """

    def __init__(self, model=1, guard_point=None, guard_radius=-1.0, help_radius=None, aggression_mode=0,
                 patrol_points=(), cyclic=True, alarm=0, alarm_condition=ALARM_ON_SIGHT, use=True):
        """
        patrol_points: (x, y, actions) of the points GUARD_PT, actions are (yaw, wait ticks) of the ACTION_PT.
        alarm: UNIT_LOGIC_NALARM, the alarm which the unit raises (0 for none), alarm_condition:
        UNIT_LOGIC_ALARM_CONDITION; use: UNIT_LOGIC_USE, the record is taken when its alarm is raised.
        """
        self.alarm = alarm
        self.alarm_condition = alarm_condition
        self.use = use
        self.model = model
        self.guard_point = tuple(guard_point[:2]) if guard_point else None
        self.guard_radius = guard_radius
        self.help_radius = help_radius
        self.aggression_mode = aggression_mode
        self.patrol_points = list(patrol_points)
        self.cyclic = cyclic

    def patrol_route(self):
        """
        5d6630: the points of the patrol in the order of the walk, backwards after them when it is not cyclic.
        """
        if self.model != PATROL_MODEL:
            return []
        route = list(self.patrol_points)
        if not self.cyclic:
            route += route[::-1]
        return route

    def __repr__(self):
        return 'UnitLogic(model %d, guard %s r %.1f, help %s)' % (self.model, self.guard_point, self.guard_radius,
                                                                 self.help_radius)

    @classmethod
    def from_node(cls, node):
        if node is None or node.children is None:
            return None
        points = []
        for point in node.children:
            if point.name != 'GUARD_PT' or point.children is None:
                continue
            position = point.get('GUARD_PT_POSITION')
            if not position:
                continue
            x, y = cell_middle(position[0]), cell_middle(position[1])
            actions = []
            for action in point.children:
                if action.name != 'ACTION_PT' or action.children is None:
                    continue
                look = action.get('ACTION_PT_LOOK_PT') or (x, y, 0.0)
                actions.append((look_yaw(look[0] - x, look[1] - y), action.get('ACTION_PT_WAIT_SEG', 0) or 0))
            points.append((x, y, actions))
        return cls(model=node.get('UNIT_LOGIC_MODEL', 1) or 0, guard_point=node.get('UNIT_LOGIC_GUARD_PT'),
                   guard_radius=node.get('UNIT_LOGIC_GUARD_R', -1.0), help_radius=node.get('UNIT_LOGIC_HELP'),
                   aggression_mode=node.get('UNIT_LOGIC_AGRESSION_MODE', 0) or 0,
                   patrol_points=points, cyclic=bool(node.get('UNIT_LOGIC_CYCLIC', 1)),
                   alarm=node.get('UNIT_LOGIC_NALARM', 0) or 0,
                   alarm_condition=node.get('UNIT_LOGIC_ALARM_CONDITION', ALARM_ON_SIGHT) or 0,
                   use=bool(node.get('UNIT_LOGIC_USE', 0)))


def eye_point(unit):
    """
    The point a unit looks from and is looked at (vf 0xc0 of the unit; the remake takes the top of its figure).
    """
    return unit.x, unit.y, unit.z + unit.height()


def line_of_sight(terrain, start, end):
    """
    5bc3e0 (partly): 1 when the ground does not rise to the ray between the points (x, y, z), else 0; the ray is
    checked every cell of the grid. game.exe also multiplies by the clearness of the objects on the way.
    """
    if terrain is None:
        return 1.0
    dx, dy, dz = end[0] - start[0], end[1] - start[1], end[2] - start[2]
    steps = int(round(math.sqrt(dx * dx + dy * dy + dz * dz) / SIGHT_STEP))
    for step in range(1, steps + 1):
        t = step / float(steps)
        if start[2] + dz * t <= terrain.height_at(start[0] + dx * t, start[1] + dy * t):
            return 0.0
    return 1.0


def sees(unit, other):
    """
    5ca5a0 (partly): the sight inside the angle of the unit with a clear line to the other (the reach is the sight
    times the detection of the other and its visibility, 5536b0 (partly): the crawl and the sneak; the stealth skill
    of a hero (Unit.detection_factor) lowers it), or the peripheral vision ahead, or the life sense for a living one.
    """
    distance = unit.distance_to(other)
    reach = unit.senses[SIGHT] * other.detections[SIGHT] * other.visibility() * other.detection_factor()
    yaw = getattr(unit, 'yaw', None)
    if yaw is None:
        ahead = distance
    else:
        ahead = (other.x - unit.x) * math.sin(yaw) - (other.y - unit.y) * math.cos(yaw)
    if distance <= reach and ahead >= getattr(unit, 'view_cos', -1.0) * distance:
        terrain = getattr(unit, 'terrain', None)
        if terrain is None or line_of_sight(terrain, eye_point(unit), eye_point(other)) * reach > distance:
            return True
    if ahead >= 0.0 and distance < getattr(unit, 'peripheral', 0.0):
        return True
    if getattr(other, 'is_dead', False):
        return False
    return distance < unit.senses[LIFE] * other.detections[LIFE]


def loses_sight(unit, other):
    """
    5caa00 (partly) for a unit of a player: a living unit seen once is lost when it is not nearer than the reach of
    the sight (no angle and no line here) and of the life sense, unless it attacks the unit.
    """
    if getattr(other, 'is_dead', False):
        return False
    if getattr(other, 'foe', None) is unit and getattr(other, 'mode', None) == unit_module.MODE_ATTACK:
        return False
    distance = unit.distance_to(other)
    reach = unit.senses[SIGHT] * other.detections[SIGHT] * other.visibility() * other.detection_factor()
    return reach <= distance and unit.senses[LIFE] * other.detections[LIFE] <= distance


def hears(unit, other):
    """
    The hearing of 5cb850 (halved while the unit runs) times the noise the other makes (5536b0 (partly)) and its
    detection by the hearing; the stealth skill of a hero lowers it.
    """
    noise = other.hearing_noise() * other.detection_factor() * unit.hearing_factor()
    return noise > 0.0 and unit.distance_to(other) < unit.senses[HEARING] * other.detections[HEARING] * noise


def notices(unit, other):
    """
    A unit of a player notices the other by the sight (sees) or by the hearing (hears).
    """
    return sees(unit, other) or hears(unit, other)


class Brain(object):
    """
    The logic of one unit of the computer.
    """

    def __init__(self, unit, player, logic=None, settings=None, phase=0, logics=None):
        """
        logics: the records UNIT_LOGIC of the alarms 0..4 (logic is the one of the alarm 0 when it is not given).
        """
        self.unit = unit
        self.player = player
        self.settings = settings or Settings()
        self.logics = list(logics or [logic or UnitLogic()])
        self.target = None
        # +0x114 of the logic: the enemies which the unit saw, kept until they die (5caa00 (partly))
        self.seen = []
        # +0xe0..+0xec of the logic (5cbbe0): the place of the threat, its weight and the step of the weight a tick
        self.threat_point = None
        self.threat_weight = 0
        self.threat_step = 0
        # the behaviour of the suspicion (73d1f8): it runs, its place (+8) and its weight (+0x14)
        self.suspicious = False
        self.suspicion_point = None
        self.suspicion_weight = 0
        self.signal = signals.Signal()
        self.ticks = float(phase % THINK_TICKS)
        self.set_logic(logic or self.logics[0])

    def set_logic(self, logic):
        """
        5d6630: the unit takes the record of the logic, the alarm is not armed.
        """
        settings = self.settings
        unit = self.unit
        self.logic = logic
        self.home = logic.guard_point or (unit.x, unit.y)
        # 5d0b70
        self.guard_radius = logic.guard_radius if logic.guard_radius >= 0 else settings.guard_radius
        # 5d6630
        help_radius = settings.call_for_help_radius_default if logic.help_radius is None else logic.help_radius
        if help_radius == settings.call_for_help_radius_default:
            help_radius = settings.call_for_help_radius
        self.help_radius = help_radius
        self.route = logic.patrol_route()
        self.route_index = -1
        # the actions of the point of the patrol which are left, None while the unit walks to the point
        self.actions = None
        # the tick of the server until which the unit waits at the point of the patrol
        self.wait_until = None
        # a shooter which runs from an enemy
        self.fleeing = False
        # 5d0be0: +0x1c of the guard, the points not found in a row, and +0x68 of the logic, the calls to wait
        self.wander_fails = 0
        self.wander_pause = 0
        # +0x34 of the logic: the condition of the alarm came, the unit raises its alarm at the next thought
        self.alarm_armed = False
        self.alarm_raised = False
        # UMPlayer: the behaviour of a unit of the player (73f0f8), it looks for enemies only in danger
        self.player_mode = False
        # the behaviours of the logic: the aggression (73d260, it looks for enemies and attacks them), the caution
        # of 5d2850 (a shooter keeps away) and the guard of the home or the patrol
        self.attacks = logic.model not in PEACEFUL_MODELS
        self.wary = self.attacks
        self.guards = logic.model not in PEACEFUL_MODELS
        # UMRevenge (73d260 with +0xc = 1): it attacks only the one who hit it
        self.revenge = False
        # UMFear( o, 1 ): the flag +0x28 of 5d2850, the threats of all the units near
        self.fears_all = False
        # UMFollow (73d1c4): the unit which it follows when the logic is free
        self.companion = None
        # the order of a script (state +4) and the leader of ORDER_FOLLOW
        self.order = ORDER_NONE
        self.leader = None
        # 555270: Walk (2) or Run (3) of a script, the way the unit goes by the orders
        self.runs = False

    def arm_alarm(self, event):
        """
        5d53b0 and 5d7e90: the event (ALARM_ON_SIGHT, ALARM_ON_HIT or ALARM_ON_FIGHT) which arms the alarm
        by UNIT_LOGIC_ALARM_CONDITION.
        """
        condition = self.logic.alarm_condition
        if event == ALARM_ON_SIGHT and condition in (ALARM_ON_SIGHT, ALARM_ON_FIGHT):
            self.alarm_armed = True
        elif event == ALARM_ON_HIT and condition == ALARM_ON_HIT:
            self.alarm_armed = True
        elif event == ALARM_ON_FIGHT and condition == ALARM_ON_FIGHT:
            self.alarm_armed = True

    @property
    def fights(self):
        return self.attacks and self.unit.fighter is not None

    @property
    def avenges(self):
        return self.revenge and self.unit.fighter is not None

    def __repr__(self):
        return 'Brain(%s, player %d, target %s)' % (self.unit, self.player, self.target)

    @property
    def sight(self):
        return self.unit.senses[SIGHT]

    def sees(self, other):
        """
        5ca5a0 (partly): the unit sees the other now.
        """
        return sees(self.unit, other)

    def remember_threat(self, point, weight, step):
        """
        CUnitLogic::RememberThreat 5cbbe0: a heavier threat replaces the one the logic keeps.
        """
        if weight <= self.threat_weight:
            return False
        self.threat_point = (float(point[0]), float(point[1]))
        self.threat_weight = weight
        self.threat_step = step
        return True

    def forget_threat(self):
        """
        5d2710: the target is dead, no threat is left.
        """
        self.threat_weight = 0
        self.threat_step = 0

    def home_distance(self, other=None):
        other = other or self.unit
        return math.hypot(other.x - self.home[0], other.y - self.home[1])

    def too_far(self, target):
        return self.home_distance(target) > self.guard_radius + self.sight

#------------------------------------------------------------------------------


class Controller(object):
    """
    The logic of all the units of the computer on a map. The other units (the party) are only known by their player.
    """

    def __init__(self, diplomacy=None, settings=None, rng=None):
        self.diplomacy = diplomacy or Diplomacy()
        self.settings = settings or Settings()
        self.rng = rng or random.Random()
        self.brains = []
        # unit -> player for the units without a brain
        self.players = {}
        # the ticks of the server since the start (7affb4)
        self.ticks = 0
        # 593fc0: alarm -> (tick, position) of the alarms raised on the map
        self.alarms = {}
        # +0xa94: a unit of the party -> the units it saw; +0x50d8: the units the player sees, None when the player
        # has no unit (nothing is hidden then)
        self.party_seen = {}
        self.visible = None
        # the effects of the spells on the map (the client keeps them), for the sight of the player
        self.effects = []

    def add(self, unit, player, logic=None, logics=None):
        brain = Brain(unit, player, logic, self.settings, phase=len(self.brains), logics=logics)
        self.brains.append(brain)
        unit.brain = brain
        unit.on_attacked = self.attacked
        unit.hostile_to = self.hostile
        return brain

    def add_player_unit(self, unit, player=PARTY_PLAYER):
        self.players[unit] = player
        unit.on_attacked = self.attacked
        unit.hostile_to = self.hostile

    def hostile(self, unit, other):
        """
        Whether the other unit is an enemy of the unit by the diplomacy of the map (the filters of the spells);
        a charmed unit is nobody's enemy.
        """
        if magic.is_charmed(getattr(unit, 'fighter', None)) or magic.is_charmed(getattr(other, 'fighter', None)):
            return False
        return self.diplomacy.at_war(self.player_of(unit), self.player_of(other))

    def player_of(self, unit):
        if unit.brain is not None:
            return unit.brain.player
        return self.players.get(unit, PARTY_PLAYER)

    def units(self):
        return [b.unit for b in self.brains] + list(self.players)

    def seen_by(self, unit):
        """
        UnitSee: the living units which the unit sees (the list +0xa94 of its logic).
        """
        return [u for u in self.units() if u is not unit and not u.is_dead and sees(unit, u)]

    def in_danger(self, player):
        """
        IsPlayerInDanger: the state +0x5120 of the player is 2 when a unit is after one of its units (66e3f0);
        the remake takes a living unit at war with the player whose target is a unit of the player.
        """
        for brain in self.brains:
            target = brain.target
            if brain.unit.is_dead or target is None or target.is_dead:
                continue
            if self.player_of(target) == player and self.diplomacy.at_war(brain.player, player):
                return True
        return False

    def enemies(self, brain):
        return [u for u in self.units() if u is not brain.unit and not u.is_dead and u.fighter is not None
                and self.hostile(brain.unit, u)]

    def look(self, brain):
        """
        5caa00 (partly) and 5ca5a0 (partly): the dead leave the list of the seen enemies, the enemies which
        the unit sees now come into it; the place of an enemy which is only heard becomes a threat
        (5cc9d0 (partly)).
        """
        u = brain.unit
        enemies = self.enemies(brain)
        brain.seen = [other for other in brain.seen if other in enemies]
        for other in enemies:
            if other in brain.seen:
                continue
            if sees(u, other):
                brain.seen.append(other)
            elif hears(u, other):
                brain.remember_threat((other.x, other.y), THREAT_SOUND, THREAT_SOUND_STEP)

    def nearest_enemy(self, brain):
        """
        5cb670: the nearest living enemy of those the unit saw; seeing one arms the alarm (5d53b0).
        """
        self.look(brain)
        best, best_distance = None, None
        for other in brain.seen:
            brain.arm_alarm(ALARM_ON_SIGHT)
            distance = brain.unit.distance_to(other)
            if best is None or distance < best_distance:
                best, best_distance = other, distance
        return best

    def attacked(self, target, attacker):
        """
        unit.Unit.on_attacked: the target takes the attacker for its foe and calls its friends for help (5d5990).
        """
        brain = target.brain
        if brain is None or target.is_dead or attacker is None or attacker.is_dead:
            return
        brain.arm_alarm(ALARM_ON_HIT)
        if not self.diplomacy.friends(brain.player, self.player_of(attacker)):
            # 5d1e90: the place of the one who hit the unit
            brain.remember_threat((attacker.x, attacker.y), THREAT_ENEMY, THREAT_ENEMY_STEP)
        if (brain.fights or brain.avenges) and (brain.target is None or brain.target.is_dead):
            self.engage(brain, attacker)
        for other in self.brains:
            if other is brain or other.unit.is_dead or not (other.fights or other.avenges):
                continue
            if not self.diplomacy.friends(brain.player, other.player):
                continue
            if target.distance_to(other.unit) > brain.help_radius:
                continue
            if other.target is None or other.target.is_dead:
                if _Debug:
                    print('%s calls %s for help against %s' % (target, other.unit, attacker))
                self.engage(other, attacker)

    def raise_alarm(self, alarm, position, force=False):
        """
        593fc0: the alarm 0..4 of the map, once unless forced (the alarm 0 forced clears all of them);
        every unit of the map takes its record of the logic of the alarm when the record is in use (5d5940).
        """
        if not 0 <= alarm < ALARMS_COUNT:
            return False
        if alarm in self.alarms and not force:
            return False
        if alarm == 0 and force:
            self.alarms.clear()
        self.alarms[alarm] = (self.ticks, tuple(position[:2]))
        for brain in self.brains:
            brain.signal.raise_level(signals.ALARM if alarm == 0 else signals.TO_ALARM)
        if _Debug:
            print('alarm %d at %.1f,%.1f' % (alarm, position[0], position[1]))
        for brain in self.brains:
            if alarm < len(brain.logics) and (brain.logics[alarm].use or alarm == 0):
                brain.set_logic(brain.logics[alarm])
        return True

    def engage(self, brain, foe):
        brain.arm_alarm(ALARM_ON_FIGHT)
        self.calm(brain)
        brain.target = foe
        if brain.unit.foe is not foe or brain.unit.mode != unit_module.MODE_ATTACK:
            brain.unit.attack(foe)

    def update(self, ticks):
        self.ticks += ticks
        for brain in self.brains:
            started = brain.signal.update(ticks)
            if started and brain.unit.on_signal is not None:
                brain.unit.on_signal(brain.unit, started)
            brain.ticks -= ticks
            if brain.ticks > 0:
                continue
            brain.ticks += THINK_TICKS
            self.think(brain)
        if self.ticks // THINK_TICKS != (self.ticks - ticks) // THINK_TICKS:
            for unit in list(self.players):
                self.player_fight(unit)
            self.update_player_sight()

    def update_player_sight(self, player=PARTY_PLAYER):
        """
        668c10, 5caa00 (partly), 5ca5a0 (partly): the lists of the units which the units of the party saw, and their
        union with the party, the units the player sees (visible). The spells of the party add to the visible
        (682610): the clairvoyance the units in its circle, the possession its target and what the target sees.
        """
        party = [u for u, p in self.players.items() if p == player and not u.is_dead]
        if not party:
            self.party_seen = {}
            self.visible = None
            return
        others = [u for u in self.units() if self.players.get(u) != player]
        visible = set(party)
        party_seen = {}
        for u in party:
            seen = [o for o in self.party_seen.get(u, ()) if o in others and not o.is_dead and not loses_sight(u, o)]
            for other in others:
                if other not in seen and not other.is_dead and sees(u, other):
                    seen.append(other)
            party_seen[u] = seen
            visible.update(seen)
        visible.update(self.spell_sight(party, others))
        self.party_seen = party_seen
        self.visible = visible

    def spell_sight(self, party, others):
        """
        682610 cases 0x14, 0x15: the units the player sees through the spells (engine.magic.SpellParticles,
        engine.magic.Possession) which the units of the party cast.
        """
        result = set()
        for effect in self.effects:
            if getattr(effect, 'caster', None) not in party:
                continue
            sight_circle = getattr(effect, 'sight_circle', None)
            circle = sight_circle() if sight_circle is not None else None
            if circle is not None:
                (x, y, _), radius = circle
                result.update(o for o in others if not o.is_dead and math.hypot(o.x - x, o.y - y) <= radius)
            sight_eye = getattr(effect, 'sight_eye', None)
            eye = sight_eye() if sight_eye is not None else None
            if eye is not None:
                result.add(eye)
                result.update(o for o in others if o is not eye and not o.is_dead and sees(eye, o))
        return result

    def is_visible(self, unit):
        """
        The flag 0x400000 of the client: the player sees the unit; the dead are always seen in the remake.
        """
        return self.visible is None or unit.is_dead or unit in self.visible

    def player_fight(self, u):
        """
        5d4170 (partly): a free unit of the player with the button "battle" on (+0x438) keeps its living foe, else
        attacks the nearest enemy which it notices or which a fighting friend of its player within
        PlayerCallForHelp attacks (5d3f50). game.exe weighs the targets by 5d0030 and limits them to SwarmRadius
        around the party in the swarm; the remake takes the nearest one.
        """
        if u.is_dead or u.is_frozen or u.fighter is None or not getattr(u, 'auto_fight', False):
            return
        if u.mode == unit_module.MODE_ATTACK and u.foe is not None and not u.foe.is_dead:
            return
        if u.mode != unit_module.MODE_IDLE:
            return
        player = self.player_of(u)
        candidates = []
        for other in self.units():
            if other is u or other.is_dead or other.fighter is None:
                continue
            if self.hostile(u, other) and notices(u, other):
                candidates.append(other)
            elif self.player_of(other) == player and other.mode == unit_module.MODE_ATTACK and \
                    u.distance_to(other) <= self.settings.player_call_for_help:
                foe = other.foe
                if foe is not None and not foe.is_dead and foe.fighter is not None and \
                        self.player_of(foe) != player and self.hostile(u, foe):
                    candidates.append(foe)
        if not candidates:
            return
        target = min(candidates, key=u.distance_to)
        if _Debug:
            print('%s fights %s on its own' % (u, target))
        u.attack(target)

    def think(self, brain):
        u = brain.unit
        if u.is_dead:
            return
        # 5499f0: a frozen unit (stun) does nothing
        if u.is_frozen:
            return
        # 5cb700: the threat fades every tick of the server
        brain.threat_weight += brain.threat_step * THINK_TICKS
        # 5d53b0: an armed alarm is raised at the next thought
        if brain.alarm_armed and not brain.alarm_raised and brain.logic.alarm:
            brain.alarm_raised = True
            self.raise_alarm(brain.logic.alarm, (u.x, u.y))
        if brain.order == ORDER_FOLLOW:
            self.follow(brain)
            return
        if brain.order == ORDER_MOVE:
            # 5d53b0 state 1: kept while the unit goes or turns
            if is_busy(u):
                return
            brain.order = ORDER_NONE
        if not brain.fights and not brain.avenges:
            self.guard_home(brain)
            return
        target = brain.target
        if magic.is_charmed(u.fighter):
            brain.target = None
            if u.mode == unit_module.MODE_ATTACK:
                u.idle()
            self.guard_home(brain)
            return
        if target is not None and (target.is_dead or magic.is_charmed(target.fighter) or
                                   brain.too_far(target)):
            if _Debug:
                print('%s gives up %s' % (u, target))
            if target.is_dead:
                brain.forget_threat()
            else:
                # 5d76a0: the place of the target which got away
                brain.remember_threat((target.x, target.y), THREAT_ENEMY, THREAT_ENEMY_STEP)
                # the remake's choice: a target the unit gave up has to be seen again
                if target in brain.seen:
                    brain.seen.remove(target)
            brain.target = target = None
            brain.signal.raise_level(signals.BACK_FROM_AGGRESSION)
            brain.actions = None
            brain.route_index = max(-1, brain.route_index - 1)
            if u.mode == unit_module.MODE_ATTACK:
                u.idle()
        if brain.wary and brain.fights and self.keep_away(brain):
            return
        if target is None:
            if brain.fights and (not brain.player_mode or self.in_danger(brain.player)):
                target = self.nearest_enemy(brain)
                if target is not None:
                    brain.signal.raise_level(signals.TO_AGGRESSION)
                    self.engage(brain, target)
                    return
        elif u.mode != unit_module.MODE_ATTACK or u.foe is not target:
            self.engage(brain, target)
            return
        if target is not None:
            return
        if self.suspect(brain):
            return
        self.guard_home(brain)

    def suspect(self, brain):
        """
        The behaviour of the suspicion (73d1f8). 5d1e90: it takes the weight of a threat at a new place, else the
        smaller of its weight and that of the threat; it starts above the weight of the guard and that of the
        running behaviour and runs until it is below the guard without those 100 (5d53b0). 5d21a0: the signal 5,
        the unit goes to the place and looks around there. 5d1f80 (partly): the weight falls while the unit walks
        far from the place or has no way; when the looks are over, new ones begin (game.exe sends the unit to
        a point within 5 around and back at the next tick). 5d2310: the signal 3. True while it runs.
        """
        u = brain.unit
        if not brain.fights or brain.player_mode or brain.companion is not None:
            self.calm(brain)
            return False
        point = brain.threat_point
        moved = point is not None and point != brain.suspicion_point
        if moved:
            brain.suspicion_point = point
            brain.suspicion_weight = brain.threat_weight
        else:
            brain.suspicion_weight = min(brain.suspicion_weight, brain.threat_weight)
        if not brain.suspicious:
            if brain.suspicion_weight <= GUARD_WEIGHT + RUNNING_WEIGHT:
                return False
            brain.suspicious = True
            brain.signal.raise_level(signals.TO_SUSPICION)
            moved = True
            if _Debug:
                print('%s suspects %.1f,%.1f' % (u, point[0], point[1]))
        elif brain.suspicion_weight + RUNNING_WEIGHT < GUARD_WEIGHT:
            self.calm(brain)
            return False
        point = brain.suspicion_point
        if moved:
            brain.actions = self.look_actions(brain, point, guard=False)
            brain.wait_until = None
            if not u.go_to(point[0], point[1], run=brain.runs):
                brain.suspicion_weight -= SUSPICION_FAR_STEP * THINK_TICKS
            return True
        if u.mode in unit_module.MOVING_MODES:
            if math.hypot(u.x - point[0], u.y - point[1]) >= SUSPICION_FAR:
                brain.suspicion_weight -= SUSPICION_FAR_STEP * THINK_TICKS
            return True
        if u.mode not in (unit_module.MODE_IDLE, unit_module.MODE_ACTION):
            return True
        if not self.do_actions(brain):
            brain.actions = self.look_actions(brain, point, guard=False)
            brain.wait_until = None
        return True

    def calm(self, brain):
        """
        5d2310: the suspicion stops with the signal 3.
        """
        if not brain.suspicious:
            return
        brain.suspicious = False
        brain.actions = None
        brain.wait_until = None
        brain.signal.raise_level(signals.BACK_FROM_SUSPICION)
        # the next behaviour gives its own order
        if brain.unit.mode in unit_module.MOVING_MODES:
            brain.unit.idle()

    def do_actions(self, brain):
        """
        The looks of a point: the unit turns to the yaw of the first action and waits its ticks, then takes the next
        one. False when no action is left.
        """
        if not brain.actions:
            return False
        if brain.wait_until is not None:
            if self.ticks < brain.wait_until:
                return True
            brain.wait_until = None
            brain.actions.pop(0)
        if not brain.actions:
            return False
        yaw, wait = brain.actions[0]
        brain.unit.target_yaw = yaw
        brain.wait_until = self.ticks + wait
        return True

    def guard_home(self, brain):
        """
        The patrol, or the walk back to the point of the guard (the sentry of a script too);
        a unit of UMFollow goes after its leader instead.
        """
        u = brain.unit
        if brain.companion is not None:
            if brain.companion.is_dead:
                brain.companion = None
            else:
                self.go_after(brain, brain.companion)
                return
        if not brain.guards:
            return
        if brain.route:
            self.patrol(brain)
        elif u.mode == unit_module.MODE_IDLE and brain.home_distance() > brain.guard_radius:
            u.go_to(brain.home[0], brain.home[1], run=brain.runs)
        elif brain.logic.model == GUARD_MODEL:
            self.wander(brain)

    def wander(self, brain):
        """
        CBehaviourGuard 5d0be0 (partly): one call of 1001 raises the idle signal 1 (5aea60); a free guard walks to
        a random point of its circle (5d7f80 (partly)) and looks around there (5d5b40 (partly)); the looks are the actions of a point
        of a patrol.
        """
        u = brain.unit
        if self.rng.randrange(GUARD_IDLE_SIGNAL_CHANCE) == 1:
            brain.signal.raise_level(signals.IDLE_A)
        if u.mode not in (unit_module.MODE_IDLE, unit_module.MODE_ACTION):
            return
        if self.do_actions(brain):
            return
        point = self.wander_to(brain, brain.home)
        if point is None:
            brain.wander_fails += 1
            if brain.wander_fails <= WANDER_FAILS:
                return
            point = self.wander_to(brain, (u.x, u.y))
            if point is None:
                return
        brain.wander_fails = 0
        brain.actions = self.look_actions(brain, point)
        brain.wait_until = None

    def wander_to(self, brain, center):
        """
        5d7f80 (partly): a random point of the circle of the guard around center which is not steep, has no friend
        near and has a path; the unit goes there. None when there is none now. game.exe also refuses a path longer
        than the distance * 3 + 10, the remake takes any path.
        """
        u = brain.unit
        if brain.wander_pause > 0:
            brain.wander_pause -= 1
            return None
        radius = brain.guard_radius
        dx = dy = 0.0
        for _ in range(WANDER_TRIES):
            dx = self.rng.uniform(-radius, radius)
            dy = self.rng.uniform(-radius, radius)
            if dx * dx + dy * dy <= radius * radius:
                break
        x, y = center[0] + dx, center[1] + dy
        if ground_slope(getattr(u, 'terrain', None), x, y) > WANDER_MAX_SLOPE:
            return None
        near = WANDER_FRIEND_RADII * float(getattr(u.fighter, 'radius', 0.5) if u.fighter is not None else 0.5)
        for other in self.units():
            if other is u or other.is_dead:
                continue
            if math.hypot(other.x - x, other.y - y) <= near and \
                    self.diplomacy.friends(self.player_of(other), brain.player):
                return None
        if not u.go_to(x, y):
            distance = math.hypot(x - u.x, y - u.y)
            brain.wander_pause = min(WANDER_MAX_PAUSE, int(round(distance * WANDER_PAUSE_PER_DISTANCE)))
            return None
        return x, y

    def look_actions(self, brain, point, guard=True):
        """
        5d5b40 (partly): the looks of a guard at the point it walks to, [(yaw, wait ticks)]. A look is turned
        to a random side pulled outwards by the place of the point in the circle; game.exe also takes a side
        which the unit can see (5bc3e0 (partly)) and plays a special idle with the long look, the remake does not.
        The suspicion (guard is False) gives no time and no circle: the default time, any side.
        """
        u = brain.unit
        count = self.rng.randrange(4) + 1
        stay = brain.settings.guard_stay_time if guard else -1.0
        longest = stay * LOOK_TICKS_PER_STAY if stay >= 0.0 else LOOK_DEFAULT_TICKS
        if self.rng.randrange(LOOK_LONG_CHANCE) == 0 and \
                not u.figure.model_name.lower().startswith(LOOK_LONG_SKIP_PREFIX):
            longest *= count
            count = 1
        ox, oy = point[0] - brain.home[0], point[1] - brain.home[1]
        if not guard:
            ox = oy = 0.0
        elif brain.guard_radius > 0.0:
            pull = math.hypot(ox, oy) / brain.guard_radius * 0.5
            ox, oy = ox * pull, oy * pull
        actions = []
        for _ in range(count):
            angle = self.rng.uniform(0.0, 2.0 * math.pi)
            dx, dy = math.cos(angle) + ox, math.sin(angle) + oy
            wait = (longest - longest * LOOK_MIN_PART) * self.rng.random() + longest * LOOK_MIN_PART
            actions.append((math.atan2(dx, -dy), wait))
        return actions

    def follow(self, brain):
        """
        5d6f90: the unit goes after its leader.
        """
        leader = brain.leader
        if leader is None or leader.is_dead:
            brain.order = ORDER_NONE
            brain.leader = None
            return
        self.go_after(brain, leader)

    def go_after(self, brain, leader):
        u = brain.unit
        if u.distance_to(leader) > FOLLOW_DISTANCE * 2.0:
            u.go_to(leader.x, leader.y, run=brain.runs)
        elif u.mode in unit_module.MOVING_MODES and u.distance_to(leader) <= FOLLOW_DISTANCE:
            u.idle()

    # the orders of the scripts (4b6760), unit is a unit of the map or of the party

    def order_move(self, unit, x, y):
        """
        MoveToPoint (2, 5d7ae0): the state 1 and the walk to the point by the way of Walk or Run.
        """
        brain = unit.brain
        if brain is not None:
            brain.order = ORDER_MOVE
            brain.target = None
            unit.go_to(x, y, run=brain.runs)
        else:
            unit.go_to(x, y)

    def order_turn(self, unit, x, y):
        """
        RotateTo (3, 5d7b30): the state 1 and the turn to the point.
        """
        if unit.brain is not None:
            unit.brain.order = ORDER_MOVE
        if unit.mode in unit_module.MOVING_MODES:
            unit.idle()
        unit.target_yaw = look_yaw(x - unit.x, y - unit.y)

    def order_follow(self, unit, leader):
        """
        Follow (0x3b, 5d7e40): the state 6 with the leader.
        """
        if unit.brain is not None and leader is not None:
            unit.brain.order = ORDER_FOLLOW
            unit.brain.leader = leader

    def set_gait(self, unit, runs):
        """
        Walk (0x40) and Run (0x41): 555270 with 2 or 3.
        """
        if unit.brain is not None:
            unit.brain.runs = runs

    def clear_behaviours(self, unit):
        """
        UMClear (0x24, 5d4df0): no behaviour is left, the unit neither fights nor guards.
        """
        brain = unit.brain
        if brain is None:
            return
        brain.attacks = False
        brain.wary = False
        brain.guards = False
        brain.revenge = False
        brain.fears_all = False
        brain.companion = None
        brain.player_mode = False
        brain.route = []
        brain.target = None

    def reset_target(self, unit):
        """
        ResetTarget (0xe8, 5d7c30): the enemies the unit knows are forgotten (5cb0f0) and the record of the logic
        of its alarm starts again (CUnitLogic::Init): no target, the patrol from its first point, no wait.
        The behaviours given by the scripts stay (the remake's reading).
        """
        brain = unit.brain
        if brain is None:
            return
        brain.target = None
        brain.route_index = -1
        brain.actions = None
        brain.wait_until = None
        brain.fleeing = False

    def stop_orders(self, unit):
        """
        Idle (0x34): 5d4df0 and 5d7790, the state 0.
        """
        self.clear_behaviours(unit)
        if unit.brain is not None:
            unit.brain.order = ORDER_NONE
            unit.brain.leader = None
        if unit.mode in unit_module.MOVING_MODES + (unit_module.MODE_ATTACK,):
            unit.idle()

    def standard_behaviours(self, unit):
        """
        UMStandard (0x2c): the aggression (4ba710) and the two behaviours 4ba610 and 4ba920, without the caution
        of 5d2850; the remake takes the logic again.
        """
        brain = unit.brain
        if brain is None:
            return
        runs = brain.runs
        brain.set_logic(brain.logic)
        brain.attacks = True
        brain.wary = False
        brain.guards = True
        brain.runs = runs

    def aggression(self, unit):
        """
        UMAggression (0x2d): the behaviour 73d260 (4ba710). Its weight (5d2390) is 10000 when it has a target
        (5d0030 chooses one of the enemies which the unit sees, 5cb260), else -1000; then (5d2580) it calls for
        help (5d5990) and attacks the target (5d7b50) or casts a spell at it (554f50); 5d26d0 and 5d2710 put on
        and off the mood of the aggression (+0x174 of the unit, the signals 6 and 4).
        """
        brain = unit.brain
        if brain is not None:
            brain.attacks = True

    def revenge(self, unit):
        """
        UMRevenge (0x2e): the same behaviour with +0xc = 1 (4ba830): the target is the one who hit the unit
        (+0x30 of the logic) unless it is a friend (7c2bb4) or the unit itself; it gives the target up when it dies.
        """
        brain = unit.brain
        if brain is not None:
            brain.revenge = True

    def fear(self, unit, all_units):
        """
        UMFear( o, f ) (0xc7): the behaviour 5d2850 with the flag f != 0 (+0x28). It weighs 15000 while a shooter
        keeps away from an enemy (5d2960); with the flag the units near (+0x114 of the logic) are remembered
        as threats (5cbbe0) and the unit runs from them (not in the remake).
        """
        brain = unit.brain
        if brain is not None:
            brain.wary = True
            brain.fears_all = bool(all_units)

    def follow_behaviour(self, unit, leader):
        """
        UMFollow( o, leader ) (0xca): the behaviour 73d1c4 (4ba450) of the weight 50; when the logic is free
        (+4 is 0) it sets the state 6 of the follow with the leader (5d1e10), so the fight outweighs it.
        """
        brain = unit.brain
        if brain is not None and is_unit_like(leader):
            brain.companion = leader

    def player_behaviour(self, unit):
        """
        UMPlayer (0x31): 5d4df0, the behaviour 5d3760 (vtable 73f0f8) and 5d7790. The behaviour (5d4a90) carries
        out the orders and fights when the player of the unit is in danger (+0x5120 is 2); the remake lets
        the unit look for an enemy only then and answer a blow at any time.
        """
        brain = unit.brain
        if brain is None:
            return
        self.stop_orders(unit)
        brain.attacks = True
        brain.player_mode = True

    def sentry(self, unit, x, y):
        """
        UMSentry (0x25, 4ba360 / 5d1b50, weight 50) and Sentry (0x94): the unit stands at the point.
        """
        self.guard(unit, x, y, SENTRY_RADIUS)

    def guard(self, unit, x, y, radius):
        """
        UMGuard (0x26) and Guard (0x33, 5d6230): the guard of the point with the radius.
        """
        brain = unit.brain
        if brain is None:
            return
        brain.guards = True
        brain.route = []
        brain.home = (x, y)
        brain.guard_radius = radius
        if brain.order == ORDER_NONE and unit.mode == unit_module.MODE_IDLE and brain.home_distance() > radius:
            unit.go_to(x, y, run=brain.runs)

    def keep_away(self, brain):
        """
        5d2960: a shooter in good health runs from the nearest enemy which is nearer than RangedAttackDist
        (RangedRunDist while it runs) to a point 5 farther from it, moved by up to 3 at random.
        """
        u = brain.unit
        fighter = u.fighter
        if not u.shoots or fighter.max_health <= 0 or fighter.health / fighter.max_health <= RANGED_HEALTH_PART:
            brain.fleeing = False
            return False
        if brain.fleeing and u.mode in unit_module.MOVING_MODES:
            return True
        enemy = self.nearest_enemy(brain)
        limit = self.settings.ranged_run_dist if brain.fleeing else self.settings.ranged_attack_dist
        if enemy is None or u.distance_to(enemy) >= limit:
            brain.fleeing = False
            return False
        dx, dy = u.x - enemy.x, u.y - enemy.y
        length = math.hypot(dx, dy) or 1.0
        x = u.x + dx / length * RANGED_RUN_STEP + self.rng.uniform(-RANGED_JITTER, RANGED_JITTER)
        y = u.y + dy / length * RANGED_RUN_STEP + self.rng.uniform(-RANGED_JITTER, RANGED_JITTER)
        if not u.go_to(x, y, run=True):
            brain.fleeing = False
            return False
        if _Debug:
            print('%s runs from %s to %.1f,%.1f' % (u, enemy, x, y))
        brain.fleeing = True
        return True

    def patrol(self, brain):
        """
        5d1700 and 5d7520: the walk to the next point, the turns and the waits of its actions.
        """
        u = brain.unit
        if u.mode not in (unit_module.MODE_IDLE, unit_module.MODE_ACTION):
            return
        if brain.actions is None:
            x, y, _ = brain.route[brain.route_index] if brain.route_index >= 0 else (None, None, None)
            if x is not None and math.hypot(u.x - x, u.y - y) <= PATROL_ARRIVE_DISTANCE:
                brain.actions = list(brain.route[brain.route_index][2])
                brain.wait_until = None
            else:
                self.next_point(brain)
                return
        if brain.wait_until is not None:
            if self.ticks < brain.wait_until:
                return
            brain.wait_until = None
            brain.actions.pop(0)
        if not brain.actions:
            self.next_point(brain)
            return
        yaw, wait = brain.actions[0]
        u.target_yaw = yaw
        brain.wait_until = self.ticks + wait

    def next_point(self, brain):
        brain.route_index = (brain.route_index + 1) % len(brain.route)
        brain.actions = None
        x, y, _ = brain.route[brain.route_index]
        if _Debug:
            print('%s patrols to %.1f,%.1f' % (brain.unit, x, y))
        if not brain.unit.go_to(x, y):
            # no way to the point: the actions are done where the unit stands
            brain.actions = list(brain.route[brain.route_index][2])
            brain.wait_until = None

#------------------------------------------------------------------------------


def ground_slope(terrain, x, y):
    """
    5bc2a0: the biggest difference of the height at the point and at the 4 points WANDER_SLOPE_STEP around it.
    """
    if terrain is None:
        return 0.0
    height = terrain.height_at(x, y)
    step = WANDER_SLOPE_STEP
    return max(abs(terrain.height_at(x + sx, y + sy) - height)
               for sx, sy in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)))


def cell_middle(value):
    """
    5d6630: the middle of the cell of the grid of the paths.
    """
    return (round(value / PATROL_CELL) + 0.5) * PATROL_CELL


def is_unit_like(obj):
    return obj is not None and hasattr(obj, 'is_dead') and hasattr(obj, 'distance_to')


def is_busy(unit):
    """
    SleepUntilIdle: the unit still goes or turns.
    """
    if unit.mode in unit_module.MOVING_MODES:
        return True
    diff = (unit.target_yaw - unit.yaw + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff) > TURN_DONE


def look_yaw(dx, dy):
    """
    The yaw of a unit which looks along (dx, dy): the models look at -y.
    """
    return math.atan2(dx, -dy)
