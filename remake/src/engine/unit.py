"""
A unit on the map: its figure, the animation it plays and the movement, as the client part
of a unit in game.exe does it (CUnitClientSpecific, sub_535850 and vf40 0x535b30).

- the height is the height of the land under the unit;
- the unit turns smoothly: the difference of the angles is wrapped into [-pi, pi] and shrinks by exp(-0.3 * dt),
  dt is counted here in ticks of 55 ms;
- the unit moves by the speeds of its stats (engine.stats: run and walk of units.udb or of UNIT_STATS,
  map units per tick, times the multiplier of the speed and the factor of the legs); without them by the natural
  speed of the cycle;
- the walk and run animations play as fast as the unit really moves (535850): the rate of the clip is
  speed * 15 / (size * movement_speed * 2), so the natural speed of a clip is movement_speed * size / 7.5
  map units per tick;
- a path (engine.pathing) starts from the direction in which the unit looks;
- an attack (engine.combat): the unit comes to the foe, plays an attack of the state ATTACK (550100 asks for
  0x90000 | weapon), the blow is resolved when the clip reaches hit_frame (the server waits hit_frame / speed
  ticks), the foe plays SUFFER or DEATH; the attacks go on until the foe is dead. A strike starts only when
  the delay of the last one is over (54d960, +0x184): round(delay * 15 / the speed of the strikes) ticks after
  its start, the delay of the weapon item or of the prototype; meanwhile the unit stands at the foe (54d140); How close the server comes
  to strike is not known yet: here the radii of both units plus the reach of the attacker;
- a cast (engine.magic): the unit comes within the range of the spell, plays a CAST action once and lets
  the spell go at its hit_frame (54ee10): a unit of the party pays the mana, the spell flies as a missile
  or hits at once, a spell without damage heals or goes to the list of the spells on the target when its filter
  lets it; how close the server comes to cast (54e600) is not known yet: here the range itself;
- a spell spreads over its targets (engine.magic.spread, 58e850, the units come from neighbours()); a spell
  of the point (cast at an engine.magic.Spot or at a unit) makes an engine.magic.AreaEffect which the world
  runs with update_spell_effects();
- the spells on a unit lose a tick every tick and change its numbers and senses (engine.combat.Fighter.recount,
  engine.magic.senses); a stunned unit stands still and takes no orders;
- when an animation which is not a cycle ends, a new idle is chosen by action_probability;
- a walk or a run starts with the clip of the stage START (scross19 for the walk of unhuma) and the cycle is
  queued after it; the server (551400) builds this list again only when the key of the movement changes
  (+0x940): another type (walk or run) or another modifier, so a new target on the way keeps the cycle;
- there is no END stage in the databases: on arrival the idle is played right away;
- wounded legs (engine.stats.WoundLevels) slow the unit, pick the limping cycle and forbid the run;
  the cycle then plays as fast as the unit really moves (535850).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import random

import numpy as np

#------------------------------------------------------------------------------

from eilib import adb
from engine import actions
from engine import animation
from engine import combat
from engine import experience
from engine import magic
from engine import missiles
from engine import pathing
from engine import sounds
from engine import stats
from engine import timing

#------------------------------------------------------------------------------

TURN_RATE = 0.3

ARRIVE_DISTANCE = 0.1

# CUnit::RunLimit 5241c0: why a unit may not run
RUN_FREE = 0
RUN_LEGS = 1
RUN_OVERLOAD = 2
# 524560: the weight a hero carries is strength * 12 * (1 + the power of the perks "lift" * 0.01),
# 1000000 for the others
CARRY_PER_STRENGTH = 12.0
NO_CARRY_LIMIT = 1000000.0
GROUP_LIFT = 21

# 535850: rate = speed * 15 / (size * movement_speed * 2)
CLIP_SPEED_DIVISOR = 7.5

MODE_IDLE = 'idle'
MODE_WALK = 'walk'
MODE_RUN = 'run'
MODE_SNEAK = 'sneak'
MODE_CRAWL = 'crawl'
MODE_ACTION = 'action'
MODE_ATTACK = 'attack'
MODE_DEAD = 'dead'

# the modes in which the unit walks to its target
MOVING_MODES = (MODE_WALK, MODE_RUN, MODE_SNEAK, MODE_CRAWL)

# CUnitServer::vf45 5536b0 (partly), read by CSquad::Detect 5cb850. A unit under the order to go (the state 0 of
# +0x164) makes a noise (+0x8f8) for the hearing of the others by its way of moving +0x264: the crawl 0.4 (73ef00),
# the sneak 0.2 (73eefc), the walk 1, the run 2 (73eef4); any other order is silent. A running unit hears with
# 0.5 (73ef10, +0x8fc). The crawl (0.5, 73ef04) and the sneak (0.75, 73ef08) make a unit harder to see (+0x900),
# whatever it does.
NOISE_CRAWL = 0.4
NOISE_SNEAK = 0.2
NOISE_WALK = 1.0
NOISE_RUN = 2.0
RUN_HEARING = 0.5
VISIBILITY_CRAWL = 0.5
VISIBILITY_SNEAK = 0.75
# 5536b0 (partly): the infra is in percents (73d09c); 58ccd0: the sight while it rains or snows (+0x88 of the world)
SIGHT_INFRA_SCALE = 0.01
WET_SIGHT = 0.9
# 55fc50: the darkness is 1 before 8 o'clock, 0 till 20 and grows by 0.25 an hour after 20 (73ce50, 73ccf8, 73c3b0)
DARK_MORNING = 8.0
DARK_EVENING = 20.0
DARK_PER_HOUR = 0.25

# the senses of stats.UnitStats: 0 sight, 1 infra, 2 life, 3 hearing, 4 smell, 5 tracking
SENSE_INFRA = 1
SENSE_HEARING = 3

# how often a unit which comes to its foe looks for a new path to it
CHASE_TICKS = 10

# a foe nearer than the distance of the strike plus this is come to straight when the path does not lead
# nearer to it (it stands on a closed cell of the grid), the remake's choice
STRAIGHT_CHASE_DISTANCE = 2.0

#------------------------------------------------------------------------------


class Unit(object):

    def __init__(self, figure_model, terrain=None, position=(0.0, 0.0), yaw=0.0, weapon=None, state=actions.STATE_NEUTRAL, rng=None):
        self.figure = figure_model
        self.terrain = terrain
        self.rng = rng or random.Random()
        self.x, self.y = float(position[0]), float(position[1])
        self.z = self.ground_height()
        self.yaw = yaw
        self.target_yaw = yaw
        self.weapons = actions.weapon_bits(weapon)
        # the type of the weapon in the hands (adb.WEAPONS), None without one
        self.weapon_code = adb.WEAPONS.index(weapon) if weapon in adb.WEAPONS else None
        self._bounds = None
        self.state = state
        self.mode = MODE_IDLE
        # the way of moving which the unit took last (engine.stats.SPEED_*, +0x264 of the unit)
        self.gait = stats.SPEED_WALK
        self.target = None
        # map units per tick
        self.speed = 0.0
        # run, walk, sneak and crawl in map units per tick (engine.stats), None: the natural speeds of the clips
        self.speeds = None
        self.speed_scale = 1.0
        self.unit_speeds = {}
        self.player = animation.AnimationPlayer(on_clip_end=self.on_clip_end)
        self.last_action = None
        self.wounds = stats.WoundLevels()
        self.legs_part = 1.0
        self.move_key = None
        self.rate = 1.0
        self.waypoints = []
        self.path_grid = None
        self.path_class = None
        # engine.combat.Fighter, None for a unit which does not fight
        self.fighter = None
        self.foe = None
        # +0x438, the button "battle" of the interface (6497d0): a unit of the player fights on its own (5d4170 (partly)),
        # on by default (52e5b0)
        self.auto_fight = True
        self.foe_run = True
        self.strike_action = None
        # the ticks until the next strike may start (+0x184 of the unit of the server less the tick of the world)
        self.strike_wait = 0.0
        # the unit stands at the foe until strike_wait is over
        self.waiting = False
        # the part of the body of the called shot (combat.HEAD...), None for a usual blow; the order keeps it
        # at +0xf0 with the flag of the aim at +0xf4 (546480 resets them to the torso and no aim)
        self.aim_part = None
        # engine.magic.Spell which the unit casts at the foe, None for the blows
        self.spell = None
        self.chase_ticks = 0.0
        # called with (attacker, target, hit, dealt), dealt is None for a miss
        self.on_blow = None
        # called with (killer, victim) when a blow of the unit kills the victim
        self.on_kill = None
        # engine.sounds.SoundTracker of the voice, None for a silent unit
        self.sound_tracker = None
        # called with (unit, sounds, stopped channels) when sounds start or stop
        self.on_sounds = None
        # the channels of engine.sounds whose sound still plays (the client keeps them)
        self.busy_channels = set()
        # called with an engine.missiles.Arrow which a shot sends; without it the blow comes at once
        self.on_shoot = None
        # called with (unit, attacker) when a blow comes to the unit, hit or missed
        self.on_attacked = None
        # called with (unit, item, message) when an item of a unit of the player is worn out or broken
        # (engine.combat.MESSAGE_ARMOR_BROKEN...)
        self.on_item_wear = None
        # called with (the unit, the level of engine.signals) when a mood signal starts
        self.on_signal = None
        # the flag 0x400000 of the client is off: the player does not see the unit (engine.ai.Controller.visible),
        # its figure, its sounds, its mark and its numbers are not shown
        self.unseen = False
        # called with (the unit, RUN_LEGS or RUN_OVERLOAD) when a unit of the player which was told to run may not
        # (548520 (partly)): its answer of acks.db
        self.on_cant_run = None
        # the reason of the last refusal to run which was told, RUN_FREE when the unit may run
        self.run_refusal = RUN_FREE
        # engine.ai.Brain of a unit of the computer
        self.brain = None
        # HideObject of the scripts (the flag 0x10000): the unit is not drawn
        self.hidden = False
        # the flags of the scripts: 0x800000 BlockUnit, 0x100000 the body was looted (looting is not made yet)
        self.blocked = False
        self.looted = False
        # eilib.idb.QuestItem which the unit carries (+0x4a0, the count +0x4a4)
        self.quest_items = []
        # the name of the prototype of units.udb (the phrases of acks.db are found by it)
        self.prototype_name = None
        # the experience which the party gets for the kill of the unit (the tag 0x2b of its prototype)
        self.kill_experience = 0.0
        # engine.experience.HeroStats of a hero or a mercenary, None for the others
        self.hero_stats = None
        # called with (unit, other): whether the other one is an enemy, for the filters of the spells
        self.hostile_to = None
        # called without arguments: the units of the world which a spell may spread over (58e850)
        self.neighbours = None
        # called with an engine.magic.AreaEffect which a spell makes on the ground
        self.on_spell_effect = None
        # called with an engine.magic.SpellSound of a spell which came to the unit
        self.on_spell_sound = None
        # called with (engine.particles.Emitter, the unit) of a spell which hangs its particles on the unit
        self.on_spell_emitter = None
        # the six senses and detections of engine.stats.UnitStats, the sight is the first one
        self.senses = (0.0, ) * stats.SENSES_COUNT
        self.detections = (1.0, ) * stats.SENSES_COUNT
        self.base_senses = self.senses
        self.base_detections = self.detections
        # +0x224 and +0x228 of the unit of the server (UNIT_STATS 10 and 11): the cosine of the half of the angle of
        # the sight and the radius of the peripheral vision (5ca5a0 (partly)); a unit without the record sees
        # all around
        self.view_cos = -1.0
        self.peripheral = 0.0
        # 55fc50: the darkness of the hour 0..1 (+0x84 of the world), and the rain or the snow falls (+0xe4)
        self.darkness = 0.0
        self.wet = False
        # UNIT_STATS 15 (+0x250), the range of the attack of a unit without a weapon item
        self.attack_range = 0.0
        self.idle()

    def __repr__(self):
        return 'Unit(%s at %.1f,%.1f %s)' % (self.figure.model_name, self.x, self.y, self.mode)

    def ground_height(self):
        return self.land_height(self.x, self.y)

    def land_height(self, x, y):
        if self.terrain is None:
            return 0.0
        return self.terrain.height_at(x, y)

    def find_action(self, action_type, stage=actions.STAGE_CYCLE, modifier=0, state=None):
        if self.figure.database is None:
            return None
        return actions.find(self.figure.database, self.state if state is None else state, action_type,
                            modifier=modifier, stage=stage, weapons=self.weapons, rng=self.rng)

    def play_action(self, action, queue=False):
        if action is None:
            return False
        clip = self.figure.clip(action)
        if clip is None:
            return False
        if queue:
            self.player.enqueue(clip)
        else:
            self.player.play(clip)
        self.last_action = action
        return True

    def teleport_to(self, x, y):
        """
        5c11d0: the unit is at the point at once, it does not go anywhere.
        """
        self.stop()
        self.x, self.y = x, y
        self.z = self.ground_height()

    def stop(self):
        self.waiting = False
        self.waypoints = []
        self.target = None
        self.speed = 0.0
        self.rate = 1.0
        self.move_key = None

    def idle(self):
        self.stop()
        self.foe = None
        self.spell = None
        self.strike_action = None
        self.mode = MODE_IDLE
        self.play_action(self.find_action(actions.TYPE_IDLE))

    @property
    def is_dead(self):
        return self.mode == MODE_DEAD

    @property
    def shoots(self):
        """
        54b6c0: a bow or a crossbow in the hands, or a range of the attack above 1 without a weapon.
        """
        if self.weapon_code is not None:
            return missiles.is_shooting(self.weapon_code)
        return self.attack_range > 1.0

    def distance_to(self, other):
        return math.hypot(other.x - self.x, other.y - self.y)

    def strike_distance(self, foe):
        if self.spell is not None:
            return self.spell.range
        if self.fighter is None or foe.fighter is None:
            return 1.0
        return self.fighter.radius + foe.fighter.radius + self.fighter.reach

    def can_strike(self, foe):
        return self.distance_to(foe) <= self.strike_distance(foe) + ARRIVE_DISTANCE

    def attack(self, foe, run=True):
        """
        Comes to the foe and strikes it until it is dead. False when the unit or the foe can not fight.
        """
        if self.is_dead or foe is None or foe is self or foe.is_dead or self.fighter is None or foe.fighter is None:
            return False
        if self.is_frozen:
            return False
        self.spell = None
        self.foe = foe
        self.foe_run = run
        self.chase()
        return True

    def cast(self, foe, spell, run=False):
        """
        Comes within the range of the spell and casts it at the foe once. False when the unit can not:
        a unit of the party needs the mana of the spell (the remake's check, the server's one is not found yet)
        and, a hero, the skill of its school not below the complexity (554f50).
        """
        if self.is_dead or foe is None or foe.is_dead or self.fighter is None or spell is None:
            return False
        if foe.fighter is None and not isinstance(foe, magic.Spot):
            return False
        if self.is_frozen:
            return False
        if self.fighter.in_party and self.fighter.mana < self.spell_mana(spell):
            return False
        if self.fighter.in_party and self.hero_stats is not None:
            if not spell.can_cast(self.hero_stats.school_skill(spell.prototype.subtype_id)):
                return False
        self.spell = spell
        self.foe = foe
        self.foe_run = run
        self.chase()
        return True

    def spell_mana(self, spell):
        """
        54ee10: the mana of the spell for a unit of the party, mana * 25 / its intelligence (+0x20).
        """
        if self.hero_stats is not None:
            return spell.mana_cost(self.hero_stats.intelligence)
        return spell.mana_cost()

    def chase(self):
        foe = self.foe
        if self.can_strike(foe):
            self.strike()
            return
        self.chase_ticks = CHASE_TICKS
        found = self.go_to(foe.x, foe.y, run=self.foe_run)
        stuck = (found and not self.waypoints and self.target is not None and
                 math.hypot(self.target[0] - self.x, self.target[1] - self.y) <= ARRIVE_DISTANCE)
        if not found or stuck:
            if self.distance_to(foe) > self.strike_distance(foe) + STRAIGHT_CHASE_DISTANCE:
                self.idle()
                return
            self.move_to(foe.x, foe.y, run=self.foe_run)
        self.mode = MODE_ATTACK
        self.foe = foe

    def strike(self):
        """
        54d960: a blow starts only when the delay of the last one is over (+0x184), then the next one may come
        after the delay of the weapon at the speed of the strikes; until then the unit stands at the foe (54d140).
        """
        foe = self.foe
        self.stop()
        self.target_yaw = math.atan2(foe.x - self.x, -(foe.y - self.y))
        self.waiting = False
        if self.spell is None and self.strike_wait > 0:
            self.waiting = True
            self.mode = MODE_ATTACK
            self.foe = foe
            if self.player.clip is None or self.player.clip.action is None or \
                    self.player.clip.action.type != actions.TYPE_IDLE:
                self.play_action(self.find_action(actions.TYPE_IDLE))
            return
        if self.spell is not None:
            action = self.find_action(actions.TYPE_CAST, stage=actions.STAGE_UNIQUE)
        else:
            action = self.find_action(actions.TYPE_ATTACK, stage=actions.STAGE_UNIQUE, state=actions.STATE_ATTACK)
        if action is None or not self.play_action(action):
            self.spell = None
            self.idle()
            return
        self.mode = MODE_ATTACK
        self.foe = foe
        self.strike_action = action
        if self.spell is None:
            self.strike_wait = max(self.strike_wait, self.fighter.strike_ticks(self.strike_speed()))

    def strike_speed(self):
        """
        524560: the speed of the strikes of the unit (combat.Fighter.strike_speed): a hero with its dexterity and
        the perk of the quickness, the wounded arms slow it as the legs slow the walk (524110).
        """
        dexterity = None
        quickness = 0.0
        if self.fighter.in_party and self.hero_stats is not None:
            dexterity = self.hero_stats.dexterity
            quickness = self.hero_stats.perk_power(experience.GROUP_QUICKNESS)
        arms = self.wounds.speed_factor(self.fighter.part_health('arm'))
        return self.fighter.strike_speed(dexterity, quickness, arms)

    def resolve_blow(self):
        """
        The moment of the blow (hit_frame of the attack): engine.combat decides whether it hits and how much.
        """
        foe = self.foe
        self.strike_action = None
        if foe is None or foe.is_dead:
            return
        if self.spell is not None:
            self.release_spell(foe)
            return
        shooting = missiles.is_shooting(self.weapon_code)
        # 530a60: no backstab with a bow or a crossbow
        backstab = (self.fighter.in_party and not shooting and
                    combat.is_backstab((self.x, self.y), (foe.x, foe.y), foe.yaw))
        skill = 0.0
        if self.hero_stats is not None:
            skill = (self.hero_stats.skill(experience.SKILL_BACKSTAB) +
                     self.hero_stats.perk_power(experience.GROUP_BACKSTAB))
        if self.aim_part is None:
            hit = combat.make_hit(self.fighter, self.rng, backstab=backstab, skill=skill)
        else:
            hit = combat.make_hit(self.fighter, self.rng, location=self.aim_part, backstab=backstab,
                                  aimed=True, skill=skill)
        if shooting and self.on_shoot is not None:
            # 54f450: the arrow carries the blow to the target
            self.on_shoot(missiles.Arrow(self, foe, hit, (self.x, self.y, self.z + self.height() * 0.5),
                                         ground_height=self.land_height))
            return
        self.deliver_blow(foe, hit)

    def release_spell(self, foe):
        """
        54ee10: the spell goes at the moment of the cast; the unit stops after it.
        """
        spell = self.spell
        self.spell = None
        self.foe = None
        if self.fighter.in_party:
            self.fighter.mana = min(max(0.0, self.fighter.mana - self.spell_mana(spell)), self.fighter.max_mana)
        if self.neighbours is not None:
            units = self.neighbours()
        else:
            units = [foe] if foe.fighter is not None else []
        for target in magic.spread(spell, self, foe, units):
            self.make_spell_effect(spell, target)

    def make_spell_effect(self, spell, target):
        make_spell_effect(self, spell, target)

    def is_hostile(self, other):
        if self.hostile_to is not None:
            return self.hostile_to(self, other)
        return other is not self

    def take_spell_blow(self, hit, caster, attacker):
        """
        A blow of an effect of a spell on the ground (vf20 of the unit by 683570); without the attacker
        the unit does not answer it.
        """
        if self.is_dead or self.fighter is None:
            return None
        if attacker is not None and not attacker.is_dead and attacker.fighter is not None:
            attacker.deliver_blow(self, hit)
            return None
        dealt = combat.apply_hit(self.fighter, hit)
        self.report_wear()
        self.suffer(dealt)
        if caster is not None and caster.on_blow is not None:
            caster.on_blow(caster, self, hit, dealt)
        if caster is not None and self.is_dead and caster.on_kill is not None:
            caster.on_kill(caster, self)
        return dealt

    def deliver_blow(self, foe, hit):
        """
        The blow comes to the foe: whether it hits, the damage, the sounds, SUFFER or DEATH.
        """
        dealt = None
        was_alive = not foe.is_dead
        if combat.hits(self.fighter, foe.fighter, self.distance_to(foe), hit):
            dealt = combat.apply_hit(foe.fighter, hit)
            self.report_wear()
            foe.report_wear()
            if hit.attack is None and self.sound_tracker is not None and foe.sound_tracker is not None:
                foe.emit_sound(sounds.blow_sound(self.sound_tracker.voice.weapon_code,
                                                 foe.sound_tracker.voice.material, dealt))
            foe.suffer(dealt)
        elif self.sound_tracker is not None:
            self.emit_sound(sounds.miss_sound(self.sound_tracker.voice))
        if self.on_blow is not None:
            self.on_blow(self, foe, hit, dealt)
        if was_alive and foe.is_dead and self.on_kill is not None:
            self.on_kill(self, foe)
        if foe.on_attacked is not None:
            foe.on_attacked(foe, self)

    def report_wear(self):
        """
        The messages about the wear of the items of the unit (5253e0 -> 552780) go to on_item_wear.
        """
        if self.fighter is None or not self.fighter.wear_messages:
            return
        messages, self.fighter.wear_messages = self.fighter.wear_messages, []
        if self.on_item_wear is None:
            return
        for item, message in messages:
            self.on_item_wear(self, item, message)

    def regenerate(self, ticks):
        """
        CUnitServer::vf06 (552300): a living unit gets back its health and, while it has no order, its mana
        (523740); a hero with the perks of the vitality and the spirit faster.
        """
        if self.fighter is None or self.is_dead:
            return
        vitality = spirit = 0.0
        if self.hero_stats is not None:
            vitality = self.hero_stats.perk_power(experience.GROUP_VITALITY)
            spirit = self.hero_stats.perk_power(experience.GROUP_SPIRIT)
        if self.fighter.regenerate(ticks, idle=self.mode == MODE_IDLE, vitality=vitality, spirit=spirit):
            self.legs_part = self.fighter.part_health('leg')

    def update_sounds(self):
        if self.sound_tracker is None:
            return
        action = self.player.clip.action if self.player.clip is not None else None
        started, stopped = self.sound_tracker.update(action, self.player.time, self.busy_channels,
                                                     step_tile=self.step_tile)
        if started or stopped:
            if self.on_sounds is not None:
                self.on_sounds(self, started, stopped)

    def step_tile(self):
        """
        534590: the type of the tile under the unit for its steps, the one of a bridge above the land.
        """
        if self.terrain is None:
            return 0
        if abs(self.z - self.ground_height()) > sounds.STEP_HEIGHT:
            return sounds.BRIDGE_TILE_TYPE
        return self.terrain.tile_type_at(self.x, self.y)

    def emit_sound(self, sound):
        if sound is not None and self.on_sounds is not None:
            self.on_sounds(self, [sound], [])

    def suffer(self, dealt):
        """
        A blow came: the legs may limp now, the unit dies when its health is gone, else it plays SUFFER.
        """
        self.legs_part = self.fighter.part_health('leg')
        if self.fighter.is_dead:
            self.die()
            return
        if dealt and self.mode in (MODE_IDLE, MODE_ACTION):
            self.do_action(self.find_action(actions.TYPE_SUFFER, stage=actions.STAGE_UNIQUE))

    def die(self):
        self.stop()
        self.foe = None
        self.spell = None
        self.strike_action = None
        self.mode = MODE_DEAD
        self.play_action(self.find_action(actions.TYPE_DEATH, stage=actions.STAGE_UNIQUE))

    def do_action(self, action):
        """
        Plays one action of the database (an attack, a cast, a death...) and returns to the idle after it.
        """
        self.stop()
        self.mode = MODE_ACTION
        return self.play_action(action)

    def move_to(self, x, y, run=False, gait=None):
        """
        The action and the speed of going to a point: gait is engine.stats.SPEED_RUN, SPEED_WALK, SPEED_SNEAK
        or SPEED_CRAWL; without it run gives the run or the walk. Crawl plays the ccrawl cycle of the model
        when it has one, sneak keeps the walk cycle at the sneak speed.
        """
        factor = self.wounds.speed_factor(self.legs_part)
        if gait is None:
            gait = stats.SPEED_RUN if run else stats.SPEED_WALK
        if gait == stats.SPEED_RUN:
            # 555270 does not take the run while RunLimit holds, 548520 (partly) tells why
            reason = self.run_limit()
            self.refuse_run(reason)
            if reason != RUN_FREE:
                gait = stats.SPEED_WALK
        action_type = actions.TYPE_RUN if gait == stats.SPEED_RUN else actions.TYPE_WALK
        cycle = self.find_action(action_type)
        if cycle is None and action_type == actions.TYPE_RUN:
            action_type = actions.TYPE_WALK
            gait = stats.SPEED_WALK
            cycle = self.find_action(action_type)
        if cycle is None:
            return
        if gait == stats.SPEED_CRAWL:
            cycle = self.find_crawl_action() or cycle
        # without the speeds of the stats the unit moves as fast as its plain cycle
        plain_speed = self.natural_speed(cycle)
        modifier = 0
        if action_type == actions.TYPE_WALK:
            modifier = self.wounds.walk_modifier(factor)
            if modifier:
                limping = self.find_action(action_type, modifier=modifier)
                if limping is not None and limping.modifier == modifier:
                    cycle = limping
                else:
                    modifier = 0
        self.target = (float(x), float(y))
        self.waypoints = []
        key = (action_type, modifier, gait)
        if key != self.move_key or self.player.clip is None:
            start = self.find_action(action_type, stage=actions.STAGE_START)
            if start is not None and start.stage == actions.STAGE_START and self.play_action(start):
                self.play_action(cycle, queue=True)
            else:
                self.play_action(cycle)
            self.move_key = key
        self.mode = {stats.SPEED_RUN: MODE_RUN, stats.SPEED_CRAWL: MODE_CRAWL,
                     stats.SPEED_SNEAK: MODE_SNEAK}.get(gait, MODE_WALK)
        self.gait = gait
        natural = self.natural_speed(cycle)
        if self.speeds:
            self.speed = self.speeds[gait] * self.speed_scale * factor
        else:
            self.speed = plain_speed * factor
        self.rate = self.speed / natural if natural > 0 else 1.0

    def find_crawl_action(self):
        """
        The crawl cycle of the model: the ccrawl actions of the walk type (the game chooses the gait by the
        speed of the movement, the remake finds the name of the cycle).
        """
        database = getattr(self.figure, 'database', None)
        if database is None:
            return None
        for action in database.actions:
            if action.name.lower().startswith('ccrawl') and action.type == actions.TYPE_WALK and action.is_cycle:
                return action
        return None

    def go_to(self, x, y, run=False, gait=None):
        """
        Walks or runs to the point by the path of the grid of the map (engine.pathing) when the unit has one,
        straight otherwise. False when there is no path.
        """
        if self.is_frozen:
            return False
        if self.path_grid is None or self.path_class is None:
            self.move_to(x, y, run=run, gait=gait)
            return True
        points = self.path_grid.find_path(self.path_class, (self.x, self.y), (x, y), direction=self.direction())
        if not points:
            return False
        self.move_to(points[0][0], points[0][1], run=run, gait=gait)
        self.waypoints = list(points[1:])
        return True

    def natural_speed(self, action):
        """
        Map units per tick of the clip played at one frame per tick (535850).
        """
        if action is None or not action.movement_speed:
            return 0.0
        return action.movement_speed * self.figure.scale / CLIP_SPEED_DIVISOR

    def direction(self):
        """
        The number of the direction of engine.pathing in which the unit looks (the forward of the models is -y).
        """
        return pathing.direction_number(math.atan2(-math.cos(self.yaw), math.sin(self.yaw)))

    def half_size(self):
        """
        The larger half size of the figure on the map (+0x144 and +0x148 of the unit, see engine.combat.unit_radius).
        """
        size = self.size()
        if size is None:
            return 0.5
        return float(max(size[0], size[1])) * 0.5

    def height(self):
        """
        The height of the figure (+0x14c of the unit), the arrows fly at its half (engine.missiles).
        """
        size = self.size()
        return 1.0 if size is None else float(size[2])

    def size(self):
        """
        The sizes of the box of the figure in its pose of rest, None for a figure without parts.
        """
        if self._bounds is None:
            corners = [c for _, part_corners in pathing.object_parts(self.figure, np.identity(4)) for c in part_corners]
            if corners:
                points = np.asarray(corners)
                self._bounds = points.max(axis=0) - points.min(axis=0)
            else:
                self._bounds = False
        return self._bounds if self._bounds is not False else None

    def set_stats(self, unit_stats):
        """
        Takes the speeds of engine.stats.UnitStats.
        """
        if unit_stats is None:
            return
        self.speeds = unit_stats.speeds
        self.speed_scale = unit_stats.speed_scale
        self.base_senses = unit_stats.senses
        self.base_detections = unit_stats.detections
        self.view_cos = unit_stats.view_cos
        self.peripheral = unit_stats.peripheral
        self.refresh_senses()
        self.attack_range = unit_stats.attack_range

    def refresh_senses(self):
        senses, detections = magic.senses(self.base_senses, self.base_detections, self.fighter)
        if self.hero_stats is not None and len(senses) > SENSE_INFRA:
            # 524560: the perk of the night vision (group 15) adds its power in percents to the infra (+0xbc)
            power = self.hero_stats.perk_power(experience.GROUP_NIGHT_VISION)
            senses = tuple(value * (1.0 + power * experience.PERCENT) if index == SENSE_INFRA else value
                           for index, value in enumerate(senses))
        if senses:
            senses = (senses[0] * self.sight_light(senses), ) + tuple(senses[1:])
        if self.hero_stats is not None and senses:
            # the awareness skill of a hero or a mercenary sharpens the sight and the hearing
            awareness = 1.0 + self.hero_stats.skill(experience.SKILL_AWARENESS) * experience.PERCENT
            senses = tuple(value * awareness if index in (0, SENSE_HEARING) else value
                           for index, value in enumerate(senses))
        self.senses, self.detections = senses, detections

    def sight_light(self, senses=None):
        """
        CUnitServer::vf45 5536b0 (partly), +0x904: the sight is taken by (1 - d + d * infra / 100) of the darkness d,
        times 0.9 while the rain or the snow falls (58ccd0).
        """
        senses = self.senses if senses is None else senses
        infra = senses[SENSE_INFRA] if len(senses) > SENSE_INFRA else 0.0
        d = self.darkness
        return (1.0 - d + d * infra * SIGHT_INFRA_SCALE) * (WET_SIGHT if self.wet else 1.0)

    def set_light(self, darkness, wet=False):
        darkness = float(darkness)
        if darkness != self.darkness or bool(wet) != self.wet:
            self.darkness = darkness
            self.wet = bool(wet)
            self.refresh_senses()

    def detection_factor(self):
        """
        The stealth skill of a hero or a mercenary: the sight, the hearing and the other senses of an enemy
        find it with this factor (the remake's reading of SKILL_STEP).
        """
        if self.hero_stats is None:
            return 1.0
        stealth = self.hero_stats.skill(experience.SKILL_STEALTH) * experience.PERCENT
        return max(0.1, 1.0 - stealth)

    def hearing_noise(self):
        """
        5536b0 (partly), +0x8f8: the noise of the unit for the hearing of engine.ai, only while it goes somewhere.
        """
        if self.is_dead:
            return 0.0
        return {MODE_CRAWL: NOISE_CRAWL, MODE_SNEAK: NOISE_SNEAK, MODE_WALK: NOISE_WALK,
                MODE_RUN: NOISE_RUN}.get(self.mode, 0.0)

    def hearing_factor(self):
        """
        5536b0 (partly), +0x8fc: a running unit hears the others worse.
        """
        return RUN_HEARING if self.mode == MODE_RUN else 1.0

    def visibility(self):
        """
        5536b0 (partly), +0x900: a unit which crawls or sneaks is harder to see.
        """
        return {stats.SPEED_CRAWL: VISIBILITY_CRAWL, stats.SPEED_SNEAK: VISIBILITY_SNEAK}.get(self.gait, 1.0)

    def carried_weight(self):
        """
        The weight of the worn weapon and armors (engine.equipment.Item.weight, 6a5d20).
        """
        if self.fighter is None:
            return 0.0
        items = ([self.fighter.weapon_item] if self.fighter.weapon_item is not None else []) + \
                list(self.fighter.armor_items.values())
        return sum(float(getattr(item, 'weight', 0.0)) for item in items)

    def carry_limit(self):
        """
        524560: the most weight (+0x54 of the stats) the unit carries and still runs.
        """
        if self.hero_stats is None:
            return NO_CARRY_LIMIT
        lift = self.hero_stats.perk_power(GROUP_LIFT)
        return self.hero_stats.strength * CARRY_PER_STRENGTH * (lift * experience.PERCENT + 1.0)

    def run_limit(self):
        """
        CUnit::RunLimit 5241c0: RUN_OVERLOAD when the things weigh more than carry_limit, RUN_LEGS when a leg is
        below DamageLevelRunLimit, RUN_FREE otherwise. game.exe neither slows an overloaded unit.
        """
        if self.carry_limit() < self.carried_weight():
            return RUN_OVERLOAD
        if not self.wounds.can_run(self.legs_part):
            return RUN_LEGS
        return RUN_FREE

    def refuse_run(self, reason):
        """
        548520 (partly): the unit of the player which was told to run says once why it walks.
        """
        if reason == self.run_refusal:
            return
        self.run_refusal = reason
        if reason != RUN_FREE and self.on_cant_run is not None:
            self.on_cant_run(self, reason)

    @property
    def is_frozen(self):
        return magic.is_frozen(self.fighter)

    def take_spell(self, spell, caster):
        """
        67f830 and 58e670: a spell without damage comes to the unit.
        """
        if self.is_dead:
            return
        hurt = self.fighter is not None and self.fighter.health < self.fighter.max_health
        healed = magic.apply(spell, self)
        if self.on_spell_emitter is not None:
            emitter = magic.unit_particles(spell, self, rng=self.rng)
            if emitter is not None:
                self.on_spell_emitter(emitter, self)
        # 67f830 cases 24, 28: the sound at the unit (the healing of a whole unit by the spread is silent); the spells
        # of particles sound with their effect (engine.magic.SpellParticles)
        if self.on_spell_sound is not None and spell.index not in magic.PARTICLE_SPELLS and \
                (spell.index != magic.HEALING_SPELL or hurt or caster is self):
            sound = magic.start_sound(spell, (self.x, self.y, self.z))
            if sound is not None:
                self.on_spell_sound(sound)
        if spell.enchants():
            self.refresh_senses()
            if self.is_frozen:
                self.stop()
                self.strike_action = None
            if self.fighter is not None and magic.is_charmed(self.fighter):
                self.foe = None
                self.target = None
                self.strike_action = None
                self.mode = MODE_IDLE
        if _Debug:
            print('%s takes %s, healed %.2f' % (self, spell, healed))
        # a trap (engine.trap) casts without a unit: there is no one to answer
        if spell.provokes() and self.on_attacked is not None and getattr(caster, 'fighter', None) is not None:
            self.on_attacked(self, caster)

    def on_clip_end(self, clip):
        if self.mode == MODE_DEAD:
            return
        if self.mode == MODE_ATTACK and self.target is None:
            if self.strike_action is not None:
                self.resolve_blow()
            if self.foe is not None and not self.foe.is_dead:
                self.chase()
            else:
                self.idle()
            return
        if self.mode == MODE_ACTION:
            self.mode = MODE_IDLE
            self.play_action(self.find_action(actions.TYPE_IDLE), queue=False)
        elif self.mode == MODE_IDLE and clip.is_cycle:
            # the server picks the next idle by the probabilities when one ends
            action = self.find_action(actions.TYPE_IDLE)
            if action is not None and action is not clip.action:
                self.play_action(action, queue=True)

    def future_position(self, ticks):
        """
        GetFutureX / GetFutureY: where the unit will be after the ticks. game.exe asks the object (vf 0xd0 with
        the tick of the world); the remake goes along its way at its speed, a frozen unit stays.
        """
        x, y = self.x, self.y
        if self.target is None or self.is_frozen:
            return x, y
        left = self.speed * max(ticks, 0.0)
        for px, py in [self.target] + list(self.waypoints):
            distance = math.hypot(px - x, py - y)
            if distance >= left:
                if distance > 0.0:
                    x += (px - x) / distance * left
                    y += (py - y) / distance * left
                return x, y
            left -= distance
            x, y = px, py
        return x, y

    def update(self, dt_ms, clock):
        dt_ms = min(dt_ms, timing.MAX_STEP_MS)
        rate = self.rate
        if self.fighter is not None and self.fighter.tick_enchantments(dt_ms / clock.tick_ms):
            self.refresh_senses()
        self.regenerate(dt_ms / clock.tick_ms)
        if self.strike_wait > 0:
            self.strike_wait -= dt_ms / clock.tick_ms
        if self.waiting and self.mode == MODE_ATTACK and self.foe is not None and not self.is_frozen:
            if self.foe.is_dead:
                self.idle()
            elif self.strike_wait <= 0 or not self.can_strike(self.foe):
                self.chase()
            else:
                self.target_yaw = math.atan2(self.foe.x - self.x, -(self.foe.y - self.y))
        frozen = self.is_frozen
        if self.mode == MODE_RUN and self.target is not None and self.run_limit() != RUN_FREE:
            # 548520 (partly): a running unit hurt in the legs or loaded too much goes on walking
            waypoints = list(self.waypoints)
            self.move_to(self.target[0], self.target[1], gait=stats.SPEED_RUN)
            self.waypoints = waypoints
        if self.target is not None and not frozen:
            dx, dy = self.target[0] - self.x, self.target[1] - self.y
            distance = math.hypot(dx, dy)
            step = self.speed * dt_ms / clock.tick_ms
            if self.mode == MODE_ATTACK and self.foe is not None:
                # a long frame must not carry the unit past the distance of the strike
                step = min(step, max(0.0, self.distance_to(self.foe) - self.strike_distance(self.foe)))
            if distance <= max(step, ARRIVE_DISTANCE):
                self.x, self.y = self.target
                if self.waypoints:
                    self.target = self.waypoints.pop(0)
                elif self.mode == MODE_ATTACK:
                    self.target = None
                    self.chase()
                else:
                    self.idle()
            else:
                self.target_yaw = math.atan2(dx, -dy)
                self.x += dx / distance * step
                self.y += dy / distance * step
        if self.mode == MODE_ATTACK and self.target is not None and self.foe is not None and not frozen:
            self.chase_ticks -= dt_ms / clock.tick_ms
            if self.foe.is_dead:
                self.idle()
            elif self.chase_ticks <= 0 or self.can_strike(self.foe):
                self.chase()
        self.turn(dt_ms / clock.tick_ms)
        self.z = self.ground_height()
        self.player.add_time(clock.frames(dt_ms) * rate)
        self.update_sounds()
        strike = self.strike_action
        if frozen:
            return
        if strike is not None and self.player.clip is not None and self.player.clip.action is strike:
            if self.player.time >= strike.hit_frame:
                self.resolve_blow()

    def turn(self, dt_ticks):
        diff = (self.target_yaw - self.yaw + math.pi) % (2.0 * math.pi) - math.pi
        k = math.exp(-TURN_RATE * dt_ticks)
        self.yaw = self.target_yaw - diff * k

    def pose(self):
        return self.figure.skeleton.pose(self.player.tracks, self.player.time)

    def world_matrix(self):
        """
        4x4 transform of the figure: the forward of the models is -y, the yaw turns it around z.
        """
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        m = np.identity(4)
        m[0:2, 0:2] = [[c, -s], [s, c]]
        m[0:3, 3] = (self.x, self.y, self.z)
        return m

#------------------------------------------------------------------------------


def darkness(hour):
    """
    55fc50: the darkness 0..1 of the hour of the day for the sight of the units.
    """
    hour = float(hour) % 24.0
    if hour > DARK_EVENING:
        return (hour - DARK_EVENING) * DARK_PER_HOUR
    if hour < DARK_MORNING:
        return 1.0
    return 0.0


def make_spell_effect(caster, spell, target):
    """
    58e670: the effect of the spell at one of its targets. The caster is a unit or another thing which casts
    (engine.trap.MagicTrap): its point, land_height, rng, is_hostile, height, deliver_blow and the callbacks
    on_spell_effect and on_shoot.
    """
    if spell.index == magic.TELEPORT_SPELL:
        if caster.on_spell_effect is not None:
            caster.on_spell_effect(magic.Teleport(spell, caster, (target.x, target.y), ground=caster.land_height,
                                                can_stand=can_stand, rng=caster.rng))
        return
    if spell.index in magic.AREA_SPELLS:
        if caster.on_spell_effect is not None:
            caster.on_spell_effect(magic.AreaEffect(spell, caster, (caster.x, caster.y), (target.x, target.y),
                                                  ground=caster.land_height, rng=caster.rng))
        return
    if spell.index == magic.POSSESSION_SPELL and caster.on_spell_effect is not None and \
            getattr(target, 'fighter', None) is not None:
        # 682610 case 0x15: the player of the caster looks through the target while it lasts
        caster.on_spell_effect(magic.Possession(spell, caster, target))
    if spell.index in magic.PARTICLE_SPELLS and caster.on_spell_effect is not None:
        caster.on_spell_effect(magic.SpellParticles(spell, caster, target, rng=caster.rng,
                                                    ground=caster.land_height))
    if target.fighter is None:
        return
    if not spell.affects(target.figure.model_name, caster.is_hostile(target)):
        return
    if not spell.hurts():
        target.take_spell(spell, caster)
        return
    hit = spell.hit()
    # SpellEffect::MakeMissile (67f130): the effect 0x4e carries the blow and homes on the target (67ac30)
    if spell.sends_missile() and caster.on_shoot is not None:
        position = (caster.x, caster.y, caster.z + caster.height() * 0.5)
        missile = missiles.Arrow(caster, target, hit, position, ground_height=caster.land_height,
                                 step=magic.MISSILE_STEP, model_name=None)
        missile.trail, missile.flash = magic.missile_particles(spell, position, rng=caster.rng)
        missile.spell = spell
        # 67f830 cases 0, 2: the sound of the cast, the one of the flight goes with the missile
        missile.sounds = [s for s in (magic.start_sound(spell, position), ) if s is not None]
        missile.loop = magic.circle_sound(spell, position)
        caster.on_shoot(missile)
        return
    if spell.index in magic.INSTANT_SPELLS and caster.on_spell_effect is not None:
        # 67f3d0: the arc of the lightning to the target
        caster.on_spell_effect(magic.LightningArc(spell, caster, target, rng=caster.rng))
    caster.deliver_blow(target, hit)


def can_stand(unit, x, y):
    """
    5b7a40, the remake's reading: the point is open for the class of the unit on the grid of the map.
    """
    grid = unit.path_grid
    if grid is None or unit.path_class is None:
        return True
    return grid.is_open(unit.path_class, *grid.point(x, y))


def push_out(unit):
    """
    5ba9d0 after an object came onto the place of the unit (SetCP, a lever): a unit which cannot stand
    there goes to the nearest open point at once. True when it was moved.
    """
    grid = unit.path_grid
    if grid is None or unit.path_class is None or can_stand(unit, unit.x, unit.y):
        return False
    point = grid.nearest_open(unit.path_class, unit.x, unit.y)
    if point is None:
        return False
    unit.teleport_to(*point)
    return True


def update_spell_effects(effects, units, ticks):
    """
    The effects of the spells on the ground run their ticks and give their blows; the ones which are over
    go away.
    """
    active = []
    for effect in effects:
        for target, attacker in effect.update(ticks, units):
            target.take_spell_blow(effect.hit, effect.caster, attacker)
        if not effect.finished:
            active.append(effect)
    return active
