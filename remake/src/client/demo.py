"""
The demo of the remake (make demo): the game as game.exe starts it, with what the remake already has.

It opens the main menu; "New game" takes the hero to the first map among its monsters. The hero is
dressed, holds a weapon and knows the spells made so far; the keys of the demo (F1 shows them) change the
weapon, choose the spell and cast it. What is not made yet answers with "not made yet" or is left out.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from eilib import mapdata

from engine import combat
from engine import experience
from engine import magic

#------------------------------------------------------------------------------

# the zone of the start of the game in map.txt (the map zone1) and its entrance "one-way entrance in the beginning
# of the game": the hero appears in the middle of its deploy rectangle as game.exe places the party (590700)
DEMO_ZONE = 'gz1g'
START_EXIT = 1
HERO_MODEL = 'unhuma'
HERO_PROTOTYPE = 'Human Hero'
HERO_ARMOR = ('gipat medium plate.thin', 'gipat medium helm.thin', 'gipat medium leggins.thin',
              'gipat medium boots.thin', 'gipat medium gloves.thin')
# the weapons which W goes through: (the name shown, the items)
KITS = (
    ('sword', ('long sword.steel',)),
    ('axe', ('battle axe.steel',)),
    ('spear', ('spear.steel',)),
    ('bow', ('long bow.steel',)),
    ('crossbow', ('crossbow.steel',)),
    # without items the hero takes the weapon of its prototype (the cheat dagger of Human Hero)
    ('the dagger of the prototype', ()),
)
# the spells of the book of the demo, the ones the remake makes; Z and X go through them
SPELLS = (
    'Fire Arrow', 'Lightning Arc', 'Acid Ray', 'Fireball', 'Invoke Lightning', 'Column of Acid',
    'Wall of Fire', 'Smtg Lightning', 'Acid Fog', 'Campfire', 'Rick Magic', 'Curse Magic',
    'Healing', 'Protection from Fire', 'Protection from Lightning', 'Protection from Acid', 'Strengthen',
    'Weaken', 'Stun', 'Eagle Sight', 'Infravision', 'Invisibility', 'Speed', 'Slow', 'Teleport',
)
# the hero of the demo knows the skills of the schools for all the spells; the mana is as in the original
DEMO_MANA = 50.0
DEMO_SCHOOL_SKILL = 50
DEMO_SCHOOL_SKILLS = (experience.SKILL_ELEMENTAL, experience.SKILL_SENSE, experience.SKILL_ASTRAL)
HELP_MS = 15000.0

HELP_LINES = (
    'DEMO KEYS (F1 shows them again)',
    'click - walk, Shift + click - run, click on a monster - attack it',
    'Ctrl + click - cast the spell at a monster or, for the spells of a point, at the ground',
    'Z / X - the previous / next spell, C - cast it at the hero, M - full mana and health',
    'W - the next weapon: sword, axe, spear, bow, crossbow, dagger',
    'I - the items, Space - pause, F - the fast clock, [ ] - the hour, T - the time runs fast',
    'arrows - move the camera, Q/E - turn it, wheel - zoom, H - the debug text, Esc - the menu',
)

#------------------------------------------------------------------------------


class Demo(object):
    """
    The state of the demo over the client: the weapon of the hero and the spell of its book.
    """

    def __init__(self, app):
        self.app = app
        self.kit_index = 0
        self.spell_index = 0
        self.spells = []

    def hero_items(self):
        return list(HERO_ARMOR) + list(KITS[self.kit_index][1])

    def start(self):
        """
        "New game": the map of the demo with the hero.
        """
        app = self.app
        zone = [z for z in app.game_data.zones() if z.name == DEMO_ZONE][0]
        start = zone.exit(START_EXIT)
        app.new_scene()
        width, height = (n * mapdata.SECTOR_CELLS for n in zone.sectors)
        app.setup_map(zone.mpr_name, position=start.deploy_center, model_name=HERO_MODEL,
                      prototype_name=HERO_PROTOTYPE, item_names=self.hero_items(),
                      yaw=start.deploy_yaw(width, height), camera_yaw=start.camera_yaw,
                      mob_name=zone.mob_name)
        book = app.game_data.spells()
        self.spells = [s for s in (magic.make_spell(book, name) for name in SPELLS) if s is not None]
        self.refill()
        self.choose_spell(0, announce=False)
        if not app.tutorial_open():
            self.show_help()

    def refill(self):
        fighter = self.app.unit.fighter if self.app.unit is not None else None
        if fighter is None:
            return
        hero_stats = self.app.unit.hero_stats
        if hero_stats is not None:
            for index in DEMO_SCHOOL_SKILLS:
                hero_stats.skills[index] = max(hero_stats.skills[index], DEMO_SCHOOL_SKILL)
        fighter.max_mana = max(fighter.max_mana, DEMO_MANA)
        fighter.mana = fighter.max_mana
        for part in fighter.parts:
            part.health = part.max_health
            part.state = combat.PART_WHOLE

    def choose_spell(self, step, announce=True):
        if not self.spells:
            return
        self.spell_index = (self.spell_index + step) % len(self.spells)
        spell = self.spells[self.spell_index]
        self.app.spell = spell
        if not announce:
            return
        if spell.flags & magic.TARGET_POINT:
            how = 'Ctrl + click on the ground or a monster'
        else:
            how = 'Ctrl + click on a monster, C - at the hero'
        self.app.show_message('spell %d/%d: %s (%s)' % (self.spell_index + 1, len(self.spells), spell.name, how))

    def next_kit(self):
        self.kit_index = (self.kit_index + 1) % len(KITS)
        self.app.replace_hero(self.hero_items())
        self.app.show_message('weapon: %s' % KITS[self.kit_index][0])

    def cast_at_hero(self):
        app = self.app
        if app.unit is None or app.spell is None:
            return
        if not app.unit.cast(app.unit, app.spell):
            app.show_message('%s can not be cast now' % app.spell.name)

    def show_help(self):
        self.app.show_help('\n'.join(HELP_LINES), HELP_MS)

    def on_key(self, name):
        """
        The keys of the demo while the hero is on the map, True when the key was taken.
        """
        if self.app.unit is None or self.app.battle is None:
            return False
        if name == 'f1':
            self.show_help()
        elif name == 'z':
            self.choose_spell(-1)
        elif name == 'x':
            self.choose_spell(1)
        elif name == 'c':
            self.cast_at_hero()
        elif name == 'w':
            self.next_kit()
        elif name == 'm':
            self.refill()
            self.app.show_message('full mana and health')
        else:
            return False
        return True
