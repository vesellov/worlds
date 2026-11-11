"""
Reader of the tagged databases of the game (items.idb and others in database.res).

A node is a tag byte, a length and the data. The length is one byte when its lowest bit is 0,
otherwise it is the dword starting at that byte; the real length is the value shifted right by one
(game.exe 409070). A section holds records, a record holds fields, records with the same tag
are told apart by their order.

items.idb, as read by game.exe 6a6760:
    section 1  materials      (record readers 69f5e0)
    section 2  weapon shapes  (69f6c0 + 69f7c0)
    section 3  armor shapes   (69f6c0 + 69f880)
    section 4  quick items (potions, scrolls), not read here yet
    section 5  quest items: the name (tag 0), the index (5), the number (21, +0x4c of the item, the scripts
               and the keys of the levers use it), the zones (22)
    section 6  spells, not read here yet

units.udb, as read by game.exe 53a200:
    section 1  hit locations    (none, skull, torso, leg, arm)
    section 2  unit types       (0x168 bytes: name, class, model name in the tag 30, the tags 12..17 are
                                 the 6 hit locations of the body: name, weight, part of the health;
                                 the tag 5 is the locomotion: 1 walks, 2 flies, 3 stands; the tag 7 holds
                                 4 floats: the speeds of the run, the walk, the sneak and the crawl;
                                 the tags 9 and 10 are the classes of the paths of a standing and of
                                 a lying unit, 0..7, see engine.pathing; the tag 11 holds the parts of
                                 the 7 types of the damage of the unit, the tag 39 the factors of
                                 its absorption against them, see engine.combat)
    section 3  unit prototypes  (0x120 bytes, 52d740: skin +0xc, hair +0x10, health, mana and absorption
                                 in the tags 9..11, to hit and parry 15, 16, damage 19, 20, weapons +0x118 and +0x11c)
    section 4  hero templates   (0x60 bytes, 52da60, found by the name of the prototype in 5145d0): the experience,
                                 the strength, the dexterity, the intelligence, the skills and the perks of a hero
                                 or a mercenary made of the prototype (5136d0), see engine.experience
perks.pdb, as read by game.exe 520220:
    section 1  skills  (0x18 bytes, the table 7af258): the name, the code, the number of the skill and the code
                       of the attribute which adds to it ("dex", "int")
    section 2  perks   (0x44 bytes, the table 7af264, 51f9e0): the name, the code, the code of the perk before it,
                       the skill, the level of the skill needed (tag 18, +0x30) and the power (tag 19, +0x34);
                       the perks go by threes, the levels 1..3 of a group (index // 3)
spells.sdb, as read by game.exe 684510 (the tags of the fields are the order of the format of EIrepack):
    section 1  spell prototypes  (0x84 bytes, 683b40)
    section 2  modifiers         (0x24 bytes, 683ab0)
    sections 3, 4, 5  the templates of the spells of the scrolls, of the armors and of the weapons
                      (0x34 bytes, 683d00)
acks.db, as read by game.exe 6d75e0 (0x1c4 bytes a record of the section 1):
    section 1  the phrases of the units by the name of their prototype (tag 1): the tags 10..44 are the groups
               (10 selected, 11 move, 12 attack, ... 41 the phrases of the scenario which "say" of the scripts
               finds by their name, 6d8a00), every phrase is a record: the .wav of sfx.res (tag 1), the chance
               in percents (2), the Russian text (3) and the name (4)
    sections 2 and 3  the phrases of the kills and of the trade, not read here yet
The names of the fields of the prototypes are those of EIrepack (Formats/db.py, MIT license, (c) 2018 Asp Adm).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

#------------------------------------------------------------------------------

ENCODING = 'cp1251'

SECTION_MATERIALS = 1
SECTION_WEAPONS = 2
SECTION_ARMORS = 3
SECTION_QUEST_ITEMS = 5

SECTION_SPELLS = 1
SECTION_SPELL_MODIFIERS = 2
SECTION_SPELL_TEMPLATES = 3
SECTION_ARMOR_SPELL_TEMPLATES = 4
SECTION_WEAPON_SPELL_TEMPLATES = 5

ACKS_FIRST_GROUP = 10
ACKS_SCENARIO = 41

SECTION_UNIT_TYPES = 2
SECTION_UNIT_PROTOTYPES = 3
SECTION_HERO_TEMPLATES = 4

SECTION_SKILLS = 1
SECTION_PERKS = 2

# the levels of a group of perks
PERK_LEVELS = 3

# the bytes of the skills of a hero template: the tag 6 (+0x18, +0x6c8 of the stats) and the tag 7 (+0x1b, +0x6cb)
TEMPLATE_SCIENCE_COUNT = 3
TEMPLATE_SKILLS_COUNT = 8

# the hit locations of a unit type: head, torso, two arms, two legs, in the order of the information window
HIT_LOCATION_TAGS = (12, 13, 14, 15, 16, 17)

# the class of the unit type which wears armor (game.exe 5141f0 creates the armors only for it)
UNIT_CLASS_HUMANOID = 0x32

# CItemWeapon type (+0xc of a weapon shape), the order of the switch in game.exe 6a0fb0
WEAPON_SWORD = 0
WEAPON_AXE = 1
WEAPON_DAGGER = 2
WEAPON_PIKE = 3
WEAPON_CLUB = 4
WEAPON_BOW = 5
WEAPON_CROSSBOW = 6

# CItemArmor slot (+0xc of an armor shape), the codes are the texture kinds of game.exe 6a3200
ARMOR_HELM = 0
ARMOR_PLATE = 1
ARMOR_LEGGINS = 2
ARMOR_SHIRT = 3
ARMOR_PANTS = 4
ARMOR_BOOTS = 5
ARMOR_GLOVES = 6

WEAPON_KINDS = ('sw', 'ax', 'dg', 'sp', 'hm', 'bw', 'cb')

# the damage of a unit is split into 7 types (game.exe 5254b0)
DAMAGE_TYPES_COUNT = 7

SENSES_COUNT = 6
# the target of a spell, a character code (+0x30 of a spell prototype)
SPELL_TARGET_UNIT = 'u'
SPELL_TARGET_POINT = 't'
SPELL_TARGET_SELF = 's'

# the kinds of the modifiers (+0xc of a modifier) and the flags of a prototype which allow them (tag 20)
MODIFIER_RANGE = 0
MODIFIER_TARGETS = 1
MODIFIER_AREA = 2
MODIFIER_EFFECT = 3
MODIFIER_DURATION = 4
MODIFIER_MANA = 5
MODIFIER_FILTER = 6
MODIFIER_ITEM = 7
SPELL_MODIFIER_FLAGS_COUNT = 5

ARMOR_KINDS = ('hl', 'pl', 'lg', 'sh', 'pt', 'bt', 'gl')

#------------------------------------------------------------------------------


class Node(object):

    def __init__(self, tag, data):
        self.tag = tag
        self.data = data

    def __repr__(self):
        return 'Node(%d, %d bytes)' % (self.tag, len(self.data))

    def children(self):
        return list(iter_nodes(self.data))

    def fields(self):
        """
        tag -> data of the first field with that tag.
        """
        result = {}
        for node in iter_nodes(self.data):
            result.setdefault(node.tag, node.data)
        return result


class Material(object):
    """
    A record of the section 1 of items.idb (CMaterial::Serialize 69f5e0).
    """

    def __init__(self, index, fields):
        self.index = index
        self.name = _text(fields, 0)
        self.kind = _text(fields, 1)
        # the code in the names of the textures: "le" in unhumagl_01.le.0
        self.code = _text(fields, 2)
        # +0x10: the price of a unit of the material (tag 4), an item takes components units (6a5d20)
        self.price = _float(fields, 4)
        # +0x14, "weight" of EIrepack: the weight of an item is that of its shape times this (6a5d20)
        self.weight_factor = _float(fields, 5)
        # +0x20: the durability of an item is that of its shape times this (6a5d20)
        self.durability = _float(fields, 8)
        # +0x28: the damage of a weapon is that of its shape times this
        self.damage_factor = _float(fields, 10)
        # +0x2c: the absorption of an armor against the 7 types of the damage is that of its shape times these
        self.absorption_factors = _floats(fields, 11, DAMAGE_TYPES_COUNT)

    def __repr__(self):
        return 'Material(%r, %r)' % (self.name, self.code)


class Shape(object):
    """
    The common part of the weapon and armor shapes (game.exe 69f6c0).
    """

    def __init__(self, index, fields):
        self.index = index
        self.name = _text(fields, 0)
        self.description = _text(fields, 1)
        self.type = _dword(fields, 2)
        self.model_text = _text(fields, 3)
        # +0x18, the number of the part and of the texture: "05" in bd.armor05 and unhumapl_05.br.0
        self.number = _dword(fields, 5)
        # +0x1c, the last number of the texture name
        self.variant = _dword(fields, 6)
        # +0x34: the price of the shape (tag 7 of EIrepack); a material adds its own price to it
        self.price = _float(fields, 7)
        # +0x38, "weight" of EIrepack
        self.weight = _float(fields, 8)
        # +0x48: the durability of an item before the factor of its material (6a5d20)
        self.durability = _float(fields, 12)
        # +0x4c: how many units of the material an item of the shape needs (6a5d20 reads a dword)
        self.components = _dword(fields, 13)

    def __repr__(self):
        return '%s(%r, type %d, %02d)' % (self.__class__.__name__, self.name, self.type, self.number)


class WeaponShape(Shape):
    """
    A record of the section 2 of items.idb (CWeaponShape::Serialize 69f7c0 after the common fields of 69f6c0).
    """

    def __init__(self, index, fields):
        Shape.__init__(self, index, fields)
        # the tag 23, "range" of EIrepack: 12 for a short bow, 16 for a long bow and a crossbow,
        # 0 for a weapon of the hand (the unit keeps its reach)
        self.range = _float(fields, 23)
        # +0x90 and +0x94: the damage is min + spread * a random part of 1 (6a2da0), "min damage" and "max damage"
        # of EIrepack; both are multiplied by the damage factor of the material
        self.damage_min = _float(fields, 24)
        self.damage_spread = _float(fields, 25)
        # +0x98: the parts of the 7 types of the damage
        self.damage_types = _floats(fields, 26, DAMAGE_TYPES_COUNT)
        # +0x84 and +0x88: the attack and the parry of the weapon (6a2d00, 6a2d50), a hero adds its skill
        # (vf50 555a00): +10 parry of a long sword, +10 attack of a dagger, -10 parry of a battle axe
        self.attack = _float(fields, 0x1d)
        self.defense = _float(fields, 0x1e)
        # +0x7b: the bonuses to the skills of the hero (526c70 adds the melee +0x7b or the archery +0x7c)
        self.skill_bonuses = _bytes(fields, 0x1c, TEMPLATE_SKILLS_COUNT)
        # +0x6c: the ticks between the strikes with the weapon (54d960): 27 a dagger, 30 a sword or a spear,
        # 33 an axe, 36 a mace, 40 a bow or a crossbow
        self.strike_delay = _dword(fields, 0x15)


class ArmorShape(Shape):
    """
    A record of the section 3 of items.idb (CArmorShape::Serialize 69f880 after the common fields of 69f6c0).
    """

    def __init__(self, index, fields):
        Shape.__init__(self, index, fields)
        # +0xb8, the plate and the leggins are drawn on the skin in the order of this value (game.exe 536670)
        self.layer = _dword(fields, 0x18)
        # +0xb4: the wounds of the parts under the armor stay seen (game.exe 536ea0), else the armor hides them
        self.shows_wounds = bool(_bytes(fields, 0x17, 1)[0])
        # +0x6c and +0x90, "absortion" of EIrepack twice: a scale, the 7 values of the types, the flags.
        # The first is for the first hit location of the armor, the second for the others (5230c0)
        self.absorption = _armor_values(fields, 21)
        self.limb_absorption = _armor_values(fields, 22)


class ItemDatabase(object):

    def __init__(self, materials, weapons, armors, quest_items=()):
        self.materials = materials
        self.weapons = weapons
        self.armors = armors
        self.quest_items = list(quest_items)
        self._quest_items = {q.name.lower(): q for q in self.quest_items}
        self._materials = {m.name.lower(): m for m in materials}
        self._shapes = {}
        for shape in weapons + armors:
            self._shapes.setdefault(shape.name.lower(), shape)

    def material(self, name):
        return self._materials.get(name.lower())

    def shape(self, name):
        return self._shapes.get(name.lower())

    def quest_item(self, name):
        """
        The QuestItem with the name (any case), None when there is none.
        """
        return self._quest_items.get(name.lower())

    def item(self, item_name):
        """
        (shape, material) of an item named like the maps do: "hadagan low leggins.thick",
        (None, None) for an unknown item.
        """
        shape_name, _, material_name = item_name.rpartition('.')
        if not shape_name:
            shape_name, material_name = material_name, ''
        shape = self.shape(shape_name)
        material = self.material(material_name) if material_name else None
        return shape, material

class QuestItem(object):
    """
    A record of the section 5 of items.idb (a CItem of the class 0x3009).
    """

    def __init__(self, fields):
        self.name = _text(fields, 0)
        self.index = _dword(fields, 5)
        self.number = _dword(fields, 21)
        self.zones = _texts(fields, 22)

    def __repr__(self):
        return 'QuestItem(%s, %d)' % (self.name, self.number)


class Quest(object):
    """
    A record of the section 1 of quests.qdb: the name ("q0g"), the experience of QuestComplete (tag 1,
    shared by the party in 5911c0), the comment of the editor (tag 4).
    """

    def __init__(self, fields):
        self.name = _text(fields, 0)
        self.experience = _float(fields, 1)
        self.comment = _text(fields, 4)

    def __repr__(self):
        return 'Quest(%s, %g)' % (self.name, self.experience)


class Phrase(object):

    def __init__(self, fields):
        self.sound_name = _text(fields, 1)
        self.chance = _dword(fields, 2)
        self.text = _text(fields, 3)
        self.name = _text(fields, 4)

    def __repr__(self):
        return 'Phrase(%s, %s)' % (self.name, self.sound_name)


class Acks(object):
    """
    The phrases of the units of one prototype (a record of the section 1 of acks.db, 0x1c4 bytes, read by
    6e3740: the name, then a list of phrases for every group tag).
    """

    def __init__(self, record):
        fields = record.fields()
        self.name = _text(fields, 1)
        # tag of the group -> [Phrase]
        self.groups = {}
        for node in record.children():
            if node.tag >= ACKS_FIRST_GROUP:
                self.groups[node.tag] = [Phrase(p.fields()) for p in node.children()]

    def __repr__(self):
        return 'Acks(%s, %d groups)' % (self.name, len(self.groups))

    def phrase(self, group, name):
        """
        The phrase of the group with the name (6d8a00 compares the names exactly), None when there is none.
        """
        for phrase in self.groups.get(group, ()):
            if phrase.name == name:
                return phrase
        return None


class UnitType(object):
    """
    A record of the section 2 of units.udb (CUnitType::Read 52dc70, 0x168 bytes).
    """

    def __init__(self, index, fields):
        self.index = index
        self.name = _text(fields, 0)
        self.unit_class = _dword(fields, 1)
        # the part of the maximum health and mana which comes back every 15 ticks (+8, +0xc; 5136d0, 523740)
        self.health_regen = _float(fields, 2)
        self.mana_regen = _float(fields, 3)
        self.model_name = _text(fields, 30).lower()
        self.locomotion = _dword(fields, 5)
        self.speeds = _floats(fields, 7, 4)
        self.path_class = _dword(fields, 9)
        self.lying_path_class = _dword(fields, 10)
        self.damage_types = _floats(fields, 11, DAMAGE_TYPES_COUNT)
        self.armor_factors = _floats(fields, 39, DAMAGE_TYPES_COUNT, default=1.0)
        # the folders of the steps in sfx.res by the type of the tile (tileDesc.reg), 16 of them (534590)
        self.step_paths = _texts(fields, 35)
        # the folder of the voice in sfx.res ("monsters\\ogre"), +0x124 of the type in game.exe
        self.sfx_path = _text(fields, 34)
        # the chances in percents of the idle and the attack voices, +0x150 and +0x154 (534760)
        self.idle_sound_chance = _dword(fields, 37)
        self.attack_sound_chance = _dword(fields, 38)
        # the skin ("Leather", "Stone"), +0x118: the material of the sound of a blow (537500)
        self.skin_material = _text(fields, 44)
        self.hit_locations = []
        for tag in HIT_LOCATION_TAGS:
            if tag in fields:
                self.hit_locations.append(HitLocation(Node(tag, fields[tag]).fields()))

    def __repr__(self):
        return 'UnitType(%r, %s, class %d)' % (self.name, self.model_name, self.unit_class)


class HitLocation(object):
    """
    A part of the body: its name (skull, torso, arm, leg), the bits of the types of the damage which cut it
    off at once (the tag 1, 525070 tests only the bits 0..6, the high bytes of the arms and legs hold junk),
    its weight (the tag 2: its share of the health of the unit, 1, 3, 1, 1, 2, 2
    for the head, the torso, the arms and the legs) and the part of the health of the unit which a lost part
    takes away (the tag 3: 1.01 for the head and the torso, so their loss kills, 0.34 for an arm,
    0.67 for a leg), see engine.combat.
    """

    def __init__(self, fields):
        self.name = _text(fields, 0)
        self.flags = _dword(fields, 1)
        self.weight = _float(fields, 2)
        self.health_part = _float(fields, 3)

    def __repr__(self):
        return 'HitLocation(%s, %.2f, %.2f)' % (self.name, self.weight, self.health_part)


class UnitPrototype(object):
    """
    A record of the section 3 of units.udb (CUnitPrototype::Read 52d740, 0x120 bytes).
    """

    def __init__(self, index, fields):
        self.index = index
        self.name = _text(fields, 0)
        self.type_name = _text(fields, 1)
        self.skin = _dword(fields, 3)
        # the number of the hair part hr.NN, -1 is a bald unit
        self.hair = _dword(fields, 4)
        self.health = _float(fields, 9)
        self.mana = _float(fields, 10)
        self.armor = _float(fields, 11)
        self.attack = _float(fields, 15)
        self.defense = _float(fields, 16)
        self.damage_min = _float(fields, 19)
        # the spread of the damage: min + 0..spread (game.exe 54f630), "damage max" in EIrepack
        self.damage_max = _float(fields, 20)
        # "tuning weapon weight" of EIrepack, the field 18 of UNIT_STATS (+0x260 of the unit of the server)
        self.weapon_weight = _float(fields, 17)
        # "attack range" of EIrepack, the field 15 of UNIT_STATS (+0x250): above 1 the unit shoots
        self.attack_range = _float(fields, 14)
        # the six senses (sight, infra, life, hearing, smell, tracking) and how well the unit is detected by them
        self.senses = _floats(fields, 25, SENSES_COUNT)
        self.detections = _floats(fields, 26, SENSES_COUNT, default=1.0)
        self.weapons = [w for w in (_text(fields, 0x21), _text(fields, 0x2a)) if w]
        # the folder of the voice which replaces the one of the unit type ("heroes\\Hero"), +0x34 (534760)
        self.sfx_path = _text(fields, 8)
        # "real weapon typeID" of EIrepack: the kind of the sound of the blow of a unit without a weapon
        # (9..15, 16 has no sound, see engine.sounds), +0x7c (537500)
        self.weapon_kind = _dword(fields, 39)
        # the experience for the kill of the unit, +0x54 (553f40 shares it among the party, 5911c0)
        self.kill_experience = _float(fields, 0x2b)
        # the ticks between the strikes without a weapon item, +0x5c (5136d0 copies it to +0x8c of the numbers,
        # 54d960): 50 for the humans, 68 for the grey wolves
        self.strike_delay = _float(fields, 12)
        # the multiplier of the speed, +0x60 (5136d0 copies it to +0x24 of the numbers, the field 4 of UNIT_STATS):
        # 1.0 for all the prototypes of the game
        self.speed_scale = _float(fields, 13) or 1.0

    def __repr__(self):
        return 'UnitPrototype(%r, %r, skin %d, hair %d)' % (self.name, self.type_name, self.skin, self.hair)


class HeroTemplate(object):
    """
    A record of the section 4 of units.udb (0x60 bytes, 52da60): what a hero or a mercenary of the prototype
    with the same name starts with (5136d0). The money and the last tags are not known yet.
    """

    def __init__(self, index, fields):
        self.index = index
        self.name = _text(fields, 0)
        # +8: the experience given by 523960 when the unit is made
        self.experience = _float(fields, 2)
        # +0xc, +0x10, +0x14
        self.strength = _float(fields, 3)
        self.dexterity = _float(fields, 4)
        self.intelligence = _float(fields, 5)
        # +0x18: three bytes for +0x6c8 of the stats, the first one is the science (Use/Steal)
        self.science = _bytes(fields, 6, TEMPLATE_SCIENCE_COUNT)
        # +0x1b: eight bytes for +0x6cb, the skills by their number of perks.pdb (melee, archery, backstab,
        # elemental, sense, astral)
        self.skills = _bytes(fields, 7, TEMPLATE_SKILLS_COUNT)
        # the codes of the perks given by 526450 ("bs1", "fire1")
        self.perk_codes = _texts(fields, 8)
        # the items which the shops of the camp offer to the unit, not used here yet
        self.weapon_names = _texts(fields, 9)
        self.potion_names = _texts(fields, 10)
        self.spell_names = _texts(fields, 11)
        # the tag 12: the most weight a hero carries, the tag 13: the percent of the speed at it, the tag 14:
        # a flag (1 for every hero template of the game; the fields are the same in all of them)
        self.max_weight = _float(fields, 12)
        self.weight_percent = _dword(fields, 13)
        self.template_flag = _bytes(fields, 14, 1)[0] if _bytes(fields, 14, 1) else 0

    def __repr__(self):
        return 'HeroTemplate(%r, exp %g, %g/%g/%g)' % (self.name, self.experience, self.strength, self.dexterity,
                                                       self.intelligence)


class Skill(object):
    """
    A record of the section 1 of perks.pdb (0x18 bytes, 51f3a0): "Melee", "melee", the number 0 and
    the attribute "dex".
    """

    def __init__(self, index, fields):
        self.index = index
        self.name = _text(fields, 0)
        self.code = _text(fields, 1)
        self.number = _dword(fields, 2)
        # a text inside a node, unlike the texts of the perks
        self.attribute = _first(_texts(fields, 10))

    def __repr__(self):
        return 'Skill(%r, %d, %s)' % (self.name, self.number, self.attribute)


class Perk(object):
    """
    A record of the section 2 of perks.pdb (0x44 bytes, 51f480). The index in the table gives the group (index // 3)
    and the level in it (index % 3 + 1); the stats of a hero keep the level of every group in a byte (+0x6d3).
    """

    def __init__(self, index, fields):
        self.index = index
        self.name = _text(fields, 0)
        self.code = _text(fields, 1)
        self.previous_code = _text(fields, 10)
        self.skill_code = _text(fields, 11)
        # +0x30: the level of the skill needed to take the perk
        self.required = _dword(fields, 18)
        # +0x34: the power, percents or points by the group
        self.power = _dword(fields, 19)

    def __repr__(self):
        return 'Perk(%r, %s, power %d)' % (self.name, self.code, self.power)

    @property
    def group(self):
        return self.index // PERK_LEVELS

    @property
    def level(self):
        return self.index % PERK_LEVELS + 1


class PerkDatabase(object):

    def __init__(self, skills, perks):
        self.skills = skills
        self.perks = perks
        self._codes = {p.code.lower(): p for p in perks}

    def perk(self, code):
        return self._codes.get((code or '').lower())

    def level_perk(self, group, level):
        """
        528270: the perk of a group at a level (1..3), None for the level 0 or outside the table.
        """
        if level <= 0:
            return None
        index = group * PERK_LEVELS - 1 + level
        return self.perks[index] if 0 <= index < len(self.perks) else None

    def power(self, group, level):
        perk = self.level_perk(group, level)
        return float(perk.power) if perk is not None else 0.0


class SpellPrototype(object):
    """
    A record of the section 1 of spells.sdb, the names of the fields are those of EIrepack.
    The offsets are those of the record of 0x84 bytes of game.exe (683b40).
    """

    def __init__(self, index, fields):
        self.index = index
        self.name = _text(fields, 0)
        # the code of the spell, the name of its figure and its texts ("arrow", "fireball")
        self.code = _text(fields, 1)
        # the school as a word ("fire", "divination") and as a number (+0x14), the class as a number (+0x10)
        self.subtype = _text(fields, 2)
        self.subtype_id = _dword(fields, 19)
        self.type_id = _dword(fields, 4)
        self.price = _float(fields, 3)
        self.mana = _float(fields, 5)
        self.slots = _dword(fields, 6)
        # the speed of the missile, -1 without one
        self.speed = _float(fields, 7)
        self.range = _float(fields, 8)
        self.area = _float(fields, 9)
        self.effect = _float(fields, 10)
        self.target = _char(fields, 11)
        self.targets = _dword(fields, 12)
        self.duration = _dword(fields, 13)
        self.actions = _dword(fields, 14)
        self.require_trace = _dword(fields, 15)
        self.complex = _dword(fields, 21)
        # the codes of the modifiers built into the spell and of those which only it may take
        self.buildin_modifiers = _codes(fields, 16)
        self.special_modifiers = _codes(fields, 17)
        self.texture_type = _dword(fields, 18)
        # the kinds of the modifiers which the spell takes: range, targets, area, effect, duration
        data = fields.get(20) or b''
        self.modifier_flags = tuple(bool(b) for b in bytes(data[:SPELL_MODIFIER_FLAGS_COUNT]))
        # the bits of the shops which sell the scroll
        self.shops = _dword(fields, 22)
        # the light of the spell: color, radius and the time of its fading
        self.color = (_float(fields, 23), _float(fields, 24), _float(fields, 25))
        self.light_radius = _float(fields, 26)
        self.fadeout_time = _float(fields, 27)

    def __repr__(self):
        return 'SpellPrototype(%r, %r, %s)' % (self.name, self.code, self.subtype)


class SpellModifier(object):
    """
    A record of the section 2 of spells.sdb (0x24 bytes, 683ab0).
    """

    def __init__(self, index, fields):
        self.index = index
        self.name = _text(fields, 0)
        # a dword of up to 4 characters: "r0", "r1", "fe", "im1"
        self.code = _code(fields.get(1))
        self.price = _float(fields, 2)
        self.kind = _dword(fields, 3)
        self.mana = _float(fields, 4)
        self.value = _float(fields, 5)
        self.level = _dword(fields, 6)
        self.race = _text(fields, 7)
        self.shops = _dword(fields, 8)

    def __repr__(self):
        return 'SpellModifier(%r, %r, kind %d, value %g)' % (self.name, self.code, self.kind, self.value)


class SpellTemplate(object):
    """
    A record of the sections 3..5 of spells.sdb (0x34 bytes, 683d00): a spell with the modifiers
    which it always has and the letters of the kinds of the modifiers which may be added.
    """

    def __init__(self, fields):
        self.code = _text(fields, 1)
        self.modifiers = _texts(fields, 2)
        self.allowed = _texts(fields, 3)
        self.kind = _text(fields, 4)
        self.mask = _dword(fields, 5)

    def __repr__(self):
        return 'SpellTemplate(%r, %r, %r)' % (self.code, self.modifiers, self.allowed)


class SpellDatabase(object):

    def __init__(self, spells, modifiers, templates, armor_templates, weapon_templates):
        self.spells = spells
        self.modifiers = modifiers
        self.templates = templates
        self.armor_templates = armor_templates
        self.weapon_templates = weapon_templates
        self._names = {s.name.lower(): s for s in spells}
        self._codes = {}
        for s in spells:
            self._codes.setdefault(s.code.lower(), s)
        self._modifiers = {m.code: m for m in modifiers}

    def spell(self, name):
        """
        A prototype by its name ("Fire Arrow") or code ("arrow").
        """
        key = (name or '').lower()
        return self._names.get(key) or self._codes.get(key)

    def modifier(self, code):
        return self._modifiers.get(code)


class UnitDatabase(object):

    def __init__(self, types, prototypes, templates=()):
        self.types = types
        self.prototypes = prototypes
        self.templates = list(templates)
        self._types = {t.name.lower(): t for t in types}
        self._prototypes = {p.name.lower(): p for p in prototypes}
        self._templates = {t.name.lower(): t for t in self.templates}

    def unit_type(self, name):
        return self._types.get((name or '').lower())

    def prototype(self, name):
        return self._prototypes.get((name or '').lower())

    def template(self, name):
        """
        5145d0: the hero template of the section 4 with the name of the prototype, None for the monsters.
        """
        return self._templates.get((name or '').lower())

#------------------------------------------------------------------------------


def iter_nodes(data):
    position = 0
    end = len(data)
    while position + 2 <= end:
        tag = data[position]
        size = data[position + 1]
        if size & 1:
            if position + 5 > end:
                return
            size = struct.unpack_from('<I', data, position + 1)[0]
            position += 5
        else:
            position += 2
        size >>= 1
        if position + size > end:
            return
        yield Node(tag, data[position:position + size])
        position += size


def root(data):
    for node in iter_nodes(data):
        return node
    return None


def parse_items(data):
    """
    The sections 1, 2, 3 and 5 of items.idb; game.exe reads the first three by 6a7900, 6a79f0 and 6a7af0.
    """
    sections = {node.tag: node for node in root(data).children()}
    materials = [Material(i, r.fields()) for i, r in enumerate(_records(sections, SECTION_MATERIALS))]
    weapons = [WeaponShape(i, r.fields()) for i, r in enumerate(_records(sections, SECTION_WEAPONS))]
    armors = [ArmorShape(i, r.fields()) for i, r in enumerate(_records(sections, SECTION_ARMORS))]
    quest_items = [QuestItem(r.fields()) for r in _records(sections, SECTION_QUEST_ITEMS)]
    if _Debug:
        print('items.idb: %d materials, %d weapons, %d armors' % (len(materials), len(weapons), len(armors)))
    return ItemDatabase(materials, weapons, armors, quest_items)


def parse_quests(data):
    """
    name -> Quest of the section 1 of quests.qdb.
    """
    sections = {node.tag: node for node in root(data).children()}
    quests = [Quest(r.fields()) for r in _records(sections, 1)]
    return {q.name.lower(): q for q in quests}

def parse_acks(data):
    """
    The name of a prototype -> Acks of the section 1 of acks.db (read by 6db3f0).
    """
    sections = {node.tag: node for node in root(data).children()}
    acks = [Acks(r) for r in _records(sections, 1)]
    return {a.name: a for a in acks}


def parse_units(data):
    """
    The sections 2, 3 and 4 of units.udb, read by game.exe in 53df50, 53dc10 and 53dd00.
    """
    sections = {node.tag: node for node in root(data).children()}
    types = [UnitType(i, r.fields()) for i, r in enumerate(_records(sections, SECTION_UNIT_TYPES))]
    prototypes = [UnitPrototype(i, r.fields()) for i, r in enumerate(_records(sections, SECTION_UNIT_PROTOTYPES))]
    templates = [HeroTemplate(i, r.fields()) for i, r in enumerate(_records(sections, SECTION_HERO_TEMPLATES))]
    if _Debug:
        print('units.udb: %d types, %d prototypes, %d templates' % (len(types), len(prototypes), len(templates)))
    return UnitDatabase(types, prototypes, templates)


def parse_perks(data):
    """
    The sections 1 and 2 of perks.pdb, read by game.exe in 5207d0 and 5208c0.
    """
    sections = {node.tag: node for node in root(data).children()}
    skills = [Skill(i, r.fields()) for i, r in enumerate(_records(sections, SECTION_SKILLS))]
    perks = [Perk(i, r.fields()) for i, r in enumerate(_records(sections, SECTION_PERKS))]
    if _Debug:
        print('perks.pdb: %d skills, %d perks' % (len(skills), len(perks)))
    return PerkDatabase(skills, perks)


def parse_spells(data):
    """
    The sections of spells.sdb: the prototypes (684eb0), the modifiers (684fa0) and the templates (684c10).
    """
    sections = {node.tag: node for node in root(data).children()}
    spells = [SpellPrototype(i, r.fields()) for i, r in enumerate(_records(sections, SECTION_SPELLS))]
    modifiers = [SpellModifier(i, r.fields()) for i, r in enumerate(_records(sections, SECTION_SPELL_MODIFIERS))]
    templates = [[SpellTemplate(r.fields()) for r in _records(sections, tag)] for tag in (
        SECTION_SPELL_TEMPLATES, SECTION_ARMOR_SPELL_TEMPLATES, SECTION_WEAPON_SPELL_TEMPLATES)]
    # every section of the templates ends with an empty record
    templates = [[t for t in group if t.code] for group in templates]
    if _Debug:
        print('spells.sdb: %d spells, %d modifiers' % (len(spells), len(modifiers)))
    return SpellDatabase(spells, modifiers, *templates)

#------------------------------------------------------------------------------


def _records(sections, tag):
    section = sections.get(tag)
    return section.children() if section is not None else []


def _text(fields, tag):
    data = fields.get(tag)
    if data is None:
        return ''
    return bytes(data).split(b'\0', 1)[0].decode(ENCODING)


def _texts(fields, tag):
    """
    A list of texts kept as the nodes inside one field ("steps\\Human\\Grass", ...), as 52df40 reads them.
    """
    data = fields.get(tag)
    if not data:
        return []
    return [bytes(child.data).split(b'\0', 1)[0].decode(ENCODING) for child in Node(tag, data).children()]


def _code(data):
    """
    A short code kept in a dword or in a field without the ending zero ("r0", "fe").
    """
    return bytes(data or b'').split(b'\0', 1)[0].decode(ENCODING)


def _codes(fields, tag):
    data = fields.get(tag)
    if not data:
        return []
    return [_code(child.data) for child in Node(tag, data).children()]


def _first(values):
    return values[0] if values else ''


def _bytes(fields, tag, count):
    data = bytes(fields.get(tag) or b'')[:count]
    return tuple(data) + (0, ) * (count - len(data))


def _char(fields, tag):
    value = _dword(fields, tag)
    return chr(value) if 0 < value < 0x100 else ''


def _float(fields, tag):
    data = fields.get(tag)
    if data is None or len(data) < 4:
        return 0.0
    return struct.unpack_from('<f', data)[0]


def _floats(fields, tag, count, default=0.0):
    data = fields.get(tag)
    if data is None or len(data) < 4 * count:
        return (default, ) * count
    return struct.unpack_from('<%df' % count, data)


def _armor_values(fields, tag):
    """
    (scale, the 7 values of the types of the damage, flags) of an absorption of an armor shape.
    """
    data = fields.get(tag)
    count = DAMAGE_TYPES_COUNT + 1
    if data is None or len(data) < 4 * (count + 1):
        return 0.0, (0.0, ) * DAMAGE_TYPES_COUNT, 0
    values = struct.unpack_from('<%df' % count, data)
    flags = struct.unpack_from('<I', data, 4 * count)[0]
    return values[0], tuple(values[1:]), flags


def _dword(fields, tag):
    data = fields.get(tag)
    if data is None or len(data) < 4:
        return 0
    return struct.unpack_from('<i', data)[0]
