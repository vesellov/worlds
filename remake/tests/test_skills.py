"""
G10: the skills and the abilities of the hero in the camp (624ed0, 6268e0, 528ab0, 528050, 5260a0).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import experience
from engine import skills

#------------------------------------------------------------------------------


class SkillsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()
        cls.perks = cls.game_data.perks()

    def make_stats(self, melee=0, experience_value=0.0):
        stats = experience.HeroStats(self.perks)
        stats.skills[experience.SKILL_MELEE] = melee
        stats.experience = experience_value
        return stats

    def test_skill_raise(self):
        stats = self.make_stats(melee=3)
        stats.experience = 1000.0
        price = skills.skill_cost(stats, experience.SKILL_MELEE)
        self.assertEqual(price, stats.settings.skill_raise_price(3))
        self.assertTrue(skills.raise_skill(stats, experience.SKILL_MELEE))
        self.assertEqual(stats.skills[experience.SKILL_MELEE], 4)
        self.assertAlmostEqual(skills.free_experience(stats), 1000.0 - stats.spent_experience)

    def test_skill_without_experience(self):
        stats = self.make_stats(melee=3)
        stats.experience = stats.spent_experience
        self.assertFalse(skills.raise_skill(stats, experience.SKILL_MELEE))
        stats.skills[experience.SKILL_MELEE] = skills.MAX_SKILL
        self.assertIsNone(skills.skill_cost(stats, experience.SKILL_MELEE))

    def test_perk_cost(self):
        stats = self.make_stats()
        sword1 = self.perks.perk('sword1')
        sword2 = self.perks.perk('sword2')
        price = stats.settings.skill_price
        self.assertEqual(skills.perk_cost(stats, sword1), int(price(10) - price(0)))
        self.assertEqual(skills.perk_cost(stats, sword2), int(price(20) - price(10)))

    def test_perk_requirements(self):
        stats = self.make_stats(melee=9)
        sword1 = self.perks.perk('sword1')
        sword2 = self.perks.perk('sword2')
        self.assertFalse(skills.perk_available(stats, sword1))
        self.assertFalse(skills.perk_available(stats, sword2))
        stats.skills[experience.SKILL_MELEE] = 10
        self.assertTrue(skills.perk_available(stats, sword1))
        self.assertFalse(skills.perk_available(stats, sword2))
        self.assertIn(sword1, [perk for perk, _ in skills.available_perks(stats)])

    def test_buy_perk(self):
        stats = self.make_stats(melee=10, experience_value=1000.0)
        sword1 = self.perks.perk('sword1')
        cost = skills.perk_cost(stats, sword1)
        free = skills.free_experience(stats)
        self.assertTrue(skills.buy_perk(stats, sword1))
        self.assertEqual(stats.perk_levels[sword1.group], 1)
        self.assertEqual(skills.perk_experience(stats), cost)
        self.assertAlmostEqual(skills.free_experience(stats), free - cost)
        # CopyStats keeps what the perks cost
        self.assertAlmostEqual(stats.copy().free_experience, free - cost)
        self.assertIn(sword1, skills.known_perks(stats))
        self.assertFalse(skills.buy_perk(stats, sword1))

    def test_ability_without_the_camp_flag(self):
        stats = self.make_stats(experience_value=1000.0)
        stats.skills[experience.SKILL_SENSE] = 15
        stats.skills[experience.SKILL_ASTRAL] = 15
        divination = self.perks.perk('divination1')
        healing = self.perks.perk('healing1')
        self.assertFalse(skills.perk_available(stats, divination))
        self.assertFalse(skills.perk_available(stats, healing))
        self.assertTrue(skills.perk_available(stats, self.perks.perk('illusion1')))
        self.assertTrue(skills.perk_available(stats, self.perks.perk('enchantments1')))

    def test_perk_without_money(self):
        stats = self.make_stats(melee=10)
        sword1 = self.perks.perk('sword1')
        stats.experience = stats.spent_experience + 1.0
        self.assertFalse(skills.buy_perk(stats, sword1))
        self.assertEqual(stats.perk_levels[sword1.group], 0)


if __name__ == '__main__':
    unittest.main()
