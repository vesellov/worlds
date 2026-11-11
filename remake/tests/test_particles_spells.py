"""
G14: the particles of the visual of the spells which the main flow does not show yet: the fireworks (18),
the clairvoyance (20) and the link (27).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import random
import unittest

#------------------------------------------------------------------------------

from engine import particles

#------------------------------------------------------------------------------


class SpellParticlesTest(unittest.TestCase):

    def test_textures_of_the_new_types(self):
        for code in (particles.FIREWORK, particles.LINK, particles.CLAY_RING):
            emitter = particles.Emitter(code, (0.0, 0.0, 0.0))
            self.assertTrue(emitter.texture_name)
        self.assertEqual(particles.Emitter(particles.FIREWORK, (0, 0, 0)).texture_name, 'fireworks')
        self.assertEqual(particles.Emitter(particles.LINK, (0, 0, 0)).texture_name, 'link')
        self.assertEqual(particles.Emitter(particles.CLAY_RING, (0, 0, 0)).texture_name, 'clayring')

    def test_firework(self):
        emitter = particles.Emitter(particles.FIREWORK, (1.0, 2.0, 3.0), rng=random.Random(3))
        for _ in range(5):
            self.assertTrue(emitter.update())
        self.assertTrue(emitter.particles)
        particle = emitter.particles[0]
        self.assertGreater(particle.life, 0)
        z = particle.z
        vz = particle.vz
        emitter.update()
        self.assertLess(particle.vz, vz)
        self.assertNotEqual(particle.z, z)

    def test_clay_ring(self):
        emitter = particles.Emitter(particles.CLAY_RING, (0.0, 0.0, 0.0), rng=random.Random(4))
        self.assertTrue(emitter.additive)
        emitter.update()
        self.assertTrue(emitter.particles)
        self.assertEqual(emitter.particles[0].color >> 24, 255)

    def test_link(self):
        emitter = particles.Emitter(particles.LINK, (10.0, 0.0, 0.0), rng=random.Random(5))
        emitter.link_from = (0.0, 0.0, 0.0)
        for _ in range(4):
            emitter.update()
        self.assertTrue(emitter.particles)
        self.assertTrue(all(-0.25 <= p.x <= 10.25 for p in emitter.particles))

    def test_emitter_of_a_spell(self):
        self.assertIsInstance(particles.spell_emitter(18, (0, 0, 0)), particles.Emitter)
        self.assertIsInstance(particles.spell_emitter(20, (0, 0, 0)), particles.Emitter)
        emitter = particles.spell_emitter(27, (4, 0, 0), origin=(0, 0, 0))
        self.assertEqual(emitter.link_from, (0, 0, 0))
        self.assertIsNone(particles.spell_emitter(24, (0, 0, 0)))


if __name__ == '__main__':
    unittest.main()
