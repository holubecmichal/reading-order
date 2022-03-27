import unittest

import shapely.geometry

from spatial.stubs import Region
from spatial.spare import TBRR


class Item(Region):
    def __init__(self):
        super().__init__()
        self.box = None

    def get_polygon(self) -> shapely.geometry.Polygon:
        return self.box


class ItemX(Item):
    def __init__(self, x_min, x_max):
        super().__init__()
        self.box = shapely.geometry.box(x_min, 0, x_max, 100)


class ItemY(Item):
    def __init__(self, y_min, y_max):
        super().__init__()
        self.box = shapely.geometry.box(0, y_min, 100, y_max)


class TestTBRR(unittest.TestCase):
    def test_precedes_x_0(self):
        tbrr = TBRR(0)

        # precedes
        a = ItemX(0, 9)
        b = ItemX(10, 20)
        self.assertTrue(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # meets
        a = ItemX(0, 10)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertTrue(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # overlaps
        a = ItemX(0, 11)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertTrue(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # starts
        a = ItemX(10, 15)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertTrue(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # during
        a = ItemX(11, 15)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertTrue(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # finishes
        a = ItemX(11, 20)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertTrue(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # equals
        a = ItemX(10, 20)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertTrue(tbrr.equals_x(a, b))

        #############

        # overlaps
        a = ItemX(0, 12)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertTrue(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # during
        a = ItemX(12, 15)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertTrue(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # during
        a = ItemX(12, 19)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertTrue(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

    def test_precedes_x_1(self):
        tbrr = TBRR(1)

        # meets
        a = ItemX(0, 9)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertTrue(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # meets
        a = ItemX(0, 10)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertTrue(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # meets
        a = ItemX(0, 11)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertTrue(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # overlaps
        a = ItemX(0, 12)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertTrue(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # starts
        a = ItemX(10, 15)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertTrue(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # starts
        a = ItemX(11, 15)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertTrue(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # during
        a = ItemX(12, 15)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertTrue(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # finish
        a = ItemX(12, 19)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertTrue(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # equals
        a = ItemX(11, 20)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertTrue(tbrr.equals_x(a, b))

        # equals
        a = ItemX(10, 20)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertTrue(tbrr.equals_x(a, b))

    def test_precedes_x_1_changed(self):
        tbrr = TBRR(1)

        # precedes
        a = ItemX(0, 8)
        b = ItemX(10, 20)
        self.assertTrue(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # meets
        a = ItemX(0, 9)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertTrue(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # overlaps
        a = ItemX(0, 12)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertTrue(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # starts
        a = ItemX(9, 15)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertTrue(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # during
        a = ItemX(12, 15)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertTrue(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # finishes
        a = ItemX(12, 20)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertTrue(tbrr.finishes_x(a, b))
        self.assertFalse(tbrr.equals_x(a, b))

        # equals
        a = ItemX(9, 21)
        b = ItemX(10, 20)
        self.assertFalse(tbrr.precedes_x(a, b))
        self.assertFalse(tbrr.meets_x(a, b))
        self.assertFalse(tbrr.overlaps_x(a, b))
        self.assertFalse(tbrr.starts_x(a, b))
        self.assertFalse(tbrr.during_x(a, b))
        self.assertFalse(tbrr.finishes_x(a, b))
        self.assertTrue(tbrr.equals_x(a, b))

    def test_precedes_y_0(self):
        tbrr = TBRR(0)

        # precedes
        a = ItemY(0, 9)
        b = ItemY(10, 20)
        self.assertTrue(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # meets
        a = ItemY(0, 10)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertTrue(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # overlaps
        a = ItemY(0, 11)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertTrue(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # starts
        a = ItemY(10, 15)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertTrue(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # during
        a = ItemY(11, 15)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertTrue(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # finishes
        a = ItemY(11, 20)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertTrue(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # equals
        a = ItemY(10, 20)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertTrue(tbrr.equals_y(a, b))

        #############

        # overlaps
        a = ItemY(0, 12)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertTrue(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # during
        a = ItemY(12, 15)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertTrue(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # during
        a = ItemY(12, 19)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertTrue(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

    def test_precedes_y_1(self):
        tbrr = TBRR(1)

        # meets
        a = ItemY(0, 9)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertTrue(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # meets
        a = ItemY(0, 10)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertTrue(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # meets
        a = ItemY(0, 11)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertTrue(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # overlaps
        a = ItemY(0, 12)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertTrue(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # starts
        a = ItemY(10, 15)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertTrue(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # starts
        a = ItemY(11, 15)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertTrue(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # during
        a = ItemY(12, 15)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertTrue(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # finish
        a = ItemY(12, 19)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertTrue(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # equals
        a = ItemY(11, 20)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertTrue(tbrr.equals_y(a, b))

        # equals
        a = ItemY(10, 20)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertTrue(tbrr.equals_y(a, b))

    def test_precedes_y_1_changed(self):
        tbrr = TBRR(1)

        # precedes
        a = ItemY(0, 8)
        b = ItemY(10, 20)
        self.assertTrue(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # meets
        a = ItemY(0, 9)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertTrue(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # overlaps
        a = ItemY(0, 12)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertTrue(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # starts
        a = ItemY(9, 15)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertTrue(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # during
        a = ItemY(12, 15)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertTrue(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # finishes
        a = ItemY(12, 20)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertTrue(tbrr.finishes_y(a, b))
        self.assertFalse(tbrr.equals_y(a, b))

        # equals
        a = ItemY(9, 21)
        b = ItemY(10, 20)
        self.assertFalse(tbrr.precedes_y(a, b))
        self.assertFalse(tbrr.meets_y(a, b))
        self.assertFalse(tbrr.overlaps_y(a, b))
        self.assertFalse(tbrr.starts_y(a, b))
        self.assertFalse(tbrr.during_y(a, b))
        self.assertFalse(tbrr.finishes_y(a, b))
        self.assertTrue(tbrr.equals_y(a, b))

if __name__ == '__main__':
    unittest.main()