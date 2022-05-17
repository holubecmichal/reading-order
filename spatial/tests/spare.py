import unittest

import shapely.geometry

from spatial.stubs import Spatial as StubSpatial
from spatial.spatial_data import TBRR, PRECEDES, MEETS, OVERLAPS, STARTS, DURING, FINISHES, EQUALS


class Item(StubSpatial):
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
    def _assert_method(self, tbrr, dir, a, b, assert_method):
        for m in tbrr.methods:
            result = tbrr.call_method(m, dir, a, b)

            if m == assert_method:
                self.assertTrue(result)
            else:
                self.assertFalse(result)

    def test_precedes_x_0(self):
        tbrr = TBRR(0)

        # precedes
        a = ItemX(0, 9)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, PRECEDES)

        # meets
        a = ItemX(0, 10)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, MEETS)

        # overlaps
        a = ItemX(0, 11)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, OVERLAPS)

        # starts
        a = ItemX(10, 15)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, STARTS)

        # during
        a = ItemX(11, 15)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, DURING)

        # finishes
        a = ItemX(11, 20)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, FINISHES)

        # equals
        a = ItemX(10, 20)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, EQUALS)

        #############

        # overlaps
        a = ItemX(0, 12)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, OVERLAPS)

        # during
        a = ItemX(12, 15)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, DURING)

        # during
        a = ItemX(12, 19)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, DURING)

    def test_precedes_x_1(self):
        tbrr = TBRR(1)

        # meets
        a = ItemX(0, 9)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, MEETS)

        # meets
        a = ItemX(0, 10)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, MEETS)

        # meets
        a = ItemX(0, 11)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, MEETS)

        # overlaps
        a = ItemX(0, 12)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, OVERLAPS)

        # starts
        a = ItemX(10, 15)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, STARTS)

        # starts
        a = ItemX(11, 15)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, STARTS)

        # during
        a = ItemX(12, 15)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, DURING)

        # finish
        a = ItemX(12, 19)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, FINISHES)

        # equals
        a = ItemX(11, 20)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, EQUALS)

        # equals
        a = ItemX(10, 20)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, EQUALS)

    def test_precedes_x_1_changed(self):
        tbrr = TBRR(1)

        # precedes
        a = ItemX(0, 8)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, PRECEDES)

        # meets
        a = ItemX(0, 9)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, MEETS)

        # overlaps
        a = ItemX(0, 12)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, OVERLAPS)

        # starts
        a = ItemX(9, 15)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, STARTS)

        # during
        a = ItemX(12, 15)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, DURING)

        # finishes
        a = ItemX(12, 20)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, FINISHES)

        # equals
        a = ItemX(9, 21)
        b = ItemX(10, 20)
        self._assert_method(tbrr, 'x', a, b, EQUALS)

    def test_precedes_y_0(self):
        tbrr = TBRR(0)

        # precedes
        a = ItemY(0, 9)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, PRECEDES)

        # meets
        a = ItemY(0, 10)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, MEETS)

        # overlaps
        a = ItemY(0, 11)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, OVERLAPS)

        # starts
        a = ItemY(10, 15)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, STARTS)

        # during
        a = ItemY(11, 15)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, DURING)

        # finishes
        a = ItemY(11, 20)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, FINISHES)

        # equals
        a = ItemY(10, 20)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, EQUALS)

        #############

        # overlaps
        a = ItemY(0, 12)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, OVERLAPS)

        # during
        a = ItemY(12, 15)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, DURING)

        # during
        a = ItemY(12, 19)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, DURING)

    def test_precedes_y_1(self):
        tbrr = TBRR(1)

        # meets
        a = ItemY(0, 9)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, MEETS)

        # meets
        a = ItemY(0, 10)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, MEETS)

        # meets
        a = ItemY(0, 11)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, MEETS)

        # overlaps
        a = ItemY(0, 12)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, OVERLAPS)

        # starts
        a = ItemY(10, 15)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, STARTS)

        # starts
        a = ItemY(11, 15)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, STARTS)

        # during
        a = ItemY(12, 15)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, DURING)

        # finish
        a = ItemY(12, 19)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, FINISHES)

        # equals
        a = ItemY(11, 20)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, EQUALS)

        # equals
        a = ItemY(10, 20)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, EQUALS)

    def test_precedes_y_1_changed(self):
        tbrr = TBRR(1)

        # precedes
        a = ItemY(0, 8)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, PRECEDES)

        # meets
        a = ItemY(0, 9)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, MEETS)

        # overlaps
        a = ItemY(0, 12)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, OVERLAPS)

        # starts
        a = ItemY(9, 15)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, STARTS)

        # during
        a = ItemY(12, 15)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, DURING)

        # finishes
        a = ItemY(12, 20)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'y', a, b, FINISHES)

        # equals
        a = ItemY(9, 21)
        b = ItemY(10, 20)
        self._assert_method(tbrr, 'x', a, b, EQUALS)

if __name__ == '__main__':
    unittest.main()