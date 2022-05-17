import unittest
import os

from document import page_xml
from reading_order.metric.prima import compare

DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(DIR, 'data')
GROUND_TRUTH_PATH = os.path.join(DATA_PATH, 'gt.xml')
ND_GROUND_TRUTH_PATH = os.path.join(DATA_PATH, 'nd_gt.xml')

ground_truth_xml = page_xml.parse(GROUND_TRUTH_PATH)
nd_grount_truth_xml = page_xml.parse(ND_GROUND_TRUTH_PATH)


class TestPrimaMetric(unittest.TestCase):
    def test_ok(self):
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), ground_truth_xml.get_reading_order())

        self.assertEqual([], results.errors())
        self.assertEqual(0, results.penalty())
        self.assertEqual(100.0, results.percentage())

    def test_UNORDERED_instead_SUCCESSOR(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '1.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())                

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual(['Ground Truth region r37 - region r38: Relation [--] instead of [->]'], output)
        self.assertEqual(20, penalty)
        self.assertEqual(97.87, penalty_percentage)

    def test_UNORDERED_instead_SUCCESSOR_2(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '2.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()

        self.assertEqual(['Ground Truth region r29 - region r30: Relation [--] instead of [->]'], output)
        self.assertEqual(20, penalty)
        self.assertEqual(97.87, penalty_percentage)

    def test_UNORDERED_instead_SUCCESSOR_3(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '3.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()

        self.assertEqual(['Ground Truth region r29 - region r30: Relation [--] instead of [->]'], output)
        self.assertEqual(20, penalty)
        self.assertEqual(97.87, penalty_percentage)

    def test_PREDECESSOR_instead_SUCCESSOR(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '4.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual(['Ground Truth region r29 - region r30: Relation [<-] instead of [->]'], output)
        self.assertEqual(40, penalty)
        self.assertEqual(95.83, penalty_percentage)

    def test_ok_2(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '5.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([], output)
        self.assertEqual(0, penalty)
        self.assertEqual(100.0, penalty_percentage)

    def test_ok_3(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '6.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([], output)
        self.assertEqual(0, penalty)
        self.assertEqual(100.0, penalty_percentage)

    def test_ok_4(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '14.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([], output)
        self.assertEqual(0, penalty)
        self.assertEqual(100.0, penalty_percentage)

    def test_ok_5(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '14.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([], output)
        self.assertEqual(0, penalty)
        self.assertEqual(100.0, penalty_percentage)

    def test_ok_6(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '18.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([], output)
        self.assertEqual(0, penalty)
        self.assertEqual(100.0, penalty_percentage)

    def test_ok_7(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '25.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([], output)
        self.assertEqual(0, penalty)
        self.assertEqual(100.0, penalty_percentage)

    def test_ok_8(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '26.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([], output)
        self.assertEqual(0, penalty)
        self.assertEqual(100.0, penalty_percentage)

    def test_ordered_group_sort_multiple_errors(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '7.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r35 - region r36: Relation [<-] instead of [->]',
            'Ground Truth region r35 - region r37: Relation [->] instead of [-x-]',
            'Ground Truth region r36 - region r37: Relation [-x-] instead of [->]'
        ], output)

        self.assertEqual(80, penalty)
        self.assertEqual(92.0, penalty_percentage)

    def test_ordered_group_sort_multiple_errors2(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '8.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r35 - region r36: Relation [-x-] instead of [->]',
            'Ground Truth region r35 - region r37: Relation [<-] instead of [-x-]',
            'Ground Truth region r35 - region r38: Relation [->] instead of [-x-]',
            'Ground Truth region r37 - region r38: Relation [-x-] instead of [->]',
        ], output)

        self.assertEqual(80, penalty)
        self.assertEqual(92.0, penalty_percentage)

    def test_ordered_group_sort_multiple_errors3(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '9.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r35 - region r36: Relation [-x-] instead of [->]',
            'Ground Truth region r35 - region r38: Relation [<-] instead of [-x-]',
        ], output)

        self.assertEqual(40, penalty)
        self.assertEqual(95.83, penalty_percentage)

    def test_ordered_group_sort_multiple_errors4(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '10.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r35 - region r36: Relation [-x-] instead of [->]',
            'Ground Truth region r35 - region r38: Relation [<-] instead of [-x-]',
            'Ground Truth region r36 - region r37: Relation [<-] instead of [->]',
            'Ground Truth region r36 - region r38: Relation [->] instead of [-x-]',
            'Ground Truth region r37 - region r38: Relation [-x-] instead of [->]',
        ], output)

        self.assertEqual(120, penalty)
        self.assertEqual(88.46, penalty_percentage)

    def test_ordered_group_sort_multiple_errors5(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '11.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r35 - region r36: Relation [<-] instead of [->]',
            'Ground Truth region r36 - region r37: Relation [-x-] instead of [->]',
            'Ground Truth region r36 - region r38: Relation [<-] instead of [-x-]',
        ], output)

        self.assertEqual(80, penalty)
        self.assertEqual(92.0, penalty_percentage)

    def test_ordered_group_sort_multiple_errors6(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '12.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r35 - region r37: Relation [<-] instead of [-x-]',
            'Ground Truth region r36 - region r37: Relation [-x-] instead of [->]',
            'Ground Truth region r37 - region r38: Relation [<-] instead of [->]',
        ], output)

        self.assertEqual(80, penalty)
        self.assertEqual(92.0, penalty_percentage)

    def test_ordered_group_sort_multiple_errors7(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '13.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r35 - region r36: Relation [<-] instead of [->]',
            'Ground Truth region r36 - region r37: Relation [<-] instead of [->]',
            'Ground Truth region r37 - region r38: Relation [<-] instead of [->]',
        ], output)

        self.assertEqual(120, penalty)
        self.assertEqual(88.46, penalty_percentage)

    def test_ordered_diff_group(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '15.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r29 - region r30: Relation [-x-] instead of [->]',
            'Ground Truth region r29 - region r35: Relation [->] instead of [-x-]',
            'Ground Truth region r30 - region r35: Relation [<-] instead of [->->]',
            'Ground Truth region r35 - region r36: Relation [-x-] instead of [->]',
        ], output)

        self.assertEqual(70, penalty)
        self.assertEqual(92.93, penalty_percentage)

    def test_ordered_diff_group2(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '17.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r37 - region r38: Relation [-x-] instead of [->]',
            'Ground Truth region r37 - region r55: Relation [->] instead of [-x-]',
            'Ground Truth region r38 - region r55: Relation [<-] instead of [->->]',
            'Ground Truth region r55 - region r56: Relation [-x-] instead of [->]',
        ], output)

        self.assertEqual(70, penalty)
        self.assertEqual(92.93, penalty_percentage)

    def test_ordered_diff_group3(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '19.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r29 - region r30: Relation [-x-] instead of [->]',
            'Ground Truth region r29 - region r55: Relation [->] instead of [-x-]',
            'Ground Truth region r30 - region r55: Relation [<-] instead of [->->]',
            'Ground Truth region r55 - region r56: Relation [-x-] instead of [->]',
        ], output)

        self.assertEqual(70, penalty)
        self.assertEqual(92.93, penalty_percentage)

    def test_ordered_diff_group4(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '20.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r29 - region r55: Relation [<-] instead of [-x-]',
            'Ground Truth region r55 - region r56: Relation [-x-] instead of [->]',
        ], output)

        self.assertEqual(40, penalty)
        self.assertEqual(95.83, penalty_percentage)

    def test_ordered_diff_group5(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '21.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r29 - region r30: Relation [-x-] instead of [->]',
            'Ground Truth region r29 - region r57: Relation [->] instead of [-x-]',
            'Ground Truth region r30 - region r57: Relation [<-] instead of [-x-]',
            'Ground Truth region r56 - region r57: Relation [-x-] instead of [->]',
            'Ground Truth region r56 - region r58: Relation [->] instead of [-x-]',
            'Ground Truth region r57 - region r58: Relation [-x-] instead of [->]',
        ], output)

        # self.assertEqual(90, penalty)
        # self.assertEqual(0.92, penalty_percentage)

        self.assertEqual(120, penalty)
        self.assertEqual(88.46, penalty_percentage)

    def test_ordered_diff_group6(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '22.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r35 - region r36: Relation [<-<-] instead of [->]',
            'Ground Truth region r35 - region r37: Relation [->] instead of [-x-]',
            'Ground Truth region r36 - region r37: Relation [-x-] instead of [->]',
            'Ground Truth region r36 - region r59: Relation [<-] instead of [-x-]',
        ], output)

        self.assertEqual(80, penalty)
        self.assertEqual(92.0, penalty_percentage)

    def test_ordered_diff_group7(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '23.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r37 - region r38: Relation [-x-] instead of [->]',
            'Ground Truth region r37 - region r55: Relation [->] instead of [-x-]',
            'Ground Truth region r38 - region r55: Relation [<-] instead of [->->]',
            'Ground Truth region r55 - region r56: Relation [-x-] instead of [->]',
        ], output)

        self.assertEqual(70, penalty)
        self.assertEqual(92.93, penalty_percentage)

    def test_ordered_diff_group8(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '24.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r35 - region r55: Relation [<-] instead of [-x-]',
            'Ground Truth region r55 - region r56: Relation [-x-] instead of [->]',
        ], output)

        self.assertEqual(40, penalty)
        self.assertEqual(95.83, penalty_percentage)

    def test_ordered_diff_group9(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '27.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r36 - region r37: Relation [-x-] instead of [->]',
            'Ground Truth region r36 - region r56: Relation [->] instead of [-x-]',
            'Ground Truth region r37 - region r56: Relation [<-] instead of [-x-]',
            'Ground Truth region r55 - region r56: Relation [-x-] instead of [->]',
            'Ground Truth region r55 - region r57: Relation [->] instead of [-x-]',
            'Ground Truth region r56 - region r57: Relation [-x-] instead of [->]',
        ], output)

        self.assertEqual(120, penalty)
        self.assertEqual(88.46, penalty_percentage)

    def test_group_switch_types(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '28.xml'))
        results = compare(ground_truth_xml, ground_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()
        
        self.assertEqual([
            'Ground Truth region r29 - region r30: Relation [--] instead of [->]',
            'Ground Truth region r35 - region r36: Relation [--] instead of [->]',
            'Ground Truth region r35 - region r37: Relation [--] instead of [-x-]',
            'Ground Truth region r35 - region r38: Relation [--] instead of [-x-]',
            'Ground Truth region r36 - region r37: Relation [--] instead of [->]',
            'Ground Truth region r36 - region r38: Relation [--] instead of [-x-]',
            'Ground Truth region r37 - region r38: Relation [--] instead of [->]',
            'Ground Truth region r55 - region r56: Relation [--] instead of [->]',
            'Ground Truth region r55 - region r57: Relation [--] instead of [-x-]',
            'Ground Truth region r55 - region r58: Relation [--] instead of [-x-]',
            'Ground Truth region r55 - region r59: Relation [--] instead of [-x-]',
            'Ground Truth region r56 - region r57: Relation [--] instead of [->]',
            'Ground Truth region r56 - region r58: Relation [--] instead of [-x-]',
            'Ground Truth region r56 - region r59: Relation [--] instead of [-x-]',
            'Ground Truth region r57 - region r58: Relation [--] instead of [->]',
            'Ground Truth region r57 - region r59: Relation [--] instead of [-x-]',
            'Ground Truth region r58 - region r59: Relation [--] instead of [->]',
        ], output)

        self.assertEqual(250, penalty)
        self.assertEqual(78.63, penalty_percentage)

    def test_not_defined_1(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '29.xml'))
        results = compare(nd_grount_truth_xml, nd_grount_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()

        self.assertEqual([
            'Ground Truth region r13 - region r24: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r15 - region r24: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r17 - region r24: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r18 - region r24: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r21 - region r24: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r22 - region r24: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r23 - region r24: Relation [->] instead of [n.d.]',
            'Ground Truth region r23 - region r25: Relation [-x-] instead of [->]',
            'Ground Truth region r24 - region r25: Relation [->] instead of [n.d.]',
            'Ground Truth region r24 - region r26: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r24 - region r28: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r24 - region r29: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r24 - region r30: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r24 - region r33: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r24 - region r34: Relation [-x-] instead of [n.d.]',
        ], output)

        self.assertEqual(20, penalty)
        self.assertEqual(97.44, penalty_percentage)

    def test_not_defined_2(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '29.xml'))
        results = compare(nd_grount_truth_xml, xml.get_reading_order(), nd_grount_truth_xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()

        self.assertEqual([
            'Ground Truth region r13 - region r24: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r15 - region r24: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r17 - region r24: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r18 - region r24: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r21 - region r24: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r22 - region r24: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r23 - region r24: Relation [n.d.] instead of [->]',
            'Ground Truth region r23 - region r25: Relation [->] instead of [-x-]',
            'Ground Truth region r24 - region r25: Relation [n.d.] instead of [->]',
            'Ground Truth region r24 - region r26: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r24 - region r28: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r24 - region r29: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r24 - region r30: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r24 - region r33: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r24 - region r34: Relation [n.d.] instead of [-x-]',
        ], output)

        self.assertEqual(60, penalty)
        self.assertEqual(92.68, penalty_percentage)

    def test_not_defined_3(self):
        xml = page_xml.parse(os.path.join(DATA_PATH, '30.xml'))
        results = compare(nd_grount_truth_xml, nd_grount_truth_xml.get_reading_order(), xml.get_reading_order())

        output = results.errors()
        penalty = results.penalty()
        penalty_percentage = results.percentage()

        self.assertEqual([
            'Ground Truth region r13 - region r18: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r13 - region r24: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r15 - region r18: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r15 - region r24: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r17 - region r18: Relation [n.d.] instead of [->]',
            'Ground Truth region r17 - region r21: Relation [->] instead of [-x-]',
            'Ground Truth region r17 - region r24: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r18 - region r21: Relation [n.d.] instead of [->]',
            'Ground Truth region r18 - region r22: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r18 - region r23: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r18 - region r25: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r18 - region r26: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r18 - region r28: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r18 - region r29: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r18 - region r30: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r18 - region r33: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r18 - region r34: Relation [n.d.] instead of [-x-]',
            'Ground Truth region r21 - region r24: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r22 - region r24: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r23 - region r24: Relation [->] instead of [n.d.]',
            'Ground Truth region r23 - region r25: Relation [-x-] instead of [->]',
            'Ground Truth region r24 - region r25: Relation [->] instead of [n.d.]',
            'Ground Truth region r24 - region r26: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r24 - region r28: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r24 - region r29: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r24 - region r30: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r24 - region r33: Relation [-x-] instead of [n.d.]',
            'Ground Truth region r24 - region r34: Relation [-x-] instead of [n.d.]',
        ], output)

        self.assertEqual(80, penalty)
        self.assertEqual(90.48, penalty_percentage)

if __name__ == '__main__':
    unittest.main()