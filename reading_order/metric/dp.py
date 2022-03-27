from reading_order.reading_order import ReadingOrder


def compare(ground_truth: ReadingOrder, actual: ReadingOrder) -> 'DpResults':
    gt = set(ground_truth.get_before_in_reading())
    a = set(actual.get_before_in_reading())

    return DpResults(gt, a)

class DpResults:
    def __init__(self, gt: set, a: set):
        self.gt = gt
        self.a = a

    def get_hits(self):
        return self.gt.intersection(self.a)

    def total(self):
        return len(self.gt)

    def hits(self):
        return len(self.get_hits())

    def accuracy(self):
        return round(self.hits() / self.total() * 100, 2)

    def get_missed(self):
        return self.gt.difference(self.a)

    def missed(self):
        return len(self.get_missed())

    def get_surplus(self):
        return self.a.difference(self.gt)

    def surplus(self):
        return len(self.get_surplus())
