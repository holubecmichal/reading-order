from reading_order.reading_order import ReadingOrder


def compare(ground_truth: ReadingOrder, identified: ReadingOrder) -> 'Recall':
    """
    Metrika Recall
    Z predane ground truht a identifikovane posloupnosti cteni vytahne stryktury Chain Reduction, dvojice
    se kterymi pracuje v samostatne tride Recall
    """

    gt = set(ground_truth.get_chain_reduction())
    i = set(identified.get_chain_reduction())

    return Recall(gt, i)

class Recall:
    def __init__(self, gt: set, i: set):
        self.gt = gt
        self.i = i

    def get_hits(self):
        """
        Vrati dvojice, ktere jsou jak v ground truth tak identifikovane posloupnosti
        """
        return self.gt.intersection(self.i)

    def total(self):
        """
        Celkovy pocet prvku ground truth
        """
        return len(self.gt)

    def hits(self):
        """
        Celkovy pocet stejnych dvojic, absolutni cislo
        """

        return len(self.get_hits())

    def recall(self):
        """
        Hodnota Recall
        """
        return round(self.hits() / self.total() * 100, 2)

    def get_missed(self):
        """
        Vraci dvojice, ktere nebyly spravne identifikovany
        """
        return self.gt.difference(self.i)

    def missed(self):
        """
        Pocet nespravne identifikovanych dvojic
        """
        return len(self.get_missed())
