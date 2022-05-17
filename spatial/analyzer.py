from document.stubs import Document as StubDocument, TextRegion as StubTextRegion
from language_model.analyzer import LmAnalyzer as LmAnalyzer
from reading_order.reading_order import ReadingOrder
from spatial.spatial_data import DocTBRR, get_neighborhood, get_headers, STARTS, ISTARTS, DURING, FINISHES, OVERLAPS, \
    IOVERLAPS, EQUALS, PRECEDES, MEETS, crosses_another_polygon
from utils.sorting import topological_sort


def chain_to_before_in_order(chain):
    """
    Pomocna funkce pro prevod do Complete Chain
    """

    before_in_order = []
    source = chain[0]
    for successor in chain[1:]:
        before_in_order.append((source, successor))
        source = successor

    return before_in_order


def chain_to_reading_order(chain, doc: StubDocument) -> ReadingOrder:
    """
    Funkce pro prevod chain do Reading Order
    chain = ['a', 'b', 'c']
    ReadingOrder = [(a, b), (b, c)]
    """


    reading_order = ReadingOrder()
    group = reading_order.add_ordered_group()

    regions = doc.get_text_regions()
    source = chain[0]
    for successor in chain[1:]:
        group.add_candidates(regions[source], regions[successor])
        source = successor

    return reading_order


def find_by_fst(key, values: [tuple]):
    """
    Pomocna funkce pro nalezeni prvku ve dvojici dle prvniho prvku
    (FIND, _)
    """

    return [x for x in values if x[0] == key]


def find_by_snd(key, values: [tuple]):
    """
    Pomocna funkce pro nalezeni prvku ve dvojici dle druheho prvku
    (_, FIND)
    """

    return [x for x in values if x[1] == key]


class DiagonalAnalyzer():
    """
    Prostorova diagonalni analyza
    """

    def analyze(self, doc: StubDocument) -> ReadingOrder:
        # inicializace TBRR instance
        tbrr = DocTBRR(doc)
        # inicializace vztahu regionu
        r = tbrr.get_relations()

        # complete chain
        before_in_reading = []

        for a in doc.get_text_regions():
            for b in doc.get_text_regions():
                if a == b:
                    continue

                if r.is_vertical_before_in_reading(a, b) and r.is_horizontal_before_in_reading(a, b):
                    before_in_reading.append((a, b))

        # prevod Complete chain na chain ['a', 'b', 'c']
        chain = list(topological_sort(before_in_reading))
        # vytvoreni Reading order, Reduction chain [(a, b), (b, c)]
        return chain_to_reading_order(chain, doc)

    def __call__(self, *args, **kwargs):
        return self.analyze(args[0])


class ColumnarAnalyzer():
    """
    Prostorova sloupcova analyza
    """


    def analyze(self, doc: StubDocument) -> ReadingOrder:
        # inicializace tbrr
        tbrr = DocTBRR(doc)
        # inicializace vzajemnych vztahu
        r = tbrr.get_relations()
        before_in_reading = []

        # nacte sloupce a regiony, ktere nejsou ve sloupcich, ktere nasledne porovna
        regions = tbrr.get_cols_and_independent_regions()
        for a in regions:
            for b in regions:
                if a == b:
                    continue

                vertical = r.is_vertical_before_in_reading(a, b)
                horizontal = r.is_horizontal_before_in_reading(a, b)

                if vertical and horizontal:
                    before_in_reading.append((a, b))

        chain = []
        # prevod Complete chain na chain ['a', 'b', 'c']
        for i in topological_sort(before_in_reading):
            if tbrr.is_col(i):
                # pokud je prvek z chain sloupcem, nahradi se puvodnimi textovymi regiony
                chain += tbrr.get_col(i).get_region_ids()
            else:
                # jinak nemodifikuju a jen pridam
                chain.append(i)

        # prevod na reading order
        return chain_to_reading_order(chain, doc)

    def __call__(self, *args, **kwargs):
        return self.analyze(args[0])


class ColumnarLmAnalyzer():
    """
    Kombinovana analyza
    """

    TWO_CANDIDATES = 0.85
    THREE_CANDIDATES = 0.6

    def __init__(self):
        self.tbrr = None
        self.r = None
        self.neighborhood = None
        self.doc = None
        self.headers = None

    def analyze(self, doc: StubDocument, lm_analyzer: LmAnalyzer) -> ReadingOrder:
        # nacteni sloupcoveho analyzatoru
        columnar_analyzer = ColumnarAnalyzer()

        # inicializace
        self.tbrr = DocTBRR(doc)
        self.r = self.tbrr.get_relations()

        # nacteni sousednosti
        self.neighborhood = get_neighborhood(doc.get_text_regions(), doc.get_box())
        self.doc = doc
        self.headers = get_headers(self.doc.get_text_regions())

        # provedeni sloupcove analyzy
        reading_order = columnar_analyzer(doc)
        before_in_reading = reading_order.get_chain_reduction()

        # nacteni regionu, ktere se jsou na poslednim miste ve sloupci,
        # tyto regiony budou podlehat jazykove analyze
        last_in_cols = [x for x in self.doc.get_text_regions() if self.tbrr.is_last_in_col(x)]

        for el in last_in_cols:
            candidates = []

            # nactu dvojici (el, successor)
            actual = find_by_fst(el, before_in_reading)
            if not actual:
                continue

            actual = actual[0]
            # el, successor
            fst, snd = actual[0], actual[1]

            # pridam aktualni napojeni
            candidates.append(snd)
            # pridam kandidaty z osy y
            candidates += self._get_y_candidates(el)
            # pridam kandidaty z osy x
            candidates += self._get_x_candidates(el)
            # pokud nektery z kandidatu byl ve sloupci, prevedu na sloupec
            candidates = self._convert_to_col(candidates)

            if len(candidates) == 1:
                # zustal jen jeden kandidat a to ten aktualne ve dvojici
                # pokracuju
                continue

            # priprava struktury pro odhad pravdepodobnosti
            candidates = {c: self._get_element(c) for c in candidates}
            # jazykova analyza
            probs = lm_analyzer.analyze_one(self.tbrr.get_col(el), candidates)

            # vytahne se vitez
            winner = max(probs, key=probs.get)
            if self.tbrr.is_col(winner):
                # pokud je vitezem sloupec, vezmu jeho prvni region
                winner = self.tbrr.get_col(winner).get_first_region_id()

            # pokud se aktualni successor nerovna viteznemu regionu a pokud je treba udelat zmenu (dle prahove hodnoty)
            # tak provedu zmenu
            if snd != winner and self._can_make_change(actual, probs, winner):
                before_in_reading = self._reconnect(before_in_reading, actual, (fst, winner))
                chain = topological_sort(before_in_reading)
                before_in_reading = chain_to_before_in_order(chain)

        # konecny chain se prevede na reading order
        chain = topological_sort(before_in_reading)
        return chain_to_reading_order(chain, doc)

    def __call__(self, *args, **kwargs):
        doc, lm = args
        return self.analyze(doc, lm)

    def _can_make_change(self, actual, probs, winner):
        """
        Metoda, kterou se overuje, zda jsou pravdepodobnosti vetsi, jak prah
        Pokud ano, provede se reconnect, jinak zustava aktualni prvek
        """

        fst, snd = actual
        # predzpracovani, vytazeni spravnych prvku
        if self.tbrr.is_in_col(snd):
            snd = self.tbrr.get_col(snd).get_id()

        if self.tbrr.is_in_col(winner):
            winner = self.tbrr.get_col(winner).get_id()

        if len(probs) >= 3 and probs[snd] < 0.10:
            return True

        # volba prahu dle poctu kandidatu
        if len(probs) == 2:
            val = self.TWO_CANDIDATES
        else:
            val = self.THREE_CANDIDATES

        # overeni pravdepodobnosti vuci prahu
        return probs[winner] >= val

    def _get_element(self, a) -> StubTextRegion:
        if self.tbrr.is_col(a):
            return self.tbrr.get_col(a)

        return self.doc.get_text_regions()[a]

    def _convert_to_col(self, candidates):
        """
        Prevod prvku na sloupce, pokud se tyto ve sloupci nachazi
        """

        for i, c in enumerate(candidates):
            if self.tbrr.is_in_col(c):
                candidates[i] = self.tbrr.get_col(c).get_id()

        candidates = set(candidates)
        return list(candidates)

    def _get_y_candidates(self, el):
        """
        Kontrola kandidatu na ose y,
        vraci nejblizsi prvek z osy y, ktery vyhovuje pozadovanym kriteriim
        """

        regions = {**self.doc.get_text_regions(), **self.tbrr.get_columns()}
        # kriteria pro x osu
        relations = [STARTS, ISTARTS, DURING, FINISHES, OVERLAPS, IOVERLAPS, EQUALS]
        # nactu sousedy
        candidates = self.neighborhood[el]
        # vyfiltruju ty, ktery el predchazi
        candidates = [x for x in candidates if self.r.precedes('y', el, x)]
        # vyfiltruju ty, ktere vyhovuji kriteriim pro osu x
        candidates = [x for x in candidates if self.r.get_relation('x', x, el) in relations]

        if not candidates:
            return []

        # vyfiltrovani toho kandidata, ktery je neblize
        candidate = min(candidates, key=lambda x: (regions[x].min_y() / 2) if self.r.equals('x', el, x)
                        else regions[x].min_y())

        return [candidate]

    def _analyze_upper_candidate(self, el):
        # regions = self.xml.get_text_regions() | self.cols
        regions = {**self.doc.get_text_regions(), **self.tbrr.get_columns()}
        relations = [STARTS, ISTARTS, DURING, FINISHES, EQUALS]
        candidates = [x for x in regions if self.r.get_relation('x', x, el) in relations]
        candidates = [x for x in candidates if self.r.get_relation('y', x, el) in [PRECEDES, MEETS]]
        candidates = [x for x in candidates if
                      not crosses_another_polygon(regions, el, x, check_cross=self.headers)]

        if not candidates:
            return None

        above = min(candidates, key=lambda x: regions[x].min_y())
        return above

    def _get_x_candidates(self, el):
        """
        Kontrola kandidatu na ose x,
        vraci prvky nejblize na ose x a zaroven prvek nejvyse polozeny prvek nad kandidatem
        """

        # inicializace hodnot
        regions = self.doc.get_regions()
        candidates = self.neighborhood[el]
        # vyfiltruju jen ty kandidaty z okoli, kterym el na ose x predchazi
        candidates = [x for x in candidates if self.r.precedes('x', el, x)]

        if not candidates:
            return []

        # z techto kandidatu vyberu ten, ktery je vyse
        above = min(candidates, key=lambda x: regions[x].min_y())
        candidates = [above]

        # analyza a vytazeni dalsiho prvku, ktery je nad timto kandidatem
        above = self._analyze_upper_candidate(above)
        if above:
            candidates.append(above)

        return candidates

    def _reconnect(self, before_in_reading, old, new):
        """
        Prepojeni grafu
        """

        if old == new:
            # lm potvrdil stavajici napojeni, nic neupravujeme
            return before_in_reading

        old_fst, old_snd = old
        new_fst, new_snd = new

        # preruseni spojeni
        found = find_by_snd(new_snd, before_in_reading)
        for i in found:
            before_in_reading.remove(i)

        # na misto puvodniho spojeni vlozi novou dvojici
        index = before_in_reading.index(old)
        before_in_reading.insert(index, new)
        before_in_reading.remove(old)

        # protoze rozpojenim a zarazenim noveho prvku doslo k naruseni spojeni,
        # pak to, ktere bylo rozpojeno znovu zaradim do chain na konec
        _, last_snd = before_in_reading[-1]
        before_in_reading.append((last_snd, old_snd))

        return before_in_reading


class TopToBottomAnalyzer():
    """
    Top to bottom analyzer, v DP neuvazovan
    """

    def analyze(self, doc: StubDocument):
        regions = doc.get_text_regions()
        chain = sorted(regions, key=lambda i: (regions[i].min_y(), regions[i].min_x()))
        return chain_to_reading_order(chain, doc)

    def __call__(self, *args, **kwargs):
        return self.analyze(args[0])