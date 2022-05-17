from copy import copy
from typing import Optional

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import voronoi_diagram

from document.stubs import Document as StubDocument, TextRegion as StubTextRegion
from spatial.stubs import Spatial as StubSpatial

PRECEDES = 'precedes'
IPRECEDES = 'i_precedes'
MEETS = 'meets'
IMEETS = 'i_meets'
OVERLAPS = 'overlaps'
IOVERLAPS = 'i_overlaps'
STARTS = 'starts'
ISTARTS = 'i_starts'
DURING = 'during'
IDURING = 'i_during'
FINISHES = 'finishes'
IFINISHES = 'i_finishes'
EQUALS = 'equals'
UNKNOWN = 'unknown'


def get_lines(polygon: Polygon):
    """
    Funkce, ktera vrati usecky polygonu
    """

    lines = []
    points = list(zip(list(polygon.boundary.xy[0]), list(polygon.boundary.xy[1])))
    for i, fst in enumerate(points):
        if i == len(points) - 1:
            break

        snd = points[i + 1]
        lines.append(LineString([fst, snd]))

    return lines


def get_horizontal_lines(polygon: Polygon, threshold=0):
    """
    Funkce, ktera vraci horizontalni usecky
    """

    lines = get_lines(polygon)
    horizontal = []

    for line in lines:
        y_1, y_2 = min(line.xy[1]), max(line.xy[1])

        if y_2 - y_1 <= threshold:
            horizontal.append(line)

    return horizontal


def calculate_horizontal_T(regions: {StubSpatial}, threshold):
    """
    Vypocet prahove hodnoty T pro horizontalni usecky, pouzito pri identifikaci sloupcu
    """

    shortest = np.inf

    for id in regions:
        h_lines = get_horizontal_lines(regions[id].get_box(), threshold)
        for line in h_lines:
            shortest = min(shortest, line.length)

    return shortest


def calculate_T(regions: {StubSpatial}):
    """
    Vypocet prahove hodnoty T pro vsechny elementy
    """

    # nalezeni nejmensiho elementu dle jeho obsahu
    smallest_id = min(regions, key=lambda x: regions[x].get_box().area)
    smallest = regions[smallest_id]
    lines = get_lines(smallest.get_box())

    # nalezeni nejkratsi usecky nejmensiho elementu
    shortest = np.inf
    for line in lines:
        shortest = min(shortest, line.length)

    return shortest / 2

"""
Nasleduji metody pro identifikaci vztahu,
inspirovano z https://www.researchgate.net/publication/2527078_Document_Understanding_for_a_Broad_Class_of_Documents 
"""
####
def precedes(b_min: float, a_max: float, T: float):
    return a_max < (b_min - T)


def meets(b_min: float, a_max: float, T: float):
    return (b_min - T) <= a_max <= (b_min + T)


def overlaps(a_min, b_min, a_max, b_max, T: float):
    return a_min < (b_min - T) and (b_min + T) < a_max < (b_max - T)


def starts(a_min, b_min, a_max, b_max, T: float):
    return (b_min - T) <= a_min <= (b_min + T) and a_max < (b_max - T)


def during(a_min, b_min, a_max, b_max, T: float):
    return a_min > (b_min + T) and a_max < (b_max - T)


def finishes(a_min, b_min, a_max, b_max, T: float):
    return a_min > (b_min + T) and (b_max - T) <= a_max <= (b_max + T)


def equals(a_min, b_min, a_max, b_max, T: float):
    return (b_min - T) <= a_min <= (b_min + T) and (b_max - T) <= a_max <= (b_max + T)


def get_voronoi_polygons(regions: {StubSpatial}, centroid='default') -> [Polygon]:
    """
    Funkce pro vypocet voroneho polygonu
    """

    if centroid == 'left':
        centroids = [regions[x].most_left_centroid() for x in regions]
    else:
        centroids = [regions[x].centroid() for x in regions]

    centroids = MultiPoint(centroids)
    voronoi = voronoi_diagram(centroids)
    geoms = list(voronoi.geoms)
    polygons = {}

    # prirazeni voroneho polygonu pro jednotlive elementy
    for i in regions:
        region = regions[i]

        for j, polygon in enumerate(geoms):
            if centroid == 'left':
                c = region.most_left_centroid()
            else:
                c = region.centroid()

            if polygon.contains(c):
                polygons[i] = polygon
                break

    return polygons


def cutout_by_box(polygons: [Polygon], box: Polygon) -> [Polygon]:
    """
    Funkce, ktera orizne elementy, ktere zasahuji za okraj stranky.
    Pouzito v kombinaci s Voroneho polygony, ktere mohout byt protazeny
    az za okraj.
    """

    for i in polygons:
        if not box.contains(polygons[i]):
            # pokud stranka polygon neobsahuje, pak se polygon usekne stranou dokumentu
            result = shapely.ops.split(polygons[i], box.boundary)
            for splitted in result.geoms:
                # nahrazeni
                if box.contains(splitted):
                    polygons[i] = splitted
                    break

    return polygons


def get_direction(x, y):
    """
    Funkce pro definici smeru
    """

    x_dir = ''
    y_dir = ''

    if x in [PRECEDES, MEETS, OVERLAPS]:
        x_dir = 'e'

    if x in [IPRECEDES, IMEETS, IOVERLAPS]:
        x_dir = 'w'

    if y in [PRECEDES, MEETS, OVERLAPS]:
        y_dir = 's'

    if y in [IPRECEDES, IMEETS, IOVERLAPS]:
        y_dir = 'n'

    return y_dir + x_dir


def get_neighborhood(regions: {StubSpatial}, box: Polygon, centroid_position='left') -> dict:
    """
    Funkce pro vypocet sousednosti
    """

    polygons = get_voronoi_polygons(regions, centroid_position)
    polygons = cutout_by_box(polygons, box)
    neighborhood = {}

    for i in polygons:
        neighborhood[i] = []

        for j in polygons:
            if polygons[i].touches(polygons[j]):
                neighborhood[i].append(j)

    return neighborhood


def get_density(regions: {StubTextRegion}):
    """
    Vypocet hustoty pisma vuci plose
    """

    density = {}

    for i in regions:
        region = regions[i]

        if len(region.get_text()) == 0:
            density[i] = 0
        else:
            density[i] = region.get_box().area / len(region.get_text())

    return density


def get_headers(regions: {StubTextRegion}):
    """
    Funkce dle hustoty pisma vraci regiony, o kterych se domniva, ze se jedna o nadpis
    """

    density = get_density(regions)
    values = np.asarray(list(density.values()))
    limit = values.mean() + values.std()
    headers = {}

    for r in density:
        if density[r] > limit:
            headers[r] = regions[r]

    return headers


def get_page_number(regions: {StubTextRegion}, relations) -> Optional[StubTextRegion]:
    """
    Funkce, ktera hleda cislo stranky
    """

    for r in regions:
        if regions[r].get_text().isdigit():
            is_up = True
            is_down = True

            for rl in relations[r]:
                if rl == r:
                    continue

                if relations[r][rl]['y'] != PRECEDES:
                    is_up = False

                if relations[r][rl]['y'] != IPRECEDES:
                    is_down = False

            if is_up or is_down:
                return regions[r]

    return None


def get_line(a: StubSpatial, b: StubSpatial, dir=None) -> LineString:
    """
    Funkce vrati usecku mezi dvema dvema regiony, respektive mezi jejich geometrickymi stredy
    """

    start = [a.centroid().x, a.centroid().y]
    stop = [b.centroid().x, b.centroid().y]

    if dir:
        if 'n' in dir:
            start[1] = a.min_y()
            stop[1] = b.max_y()
        if 's' in dir:
            start[1] = a.max_y()
            stop[1] = b.min_y()
        if 'w' in dir:
            start[0] = a.min_x()
            stop[0] = b.max_x()
        if 'e' in dir:
            start[0] = a.max_x()
            stop[0] = b.min_x()

    return LineString([start, stop])


def get_distance(a: StubSpatial, b: StubSpatial, dir=None) -> float:
    """
    Vypocet vzdalenosti mezi regiony
    """

    return get_line(a, b, dir).length


def crosses_another_polygon(regions: {StubSpatial}, a, b, dir=None, check_cross=None):
    """
    Funkce kontroluj, zda usecka mezi 'a' a 'b' neprochazi jinym regionem
    """

    line = get_line(regions[a], regions[b], dir)
    check_cross = check_cross if check_cross else regions

    for r in check_cross:
        if r in [a, b]:
            continue

        if line.crosses(regions[r].get_box()):
            return True

    return False


def find_transitivity(candidates: [tuple]):
    """
    Pomocna funkce pro overeni tranzitivity
    """

    for i, source in enumerate(candidates):
        for j, candidate in enumerate(candidates):
            if i == j:
                continue

            if source[-1] == candidate[0]:
                return i, j

    return None


def transitivity_groups(candidates: [tuple]) -> list:
    """
    Funkce vytvori seznam dvojic, ktere splnuji tranzitivni relaci
    """

    candidates_cp = copy(candidates)
    merging = find_transitivity(candidates)

    while merging:
        i, j = merging
        source = candidates_cp[i]
        target = candidates_cp[j]

        del candidates_cp[i]
        del candidates_cp[candidates_cp.index(target)]

        candidates_cp.append(source + target)
        merging = find_transitivity(candidates_cp)

    return [list(set(x)) for x in candidates_cp]


class Relations(object):
    """
    Pomocna trida, ktera udrzuje vztahy mezi regiony a nabizi metody pro jejich overeni
    """

    def __init__(self, relations):
        self.relations = relations

    def precedes(self, axis, a, b):
        return self._check(axis, a, b, PRECEDES)

    def i_precedes(self, axis, a, b):
        return self._check(axis, a, b, IPRECEDES)

    def meets(self, axis, a, b):
        return self._check(axis, a, b, MEETS)

    def i_meets(self, axis, a, b):
        return self._check(axis, a, b, IMEETS)

    def overlaps(self, axis, a, b):
        return self._check(axis, a, b, OVERLAPS)

    def i_overlaps(self, axis, a, b):
        return self._check(axis, a, b, IOVERLAPS)

    def starts(self, axis, a, b):
        return self._check(axis, a, b, STARTS)

    def i_starts(self, axis, a, b):
        return self._check(axis, a, b, ISTARTS)

    def during(self, axis, a, b):
        return self._check(axis, a, b, DURING)

    def i_during(self, axis, a, b):
        return self._check(axis, a, b, IDURING)

    def finishes(self, axis, a, b):
        return self._check(axis, a, b, FINISHES)

    def i_finishes(self, axis, a, b):
        return self._check(axis, a, b, IFINISHES)

    def equals(self, axis, a, b):
        return self._check(axis, a, b, EQUALS)

    def get_relation(self, axis, a, b):
        return self.relations[a][b][axis]

    def _check(self, axis, a, b, relation):
        return self.get_relation(axis, a, b) == relation


    def is_vertical_before_in_reading(self, a, b):
        """
        Tato metoda byla s mirnou modifikaci prevzata z clanku Document Understanding for a Broad Class of Documents
        https://www.researchgate.net/publication/2527078_Document_Understanding_for_a_Broad_Class_of_Documents

        Metoda pro overeni posloupnosti cteni mezi dvema elementy ve vertikalnim smeru
        """
        if self.precedes('x', a, b):
            return True

        if self.meets('x', a, b):
            return True

        if self.overlaps('x', a, b):
            if self.precedes('y', a, b):
                return True

            if self.meets('y', a, b):
                return True

            if self.overlaps('y', a, b):
                return True

        if self.precedes('y', a, b) or self.meets('y', a, b) or self.overlaps('y', a, b):
            if self.precedes('x', a, b):
                return True

            if self.meets('x', a, b):
                return True

            if self.overlaps('x', a, b):
                return True

            if self.starts('x', a, b):
                return True

            if self.i_finishes('x', a, b):
                return True

            if self.equals('x', a, b):
                return True

            if self.during('x', a, b):
                return True

            if self.i_during('x', a, b):
                return True

            if self.finishes('x', a, b):
                return True

            if self.i_starts('x', a, b):
                return True

            if self.i_overlaps('x', a, b):
                return True

        return False

    def is_horizontal_before_in_reading(self, a, b):
        """
        Tato metoda byla s mirnou modifikaci prevzata z clanku Document Understanding for a Broad Class of Documents
        https://www.researchgate.net/publication/2527078_Document_Understanding_for_a_Broad_Class_of_Documents

        Metoda pro overeni posloupnosti cteni mezi dvema elementy v horizontalnim smeru
        """

        if self.precedes('y', a, b):
            return True

        if self.meets('y', a, b):
            return True

        if self.overlaps('y', a, b):
            if self.precedes('x', a, b):
                return True

            if self.meets('x', a, b):
                return True

            if self.overlaps('x', a, b):
                return True

        if self.precedes('x', a, b) or self.meets('x', a, b) or self.overlaps('x', a, b):
            if self.precedes('y', a, b):
                return True

            if self.meets('y', a, b):
                return True

            if self.overlaps('y', a, b):
                return True

            if self.starts('y', a, b):
                return True

            if self.i_finishes('y', a, b):
                return True

            if self.equals('y', a, b):
                return True

            if self.during('y', a, b):
                return True

            if self.i_during('y', a, b):
                return True

            if self.finishes('y', a, b):
                return True

            if self.i_starts('y', a, b):
                return True

            if self.i_overlaps('y', a, b):
                return True

        return False


class TBRR(object):
    """
    Base trida TBRR pro vypocet vztahu mezi regiony
    """

    methods = [
        EQUALS,
        PRECEDES,
        MEETS,
        OVERLAPS,
        STARTS,
        DURING,
        FINISHES,
    ]

    def __init__(self, T: float):
        self.T = T

    def call_method(self, method, direction, a: StubSpatial, b: StubSpatial, T=None):
        if T is None:
            T = self.T

        if method not in self.methods:
            raise Exception('Unknown method "{}"'.format(method))

        if direction == 'x':
            min_a = a.min_x()
            min_b = b.min_x()
            max_a = a.max_x()
            max_b = b.max_x()
        else:
            min_a = a.min_y()
            min_b = b.min_y()
            max_a = a.max_y()
            max_b = b.max_y()

        if method in ['precedes', 'meets']:
            return globals()[method](min_b, max_a, T)
        else:
            return globals()[method](min_a, min_b, max_a, max_b, T)


class DocTBRR(TBRR):
    """
    Trida DocTBRR vypocitava vztahy mezi regiony, hleda sloupce a nabizi podpurne metody pro prostorovou analyzu
    """

    methods = [
        EQUALS,
        PRECEDES,
        MEETS,
        OVERLAPS,
        STARTS,
        DURING,
        FINISHES,
    ]

    def __init__(self, doc: StubDocument):
        self.doc = doc
        super().__init__(calculate_T(self.doc.get_text_regions()))
        self.col_horizontal_T = calculate_horizontal_T(self.doc.get_text_regions(), self.T)
        self.columns = None
        self.relations = None
        self.neighborhood = None

    def get_cols_and_independent_regions(self):
        """
        Metoda vrati dict se sloupci a s regiony, ktere se v techto sloupcich nenachazi
        """

        return {**self.get_columns(), **self.get_independent_regions()}

    def get_relations(self) -> 'Relations':
        """
        Vypocet vztahu
        """

        if self.relations:
            return self.relations

        cols = self.get_columns()
        text_regions = self.doc.get_text_regions()
        self.relations = self._build_relations({**text_regions, **cols}, self.T, self.T)
        return self.relations

    def _get_relation(self, a: StubSpatial, b: StubSpatial, h_T, v_T) -> (str, str):
        """
        Protected metoda pro ziskani vztahu mezi dvema elementy
        Vraci vztah pro oba elementy
        """

        x, y = self._relation(a, b, h_T, v_T)

        if x == UNKNOWN or y == UNKNOWN:
            ix, iy = self._relation(b, a, h_T, v_T)

            if x == UNKNOWN:
                x = self._get_inverse(ix)

            if y == UNKNOWN:
                y = self._get_inverse(iy)

        return x, y

    def _relation(self, a: StubSpatial, b: StubSpatial, h_T, v_T) -> (str, str):
        """
        Protected metoda pro ziskani vztahu mezi dvema elementy
        """

        x = UNKNOWN
        y = UNKNOWN

        for method in self.methods:
            if self.call_method(method, 'x', a, b, h_T):
                x = method
                break

        for method in self.methods:
            if self.call_method(method, 'y', a, b, v_T):
                y = method
                break

        return x, y

    def _get_inverse(self, relation) -> str:
        """
        Definice inverzniho vztahu
        """

        if 'i_' in relation:
            return relation.replace('i_', '')

        return 'i_' + relation

    def _build_relations(self, regions: {StubSpatial}, h_T, v_T):
        """
        Metoda, ktera pocita strukturu (matici) vztahu
        """

        relations = {}

        for i in regions:
            source = regions[i]

            if i not in relations:
                relations[i] = {}

            for j in regions:
                if j in relations and i in relations[j]:
                    # zpracovany prvek preskakuju
                    continue

                if j not in relations:
                    relations[j] = {}

                # inmit
                relations[i][j] = {'x': UNKNOWN, 'y': UNKNOWN}
                relations[j][i] = {'x': UNKNOWN, 'y': UNKNOWN}
                target = regions[j]

                # vypocet vztahu
                x, y = self._get_relation(source, target, h_T, v_T)
                relations[i][j] = {'x': x, 'y': y}

                # definice inverznich vztahu
                if x == EQUALS:
                    relations[j][i]['x'] = EQUALS
                else:
                    relations[j][i]['x'] = self._get_inverse(x)

                if y == EQUALS:
                    relations[j][i]['y'] = EQUALS
                else:
                    relations[j][i]['y'] = self._get_inverse(y)

        return Relations(relations)

    def is_col(self, a):
        """
        Kontroluje, zda prvek 'a' je sloupcem
        """

        return a in self.get_columns()

    def get_columns(self) -> {'Column'}:
        """
        Metoda pro vypocet sloupcu
        """

        if self.columns:
            return self.columns

        Column.C = 0
        text_regions = self.doc.get_text_regions()
        r = self._build_relations(text_regions, self.col_horizontal_T, self.T)
        all_regions = self.doc.get_regions()
        neighborhood = get_neighborhood(self.doc.get_text_regions(), self.doc.get_box())

        candidates = []
        distances = []
        cols = {}

        relations = [EQUALS]

        for i in text_regions:
            for j in text_regions:
                if i == j:
                    continue

                # pokud jsou dva prvky na x ose EQUALS
                # a pokud spolu sousedi
                # a pokud respektuji posloupnost cteni
                # pak je pridam mezi kandidaty, ktere nejspis patri do sloupce - bude dale analyzovano
                if r.get_relation('x', i, j) in relations \
                        and j in neighborhood[i] \
                        and r.is_vertical_before_in_reading(i, j) \
                        and r.is_horizontal_before_in_reading(i, j):
                    dir = get_direction(r.get_relation('x', i, j), r.get_relation('y', i, j))

                    candidates.append((i, j))
                    distances.append(get_distance(text_regions[i], text_regions[j], dir))

        # vypocet limitni vzdalenosti
        distance_limit = np.std(distances) + np.mean(distances)
        to_remove = []

        for (a, b) in candidates:
            dir = get_direction(r.get_relation('x', a, b), r.get_relation('y', a, b))

            # pokud vzdalenosti elementu prekrocily limitni delku
            # nebo pokud usecka mezi nimi protina jiny region, pak jej z kandidatu odstranim
            if get_distance(text_regions[a], text_regions[b], dir) > distance_limit or crosses_another_polygon(
                    all_regions, a, b):
                to_remove.append((a, b))

        candidates = [c for c in candidates if c not in to_remove]

        # vytvoreni skupin, ktere respektuji tranzitivni relaci
        groups = transitivity_groups(candidates)
        for i, _ in enumerate(groups):
            groups[i] = sorted(groups[i], key=lambda x: text_regions[x].centroid().y)

        # pro kazdou skupinu vytvorim sloupec, do ktereho zaradim prvky ze skupiny
        for col in groups:
            column = Column({id: text_regions[id] for id in col})
            cols[column.get_id()] = column

        self.columns = cols
        return cols

    def get_col(self, a) -> Optional['Column']:
        """
        Metoda pro prvek 'a' vraci jeho sloupec
        """

        cols = self.get_columns()

        if a in cols:
            return cols[a]

        for c in cols:
            col = cols[c]
            if col.contains(a):
                return col

        return None

    def is_in_col(self, id) -> bool:
        """
        Kontrola, zda je prvek ve sloupci
        """

        return id in self.doc.get_text_regions() and self.get_col(id) is not None

    def get_independent_regions(self):
        """
        Metoda, ktera vraci regiony, ktere nejsou ve sloupcich
        """

        regions = {}

        for r in self.doc.get_text_regions():
            if not self.is_in_col(r):
                regions[r] = self.doc.get_text_regions()[r]

        return regions

    def is_last_in_col(self, a) -> bool:
        """
        Metoda, ktera rika, zda je prvek 'a' na konci sloupce
        """

        if not self.is_in_col(a):
            return False

        col = self.get_col(a)
        return col.get_region_ids()[-1] == a


class Column(StubTextRegion):
    """
    Definice sloupce
    """

    C = 0

    def __init__(self, regions: {StubTextRegion}):
        super().__init__()
        self.regions = regions
        self.id = 'c' + str(Column.C)
        Column.C += 1

    def get_id(self):
        """
        Kazdy sloupec ma sve jednoznacne id
        """
        return self.id

    def get_coords(self) -> [tuple]:
        """
        Metoda pro ziskani souradnic polygonu sloupce.
        Ty jsou definovany hranicemi regionu uvnitr sloupce
        """

        coords = []

        for r in self.regions:
            coords += self.regions[r].get_coords()

        return coords

    def get_text(self) -> str:
        """
        Vrati text sloupce.
        Ten se sklada ze vsech regionu uvnitr sloupce
        """

        text = []

        for id in self.get_region_ids():
            text.append(self.regions[id].get_text())

        return ' '.join(text)

    def get_first_region_id(self):
        return self.get_region_ids()[0]

    def get_region_ids(self):
        return list(self.regions.keys())

    def contains(self, a):
        return a in self.get_region_ids()

    def get_regions(self) -> {StubTextRegion}:
        return self.regions

    def get_color(self):
        return 'b'
