import random
from collections import defaultdict
from copy import copy
from graphlib import TopologicalSorter
from typing import Optional

import numpy as np
import shapely
import torch
from shapely.geometry import LineString, MultiPoint, Polygon, Point
from shapely.ops import voronoi_diagram


from document.stubs import Document as StubDocument, TextRegion as StubTextRegion
from reading_order.processor import Processor
from reading_order.reading_order import ReadingOrder
from spatial.stubs import Spatial as StubSpatial, Region as StubRegion
import jenkspy

from utils.sorting import topological_sort, original_topological_sort

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


def calculate_T(regions: {StubSpatial}):
    smallest = None

    for i in regions:
        region = regions[i]

        if smallest is None or region.get_box().area < smallest.get_box().area:
            smallest = region

    points = list(zip(list(smallest.get_box().boundary.xy[0]), list(smallest.get_box().boundary.xy[1])))

    shortest = None
    for i, fst in enumerate(points):
        if i == len(points) - 1:
            break

        snd = points[i + 1]
        line = LineString([fst, snd])

        if shortest is None or line.length < shortest.length:
            shortest = line

    return shortest.length / 2


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


class UnknownRelation(Exception): ...


class UnknownMethod(Exception): ...


class TBRR(object):
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

    def precedes_x(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('precedes', 'x', a, b)

    def precedes_y(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('precedes', 'y', a, b)

    def meets_x(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('meets', 'x', a, b)

    def meets_y(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('meets', 'y', a, b)

    def overlaps_x(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('overlaps', 'x', a, b)

    def overlaps_y(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('overlaps', 'y', a, b)

    def starts_x(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('starts', 'x', a, b)

    def starts_y(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('starts', 'y', a, b)

    def during_x(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('during', 'x', a, b)

    def during_y(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('during', 'y', a, b)

    def finishes_x(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('finishes', 'x', a, b)

    def finishes_y(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('finishes', 'y', a, b)

    def equals_x(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('equals', 'x', a, b)

    def equals_y(self, a: StubSpatial, b: StubSpatial):
        return self._call_method('equals', 'y', a, b)

    def _call_method(self, method, direction, a: StubSpatial, b: StubSpatial):
        if method not in self.methods:
            raise UnknownMethod()

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
            return globals()[method](min_b, max_a, self.T)
        else:
            return globals()[method](min_a, min_b, max_a, max_b, self.T)

    def get_relation(self, a: StubSpatial, b: StubSpatial) -> (str, str):
        x, y = self._relation(a, b)

        if x == UNKNOWN or y == UNKNOWN:
            ix, iy = self._relation(b, a)

            if x == UNKNOWN:
                x = self._get_inverse(ix)

            if y == UNKNOWN:
                y = self._get_inverse(iy)

        return x, y

    def _relation(self, a: StubSpatial, b: StubSpatial) -> (str, str):
        x = UNKNOWN
        y = UNKNOWN

        for method in self.methods:
            if self._call_method(method, 'x', a, b):
                x = method
                break

        for method in self.methods:
            if self._call_method(method, 'y', a, b):
                y = method
                break

        return x, y

    def _get_inverse(self, relation) -> str:
        if 'i_' in relation:
            return relation.replace('i_', '')

        return 'i_' + relation

    def get_relations(self, regions: {StubSpatial}):
        relations = {}

        for i in regions:
            source = regions[i]

            if i not in relations:
                relations[i] = {}

            for j in regions:
                if j in relations and i in relations[j]:
                    continue

                if j not in relations:
                    relations[j] = {}

                relations[i][j] = {'x': UNKNOWN, 'y': UNKNOWN}
                relations[j][i] = {'x': UNKNOWN, 'y': UNKNOWN}
                target = regions[j]

                x, y = self.get_relation(source, target)
                relations[i][j] = {'x': x, 'y': y}

                if x == EQUALS:
                    relations[j][i]['x'] = EQUALS
                else:
                    relations[j][i]['x'] = self._get_inverse(x)

                if y == EQUALS:
                    relations[j][i]['y'] = EQUALS
                else:
                    relations[j][i]['y'] = self._get_inverse(y)

        return relations

    def get_columns(self, text_regions: {StubTextRegion}, all_regions: {StubRegion}, neighborhood) -> {'Column'}:
        Column.C = 0
        r = Relations(self.get_relations(text_regions))
        candidates = []
        distances = []
        cols = {}

        for i in text_regions:
            for j in text_regions:
                if i == j:
                    continue

                if r.relations[i][j]['x'] == EQUALS \
                        and j in neighborhood[i] \
                        and r.is_vertical_before_in_reading(i, j) \
                        and r.is_horizontal_before_in_reading(i, j):

                    dir = get_direction(r.relations[i][j]['x'], r.relations[i][j]['y'])

                    candidates.append((i, j))
                    distances.append(get_distance(text_regions[i], text_regions[j], dir))

        distance_limit = np.std(distances) + np.mean(distances) * 2
        to_remove = []

        for (a, b) in candidates:
            dir = get_direction(r.relations[a][b]['x'], r.relations[a][b]['y'])

            if get_distance(text_regions[a], text_regions[b], dir) > distance_limit or crosses_another_polygon(all_regions, a, b):
                to_remove.append((a, b))

        candidates = [c for c in candidates if c not in to_remove]

        groups = transitivity_groups(candidates)
        for i, _ in enumerate(groups):
            groups[i] = sorted(groups[i], key=lambda x: text_regions[x].centroid().y)

        for col in groups:
            column = Column({id: text_regions[id] for id in col})
            cols[column.get_id()] = column

        return cols


def get_centroid_multipoint(regions: {StubSpatial}) -> MultiPoint:
    centroids = [regions[x].centroid() for x in regions]
    return MultiPoint(centroids)


def get_most_left_centroid_multipoint(regions: {StubSpatial}) -> MultiPoint:
    centroids = [regions[x].most_left_centroid() for x in regions]
    return MultiPoint(centroids)


def get_voronoi_polygons(regions: {StubSpatial}, centroid='default') -> [Polygon]:
    if centroid == 'left':
        centroids = get_most_left_centroid_multipoint(regions)
    else:
        centroids = get_centroid_multipoint(regions)

    voronoi = voronoi_diagram(centroids)
    geoms = list(voronoi.geoms)
    polygons = {}

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
    for i in polygons:
        if not box.contains(polygons[i]):
            result = shapely.ops.split(polygons[i], box.boundary)
            for splitted in result.geoms:
                if box.contains(splitted):
                    polygons[i] = splitted
                    break

    return polygons


def get_direction(x, y):
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


def get_neighborhood(regions: {StubSpatial}, box: Polygon) -> dict[list]:
    polygons = get_voronoi_polygons(regions, 'left')
    polygons = cutout_by_box(polygons, box)
    neighborhood = {}

    for i in polygons:
        neighborhood[i] = []

        for j in polygons:
            if polygons[i].touches(polygons[j]):
                neighborhood[i].append(j)

    return neighborhood


def get_density(regions: {StubTextRegion}):
    density = {}

    for i in regions:
        region = regions[i]
        density[i] = region.get_box().area / len(region.get_text())

    return density


def get_headers(regions: {StubTextRegion}):
    density = get_density(regions)
    values = np.asarray(list(density.values()))
    limit = values.mean() + values.std()
    headers = {}

    for r in density:
        if density[r] > limit:
            headers[r] = regions[r]

    return headers


def get_page_number(regions: {StubTextRegion}, relations) -> Optional[StubTextRegion]:
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
    return get_line(a, b, dir).length


def crosses_another_polygon(regions: {StubRegion}, a, b):
    line = get_line(regions[a], regions[b])

    for r in regions:
        if r in [a, b]:
            continue

        if line.crosses(regions[r].get_box()):
            return True

    return False


class Relations(object):
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

    def _check(self, axis, a, b, relation):
        return self.relations[a][b][axis] == relation

    def is_vertical_before_in_reading(self, a, b):
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

    def is_before_in_reading(self, a, b):
        if self.precedes('x', a, b):
            return True

        if self.precedes('y', a, b):
            return True

        if self.meets('x', a, b):
            return True

        if self.meets('y', a, b):
            return True

        if self.overlaps('x', a, b):
            return True

        if self.overlaps('y', a, b):
            return True

        return False


def find_transitivity(candidates: [tuple]):
    for i, source in enumerate(candidates):
        for j, candidate in enumerate(candidates):
            if i == j:
                continue

            if source[-1] == candidate[0]:
                return i, j

    return None


def transitivity_groups(candidates: [tuple]) -> list:
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


def transitivity_groups_tuples(candidates: [tuple]) -> [tuple]:
    identified_groups = transitivity_groups(candidates)

    if len(identified_groups) == 1:
        return [candidates]

    groups = [[] for _ in identified_groups]
    for (a, b) in candidates:
        for i, pair in enumerate(identified_groups):
            if a in pair:
                groups[i].append((a, b))

    return groups



class SpaRe(object):
    def __init__(self, xml: StubDocument):
        self.xml = xml
        self.tbrr = TBRR(calculate_T(xml.get_text_regions()))
        self.neighborhood = get_neighborhood(self.xml.get_text_regions(), self.xml.get_box())
        self.cols = self.tbrr.get_columns(xml.get_text_regions(), xml.get_regions(), self.neighborhood)
        self.r = Relations(self.tbrr.get_relations(xml.get_text_regions() | self.cols))

    def get_region_by_id(self, id) -> StubRegion:
        return self.xml.get_regions()[id]

    def get_text_region_by_id(self, id) -> StubTextRegion:
        return self.xml.get_text_regions()[id]

    def get_neighborhood(self):
        return self.neighborhood

    def get_neighbours(self, id):
        return self.get_neighborhood()[id]

    def get_relations(self):
        return self.r.relations

    def is_vertical_before_in_reading(self, a, b):
        return self.r.is_vertical_before_in_reading(a, b)

    def is_horizontal_before_in_reading(self, a, b):
        return self.r.is_horizontal_before_in_reading(a, b)

    def is_before_in_reading(self, a, b):
        return self.r.is_before_in_reading(a, b)

    def get_distance(self, a, b):
        return get_distance(self.get_region_by_id(a), self.get_region_by_id(b), self.get_direction(a, b))

    def get_direction(self, a, b):
        relation = self.get_relations()[a][b]
        return get_direction(relation['x'], relation['y'])

    def get_col(self, a) -> Optional['Column']:
        if a in self.cols:
            return self.cols[a]

        for c in self.cols:
            col = self.cols[c]
            if col.contains(a):
                return col

        return None

    def is_in_col(self, a) -> bool:
        return self.is_text_region(a) and self.get_col(a) is not None

    def is_last_in_col(self, a) -> bool:
        if not self.is_in_col(a):
            return False

        col = self.get_col(a)
        return col.get_region_ids()[-1] == a

    def is_col(self, a):
        return a in self.cols

    def is_text_region(self, a):
        return a in self.xml.get_regions()

    def get_regions_not_in_columns(self):
        regions = {}

        for r in self.xml.get_text_regions():
            if not self.is_in_col(r):
                regions[r] = self.get_region_by_id(r)

        return regions

    def get_columns(self):
        return self.cols

    def get_headings(self):
        return get_headers(self.xml.get_text_regions())

    def get_element(self, a) -> StubTextRegion:
        if self.is_col(a):
            return self.get_col(a)

        return self.get_text_region_by_id(a)

    def get_original_diagonal_reading_order(self):
        before_in_reading = []

        for a in self.xml.get_text_regions():
            for b in self.xml.get_text_regions():
                if a == b:
                    continue

                if self.is_vertical_before_in_reading(a, b) and self.is_horizontal_before_in_reading(a, b):
                    before_in_reading.append((a, b))

        chain = list(self._original_topological_sort(before_in_reading))
        return self._chain_to_reading_order(chain)

    def get_diagonal_reading_order(self):
        before_in_reading = []

        for a in self.xml.get_text_regions():
            for b in self.xml.get_text_regions():
                if a == b:
                    continue

                if self.is_vertical_before_in_reading(a, b) and self.is_horizontal_before_in_reading(a, b):
                    before_in_reading.append((a, b))

        chain = list(self._topological_sort(before_in_reading))
        return self._chain_to_reading_order(chain)

    def get_columnar_reading_order(self):
        before_in_reading = []

        regions = self.get_regions_not_in_columns() | self.get_columns()
        for a in regions:
            for b in regions:
                if a == b:
                    continue

                vertical = self.is_vertical_before_in_reading(a, b)
                horizontal = self.is_horizontal_before_in_reading(a, b)

                if vertical and horizontal:
                    before_in_reading.append((a, b))

        chain = []
        for i in self._topological_sort(before_in_reading):
            if self.is_col(i):
                chain += self.get_col(i).get_region_ids()
            else:
                chain.append(i)

        return self._chain_to_reading_order(chain)

    def get_columnar_heading_reading_order(self):
        headers_reading = []
        headers = self.get_headings()
        for h in headers:
            candidates = []
            neighbours = self.get_neighbours(h)

            for n in neighbours:
                if (self.r.precedes('y', h, n) or self.r.meets('y', h, n)) and (
                        self.r.equals('x', h, n)
                        or self.r.during('x', h, n)
                        or self.r.starts('x', h, n)
                        or self.r.finishes('x', h, n)
                        or self.r.overlaps('x', h, n)
                        or self.r.i_overlaps('x', h, n)
                        or self.r.i_starts('x', h, n)
                ):
                    candidates.append(n)

            if len(candidates) > 1:
                candidates = sorted(candidates, key=lambda x: self.get_region_by_id(x).min_x())

            if candidates:
                headers_reading.append((h, candidates[0]))

        reading_order = self.get_columnar_reading_order()
        before_in_reading = reading_order.get_before_in_reading()
        chain = self._topological_sort(before_in_reading)

        limit = 200
        loops = 0

        changed = True
        while changed:
            changed = False

            graph = []
            a = chain[0]
            for i in range(1, len(chain)):
                b = chain[i]
                graph.append((a, b))
                a = b

            for i, (header, succ) in enumerate(headers_reading):
                (a, b) = [item for item in graph if item[0] == header][0]

                if succ != b:
                    changed = True
                    index = graph.index((a, b))
                    graph[index] = (header, succ)

                    (a, b) = [item for item in graph if item[1] == succ and item[0] != header][0]
                    index = graph.index((a, b))
                    graph[index] = (a, header)

            chain = self._topological_sort(graph)
            loops += 1

            if loops == limit:
                print('break joining loops')
                break

        return self._chain_to_reading_order(chain)

    def get_columnar_reading_order3(self):
        reading_order = self.get_columnar_heading_reading_order()
        graph = reading_order.get_before_in_reading()

        for c in self.get_columns():
            column = self.get_columns()[c]
            ids = column.get_region_ids()

            a = ids[0]
            for i in range(1, len(ids)):
                b = ids[i]
                if (a, b) not in graph:
                    graph = [x for x in graph if x[0] != a]
                    graph = [x for x in graph if x[1] != b]
                    graph.insert(0, (a, b))

                a = b

        chain = self._topological_sort(graph)
        return self._chain_to_reading_order(chain)

    def get_columnar_lm_reading_order(self, processor: Processor):
        reading_order = self.get_columnar_reading_order3()
        before_in_reading = reading_order.get_before_in_reading()
        new_connections = []
        already_removed = []
        headings = self.get_headings()

        last_in_cols = [x for x in self.xml.get_text_regions() if self.is_last_in_col(x)]
        for el in last_in_cols:
            x_candidates = []
            y_candidates = []
            source_col = self.get_col(el).get_id()
            for target_col in self.xml.get_text_regions() | self.get_columns():
                if (self.r.equals('x', source_col, target_col)
                    or self.r.starts('x', source_col, target_col)
                    or self.r.i_starts('x', source_col, target_col)
                ) and (
                    self.r.precedes('y', source_col, target_col)
                    or self.r.meets('y', source_col, target_col)
                ):
                    x_candidates.append(target_col)

                if self.is_col(target_col) and self.r.precedes('x', source_col, target_col) and (
                    self.r.equals('y', source_col, target_col)
                    or self.r.during('y', source_col, target_col)
                    or self.r.finishes('y', source_col, target_col)
                    or self.r.i_starts('y', source_col, target_col)
                    or self.r.i_precedes('y', source_col, target_col)
                    or self.r.i_overlaps('y', source_col, target_col)
                ):
                    y_candidates.append(target_col)

            if x_candidates or y_candidates:
                candidates = []

                if x_candidates:
                    candidate_x = sorted(x_candidates, key=lambda x: self.get_element(x).min_y())[0]
                    c_regions = copy(headings)
                    c_regions[el] = self.get_region_by_id(el)
                    c_regions[candidate_x] = self.get_element(candidate_x)

                    if not crosses_another_polygon(c_regions, el, candidate_x):
                        candidates.append(candidate_x)

                if y_candidates:
                    candidate_y = sorted(y_candidates, key=lambda y: self.get_columns()[y].min_x())[0]
                    min_x = self.get_col(candidate_y).min_x()
                    min = min_x - self.tbrr.T
                    max = min_x + self.tbrr.T

                    for candidate in y_candidates:
                        c_regions = copy(headings)
                        c_regions[el] = self.get_region_by_id(el)
                        c_regions[candidate] = self.get_col(candidate)

                        if min < self.get_col(candidate).min_x() < max and not crosses_another_polygon(c_regions, el, candidate):
                            candidates.append(candidate)

                if not candidates:
                    # nenaslo se nic, na co by se mohl element napojit, pokracuju dal
                    continue

                # actual_connection = [x for x in before_in_reading if x[0] == el][0]

                # if actual_connection[1] in [self.get_col(x).get_first_region_id() for x in candidates]:
                #     # tady uz je to napojeno spravne
                #     continue

                source_col = self.get_col(source_col)

                data = {}
                actual_connection = self._find_by_fst(el, before_in_reading)
                if actual_connection:
                    actual_connection = actual_connection[0]
                    data = {actual_connection[1]: self.get_text_region_by_id(actual_connection[1])}

                for c in candidates:
                    data[c] = self.get_element(c)

                probs = processor.estimate(source_col, data, 10)
                max_index = torch.tensor(list(probs.values())).argmax()
                keys = list(data.keys())
                candidate = keys[max_index]

                if not actual_connection or candidate != actual_connection[1]:
                    if self.is_col(candidate):
                        candidate = self.get_col(candidate).get_first_region_id()

                    new_connection = (el, candidate)

                    if actual_connection:
                        before_in_reading.remove(actual_connection)
                        already_removed.append(actual_connection)

                    for pair in self._find_by_snd(candidate, before_in_reading):
                        if pair not in new_connections:
                            before_in_reading.remove(pair)
                            already_removed.append(pair)

                    if new_connection not in already_removed:
                        before_in_reading.insert(0, new_connection)
                        new_connections.append(new_connection)

        groups = transitivity_groups_tuples(before_in_reading)
        groups = sorted(groups, key=lambda x: len(x), reverse=True)

        before_in_reading = []
        for group in groups:
            before_in_reading += group

        chain = self._topological_sort(before_in_reading)
        return self._chain_to_reading_order(chain)

    def get_top_to_bottom_reading_order(self) -> ReadingOrder:
        regions = self.xml.get_text_regions()
        chain = sorted(regions, key=lambda i: (regions[i].min_y(), regions[i].min_x()))
        return self._chain_to_reading_order(chain)

    def get_ocr_reading_order(self):
        regions = self.xml.get_text_regions()
        chain = [i for i in regions]
        return self._chain_to_reading_order(chain)

    def _topological_sort(self, before_in_reading: [tuple]):
        return topological_sort(before_in_reading)

    def _original_topological_sort(self, before_in_reading: [tuple]):
        return original_topological_sort(before_in_reading)

    def _chain_to_reading_order(self, chain) -> ReadingOrder:
        reading_order = ReadingOrder()
        group = reading_order.add_ordered_group()

        regions = self.xml.get_text_regions()
        source = chain[0]
        for successor in chain[1:]:
            group.add_candidates(regions[source], regions[successor])
            source = successor

        return reading_order

    def _find_by_fst(self, key, values: [tuple]):
        return [x for x in values if x[0] == key]

    def _find_by_snd(self, key, values: [tuple]):
        return [x for x in values if x[1] == key]

class Column(StubTextRegion):
    C = 0

    def __init__(self, regions: {StubTextRegion}):
        super().__init__()
        self.regions = regions
        self.id = 'c' + str(Column.C)
        Column.C += 1

    def get_id(self):
        return self.id

    def get_coords(self) -> [tuple]:
        coords = []

        for r in self.regions:
            coords += self.regions[r].get_coords()

        return coords

    def get_text(self) -> str:
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

    def get_before_in_reading(self) -> [tuple]:
        region_ids = self.get_region_ids()
        before_in_reading = []

        for i in range(len(region_ids)):
            if i + 1 == len(region_ids):
                break

            before_in_reading.append((region_ids[i], region_ids[i + 1]))

        return before_in_reading