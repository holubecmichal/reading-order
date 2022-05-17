import shapely.geometry
from shapely.geometry import Polygon, Point


class Spatial(object):
    """
    Definice metod pro podporu prostorove analyzy, s predpripravenym zpracovanim
    """

    def __init__(self):
        self._polygon = None
        self._box = None
        self._box_bounds = None
        self._centroid = None

    def get_coords(self) -> [tuple]:
        ...

    def get_polygon(self) -> Polygon:
        """
        Vraci polygon elementu, respektive hranice elementu
        """

        if self._polygon is None:
            self._polygon = Polygon(self.get_coords())

        return self._polygon

    def get_box(self) -> Polygon:
        """
        Polygon prevedeny na obdelnik
        """

        if self._box is None:
            self._box = shapely.geometry.box(*self.get_polygon().bounds)

        return self._box

    def min_x(self):
        return self._get_box_bounds()[0]

    def min_y(self):
        return self._get_box_bounds()[1]

    def max_x(self):
        return self._get_box_bounds()[2]

    def max_y(self):
        return self._get_box_bounds()[3]

    def centroid(self):
        if self._centroid is None:
            self._centroid = self.get_box().centroid

        return self._centroid

    def most_left_centroid(self):
        centroid = self.centroid()
        return Point(self.min_x(), centroid.y)

    def _get_box_bounds(self):
        if self._box_bounds is None:
            self._box_bounds = self.get_box().bounds

        return self._box_bounds
