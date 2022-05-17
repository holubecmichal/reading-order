from shapely.geometry import Polygon
from spatial.stubs import Spatial as SpatialStub


class Region(SpatialStub):
    """Region"""
    def get_id(self) -> str: ...


class TextRegion(Region):
    """Text region"""
    def get_text(self) -> str: ...


class Document(object):
    """
    Definice metod pro zpracovani vstupniho dokumentu
    """
    def get_regions(self) -> {Region}: ...

    def get_text_regions(self) -> {TextRegion}: ...

    def get_image_width(self) -> int: ...

    def get_image_height(self) -> int: ...

    def get_box(self) -> Polygon: ...
