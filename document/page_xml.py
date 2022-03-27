import re
from xml.etree import ElementTree

from reading_order.reading_order import from_page_xml_element, ReadingOrder
from .stubs import Document, TextRegion as StubTextRegion, Region as StubRegion
from utils.xml import element_schema, element_type
import shapely.geometry


class Region(StubRegion):
    def __init__(self, element: ElementTree.Element, schema: {str}):
        super().__init__()
        self.schema = schema
        self.element = element

    def get_id(self) -> str:
        return self.element.get('id')

    def get_coords(self):
        points = []

        if element_schema(self.element) == 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19':
            elements = self.element.findall('./Coords/Point', self.schema)
            for el in elements:
                points.append((int(el.get('x')), int(el.get('y'))))
        else:
            points = self.element.find('./Coords', self.schema).get('points').split()
            points = [tuple(map(int, x.split(','))) for x in points]

        return points

    def get_color(self):
        return 'orange'


class ImageRegion(Region):
    def get_color(self):
        return '#74ced4'


class SeparatorRegion(Region):
    def get_color(self):
        return '#fc54fc'


class TextRegion(StubTextRegion, Region):
    def get_text(self) -> str:
        text = self.element.find('./TextEquiv/Unicode', self.schema).text
        text = re.sub('-\n', '', text)
        text = re.sub('-$', '', text)
        text = re.sub('\n', ' ', text)
        return text

    def get_color(self):
        return '#7676da'


class PageXML(Document):
    def __init__(self, page: ElementTree.Element, schema: {str}):
        self.page = page
        self.schema = schema
        self.regions = None
        self.text_regions = None

    def get_regions(self) -> {Region}:
        if self.text_regions is not None:
            return self.regions

        regions = {}

        for el in self.page.findall('./'):
            type = element_type(el)

            if 'Region' not in type:
                continue

            if type == 'TextRegion':
                region = TextRegion(el, self.schema)
            elif type == 'ImageRegion':
                region = ImageRegion(el, self.schema)
            elif type == 'SeparatorRegion':
                region = SeparatorRegion(el, self.schema)
            else:
                region = Region(el, self.schema)

            regions[region.get_id()] = region

        self.regions = regions
        return self.regions

    def get_text_regions(self) -> {TextRegion}:
        if self.text_regions is not None:
            return self.text_regions

        all = self.get_regions()
        regions = {}

        for i in all:
            region = all[i]
            if isinstance(region, TextRegion):
                regions[i] = region

        self.text_regions = regions
        return self.text_regions

    def get_reading_order(self) -> ReadingOrder:
        reading_order = self.page.find('ReadingOrder', self.schema)
        return from_page_xml_element(reading_order)

    def get_image_width(self) -> int:
        return int(self.page.get('imageWidth'))

    def get_image_height(self) -> int:
        return int(self.page.get('imageHeight'))

    def get_box(self) -> shapely.geometry.Polygon:
        return shapely.geometry.box(0, 0, self.get_image_width(), self.get_image_height())


def parse(path) -> PageXML:
    page_tree = ElementTree.parse(path)
    schema = element_schema(page_tree.getroot())
    struct_schema = {'': schema}

    page = page_tree.find('Page', struct_schema)

    return PageXML(page, struct_schema)
