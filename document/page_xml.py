import re
import sys
from typing import Optional
from xml.etree import ElementTree

from reading_order.reading_order import from_page_xml_element, ReadingOrder
from .stubs import Document, TextRegion as StubTextRegion, Region as StubRegion
from utils.xml import element_schema, element_type
import shapely.geometry

SCHEMA = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/'


class Region(StubRegion):
    """
    Implementace Regionu pro PageXML
    """

    def __init__(self, element: ElementTree.Element, schema: {str}):
        super().__init__()
        self.schema = schema
        self.element = element

    def get_id(self) -> str:
        return self.element.get('id')

    def get_coords(self) -> list[tuple]:
        """
        Metoda pro nacteni souradnich regionu,
        """
        points = []

        # existuje dvoji ulozeni souradnic, ktere je definovano dle uziteho schematu xml
        if element_schema(self.element) == 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19':
            elements = self.element.findall('./Coords/Point', self.schema)
            for el in elements:
                points.append((int(el.get('x')), int(el.get('y'))))
        else:
            points = self.element.find('./Coords', self.schema).get('points').split()
            points = [tuple(map(int, x.split(','))) for x in points]

        return points

    def get_color(self):
        """Pomocna metoda pro vykresleni regionu skrze tridu Plotter"""
        return 'orange'


class ImageRegion(Region):
    """Region pokravajici obrazek"""

    def get_color(self):
        return '#74ced4'


class SeparatorRegion(Region):
    """Region pro oddelovac, standardne se jedna o element opticky oddelujici text, napr sloupce, clanky"""

    def get_color(self):
        return '#fc54fc'


class TextRegion(StubTextRegion, Region):
    """Textovy region"""

    def get_text(self) -> str:
        """
        Metoda pro nacteni textu.
        """

        texts = [i.text for i in self.element.findall('.//TextEquiv/Unicode', self.schema) if i.text is not None]

        text = ' '.join(texts)
        # odstraneni oddelovace
        text = re.sub('-\n\s*', '', text)
        text = re.sub('-$', '', text)
        # odstraneni konce radku a spojeni pres mezeru
        text = re.sub('\n\s*', ' ', text)
        return text

    def get_color(self):
        return '#7676da'


class PageXML(Document):
    """
    Trida PageXML slouzi jako obalka nad vstupni xml ve formatu PageXML
    """

    def __init__(self, page: ElementTree.Element, schema: {str}):
        self.page = page
        self.schema = schema
        self.regions = None
        self.text_regions = None

    def get_regions(self) -> {Region}:
        """
        Nacteni vsech regionu v PageXML
        """

        if self.text_regions is not None:
            return self.regions

        regions = {}

        for el in self.page.findall('./'):
            type = element_type(el)

            if 'Region' not in type:
                continue

            # rozdeleni dle jednotlivych typu
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
        """
        Nacteni textovych regionu v PageXML
        """
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

    def get_reading_order(self) -> Optional[ReadingOrder]:
        """
        Vraci instanci ReadingOrder v pripade, ze se v PageXML souboru nachazi element <ReadingOrder></ReadingOrder>
        """
        reading_order = self.page.find('ReadingOrder', self.schema)

        if not reading_order:
            return None

        return from_page_xml_element(reading_order)

    def get_image_width(self) -> int:
        return int(self.page.get('imageWidth'))

    def get_image_height(self) -> int:
        return int(self.page.get('imageHeight'))

    def get_box(self) -> shapely.geometry.Polygon:
        """
        Vraci polygon predstavujici hranici stranky
        """
        return shapely.geometry.box(0, 0, self.get_image_width(), self.get_image_height())


def parse(path) -> PageXML:
    """
    Parsovani xml struktury
    """

    try:
        page_tree = ElementTree.parse(path)
    except:
        print('File "{}" is not xml'.format(path), file=sys.stderr)
        exit(1)

    schema = element_schema(page_tree.getroot())

    if SCHEMA not in schema:
        print('Schema "{}" is not supported'.format(schema), file=sys.stderr)
        exit(1)

    struct_schema = {'': schema}
    page = page_tree.find('Page', struct_schema)

    return PageXML(page, struct_schema)
