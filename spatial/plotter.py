from matplotlib import pyplot as plt

from document.stubs import Document, TextRegion
from reading_order.reading_order import ReadingOrder
from spatial.spatial_data import get_voronoi_polygons, cutout_by_box, get_density, DocTBRR


class Plotter(object):
    """
    Plotter, trida pro pomocne vykresleni struktur
    """


    def __init__(self, document: Document):
        self.doc = document
        self.tbrr = DocTBRR(self.doc)
        self.ax = plt.subplot()

    def plot_document_border(self):
        """
        Vykresleni hranic dokumentu
        """

        self.ax.plot(*self.doc.get_box().exterior.xy)
        return self

    def plot_region_border(self, id, color):
        """
        Vykresleni konkretniho region v dane barve
        """

        region = self._get_element(id)
        self.ax.plot(*region.get_box().exterior.xy, color)

    def plot_all_regions(self, alpha=None):
        """
        Vykresleni vsech regionu, at uz textovych, tak i graficky
        """

        regions = self.doc.get_regions()

        for r in self.doc.get_regions():
            region = regions[r]
            self.ax.plot(*region.get_box().exterior.xy, region.get_color(), alpha=alpha)

        return self

    def plot_regions_not_in_columns(self):
        """
        Vykresli regiony, ktere se nenachazi ve sloupci
        """

        regions = self.doc.get_regions()

        for r in self.doc.get_regions():
            if self.tbrr.is_in_col(r):
                continue

            region = regions[r]
            self.ax.plot(*region.get_box().exterior.xy, region.get_color())

    def annotate_text_regions_not_in_columns(self):
        """
        Vypis id regionu, ktere nejsou ve sloupci
        """

        regions = self.doc.get_text_regions()

        for r in regions:
            if self.tbrr.is_in_col(r):
                continue

            region = regions[r]
            self.ax.annotate(r, (region.get_box().centroid.x, region.get_box().centroid.y), color=region.get_color(),
                             fontsize=8, ha='center', va='center')

        return self

    def annotate_text_regions(self, alpha=None):
        """
        Vypis id regionu
        """

        regions = self.doc.get_text_regions()

        for r in regions:
            region = regions[r]
            self.ax.annotate(r, (region.get_box().centroid.x, region.get_box().centroid.y), color=region.get_color(),
                             fontsize=8, ha='center', va='center', alpha=alpha)

        return self

    def plot_voronoi_text_regions(self):
        """
        Vykresleni Voroneho diagramu mezi textovymi regiony
        """

        polygons = get_voronoi_polygons(self.doc.get_text_regions())

        for i in cutout_by_box(polygons, self.doc.get_box()):
            plt.plot(*polygons[i].exterior.xy, 'r')

        return self

    def plot_regions_centroid(self):
        """
        Vykresleni bodu geometrickeho stredu textovych regionu
        """

        regions = self.doc.get_text_regions()

        for r in regions:
            region = regions[r]
            plt.plot(region.get_box().centroid.x, region.get_box().centroid.y, marker='.', color='r')

    def plot_voronoi_columns(self):
        """
        Vykresleni voroneho diagramu pro sloupce
        """

        polygons = get_voronoi_polygons(self.tbrr.get_cols_and_independent_regions())

        for i in cutout_by_box(polygons, self.doc.get_box()):
            plt.plot(*polygons[i].exterior.xy, 'r')

        return self

    def plot_columns(self, alpha=None):
        """
        Vykresleni sloupcu
        """

        cols = self.tbrr.get_columns()
        for c in cols:
            plt.plot(*cols[c].get_box().exterior.xy, cols[c].get_color(), alpha=alpha)

    def annotate_columns(self):
        """
        Vykresleni id sloupcu
        """

        cols = self.tbrr.get_columns()
        for c in cols:
            col = cols[c]
            self.ax.annotate(c, (col.get_box().centroid.x, col.get_box().centroid.y), color=col.get_color(),
                             fontsize=8, ha='center', va='center')

    def plot_reading_order(self, reading_order: ReadingOrder):
        """
        Vykresleni posloupnosti cteni
        """

        regions = self.doc.get_regions()

        for group in reading_order.get_ordered_groups():
            a = group.get_beginnings()[0]
            b = a.get_successor()

            is_first = True

            while b:
                a_reg = regions[a.get_id()]
                b_reg = regions[b.get_id()]
                self.ax.annotate("", xy=(b_reg.centroid().x, b_reg.centroid().y), xytext=(a_reg.centroid().x, a_reg.centroid().y),
                                 arrowprops=dict(arrowstyle="->", color='r'))

                if is_first:
                    plt.plot(a_reg.centroid().x, a_reg.centroid().y, marker="o", markersize=10, markeredgecolor="#00ff00", markerfacecolor="#00ff00")
                    is_first = False

                a = b
                b = b.get_successor()

                if b is None:
                    plt.plot(b_reg.centroid().x, b_reg.centroid().y, marker="o", markersize=10,
                             markeredgecolor="red", markerfacecolor="red")

    def _build(self):
        plt.axis('equal')
        plt.gca().invert_yaxis()

    def remove_axis(self):
        plt.axis('off')

    def show(self):
        self._build()
        plt.show()

    def save_pdf(self, path):
        self._build()
        self.remove_axis()
        plt.savefig(path, format="pdf", bbox_inches='tight', pad_inches=0)

    def _get_element(self, a) -> TextRegion:
        if self.tbrr.is_col(a):
            return self.tbrr.get_col(a)

        return self.doc.get_text_regions()[a]
