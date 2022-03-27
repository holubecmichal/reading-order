import numpy as np
from matplotlib import pyplot as plt

from document.stubs import Document, TextRegion
from reading_order.reading_order import ReadingOrder
from spatial.spare import get_voronoi_polygons, cutout_by_box, get_density, SpaRe


def plot_density(regions: {TextRegion}):
    density, intervals = get_density(regions)

    plt.scatter(list(density.values()), np.zeros(len(density)))
    for interval in intervals:
        plt.axvline(interval)

    plt.show()


class Plotter(object):
    def __init__(self, document: Document):
        self.document = document
        self.spare = SpaRe(self.document)
        self.ax = plt.subplot()

    def plot_document_border(self):
        self.ax.plot(*self.document.get_box().exterior.xy)
        return self

    def plot_all_regions(self):
        regions = self.document.get_regions()

        for r in self.document.get_regions():
            region = regions[r]
            self.ax.plot(*region.get_box().exterior.xy, region.get_color())

        return self

    def annotate_text_regions(self):
        regions = self.document.get_text_regions()

        for r in regions:
            region = regions[r]
            self.ax.annotate(r, (region.get_box().centroid.x, region.get_box().centroid.y), color=region.get_color(),
                             fontsize=8, ha='center', va='center')

        return self

    def plot_voronoi_text_regions(self):
        polygons = get_voronoi_polygons(self.document.get_text_regions())

        for i in cutout_by_box(polygons, self.document.get_box()):
            plt.plot(*polygons[i].exterior.xy, 'r')

        return self

    def plot_voronoi_columns(self):
        polygons = get_voronoi_polygons(self.spare.get_columns() | self.spare.get_regions_not_in_columns())

        for i in cutout_by_box(polygons, self.document.get_box()):
            plt.plot(*polygons[i].exterior.xy, 'r')

        return self

    def plot_columns(self):
        cols = self.spare.get_columns()
        for c in cols:
            plt.plot(*cols[c].get_box().exterior.xy, cols[c].get_color())

    def annotate_columns(self):
        cols = self.spare.get_columns()
        for c in cols:
            col = cols[c]
            self.ax.annotate(c, (col.get_box().centroid.x, col.get_box().centroid.y), color=col.get_color(),
                             fontsize=8, ha='center', va='center')

    def plot_reading_order(self, reading_order: ReadingOrder):
        for group in reading_order.get_ordered_groups():
            a = group.get_beginnings()[0]
            b = a.get_successor()

            while b:
                a_reg = self.spare.get_region_by_id(a.get_id())
                b_reg = self.spare.get_region_by_id(b.get_id())
                self.ax.annotate("", xy=(b_reg.centroid().x, b_reg.centroid().y), xytext=(a_reg.centroid().x, a_reg.centroid().y),
                                 arrowprops=dict(arrowstyle="->", color='r'))

                a = b
                b = b.get_successor()

    def _build(self):
        plt.axis('equal')
        plt.gca().invert_yaxis()

    def remove_axis(self):
        plt.axis('off')

    def show(self):
        self._build()
        plt.show()

    def save_svg(self, path):
        self._build()
        self.remove_axis()
        plt.savefig(path, format="svg", bbox_inches='tight', pad_inches=0)

    def save_pdf(self, path):
        self._build()
        self.remove_axis()
        plt.savefig(path, format="pdf", bbox_inches='tight', pad_inches=0)

    def save_jpg(self, path):
        self._build()
        self.remove_axis()
        plt.savefig(path, format="jpg", bbox_inches='tight', pad_inches=0)
