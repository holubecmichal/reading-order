import glob
import os

from document.page_xml import parse

dir = "/Users/michaelholubec/Desktop/prima"
files = glob.glob(dir + "/*.xml")
ro = 0
short = []

for file in files:
    xml = parse(file)
    regions = xml.get_text_regions()
    key = next(iter(regions))
    box = regions[key].get_box()
    ro = xml.get_reading_order()
    bir = ro.get_before_in_reading()

    if len(bir) >= 5 and len(bir) < 10:
        asd = 0
        short.append(os.path.basename(file).replace('.xml', ''))

asd = 0