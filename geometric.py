from document import page_xml
from language_model.carrier import default_model, default_vocab, de_model, de_vocab
from reading_order.metric.dp import compare as dp_compare
from reading_order.metric.prima import compare as prima_compare
from reading_order.processor import Processor
from spatial.plotter import Plotter
from spatial.spare import SpaRe

path = './gt.xml'

xml = page_xml.parse(path)
processor = Processor(default_model(), default_vocab())

spare = SpaRe(xml)
# geometric = spare.get_original_diagonal_reading_order()
# geometric = spare.get_diagonal_reading_order()
geometric = spare.get_columnar_reading_order()
# geometric = spare.get_columnar_lm_reading_order(processor)
# geometric = spare.get_top_to_bottom_reading_order()
# geometric = spare.get_ocr_reading_order()
plotter = Plotter(xml)

plotter.plot_document_border()
plotter.plot_all_regions()
# plotter.plot_columns()
# plotter.annotate_columns()
# plotter.plot_voronoi_columns()
# plotter.plot_voronoi_text_regions()
plotter.annotate_text_regions()
# plotter.plot_reading_order(geometric)
plotter.remove_axis()
# plotter.show()
plotter.save_pdf("/Users/michaelholubec/Desktop/test.pdf")

prima_results = prima_compare(xml, xml.get_reading_order(), geometric)
print('prima penalty: {}'.format(prima_results.penalty()))
print('prima penalty percentage: {}%'.format(prima_results.percentage()))

dp_results = dp_compare(xml.get_reading_order(), geometric)
print('total: {}'.format(dp_results.total()))
print('hits: {}'.format(dp_results.hits()))
print('missed: {}'.format(dp_results.missed()))
print('accuracy: {}%'.format(dp_results.accuracy()))