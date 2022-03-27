from xml.etree import ElementTree


def element_type(el: ElementTree.Element):
    schema = element_schema(el)

    if schema:
        schema = '{' + element_schema(el) + '}'
        return el.tag.replace(schema, '')

    return el.tag


def element_schema(elem):
    if elem.tag[0] == "{":
        schema, _, _ = elem.tag[1:].partition("}")
    else:
        schema = None
    return schema