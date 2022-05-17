from typing import Optional
from xml.etree import ElementTree

from document.stubs import TextRegion as StubTextRegion
from reading_order.stubs import Item as StubItem
from utils.xml import element_type

INDENT = 4
UNORDERED_GROUP = 'UnorderedGroup'
UNORDERED_GROUP_INDEXED = 'UnorderedGroupIndexed'
ORDERED_GROUP = 'OrderedGroup'
ORDERED_GROUP_INDEXED = 'OrderedGroupIndexed'
REGION_REF = 'RegionRef'
REGION_REF_INDEXED = 'RegionRefIndexed'

ORDERED_GROUPS = [ORDERED_GROUP, ORDERED_GROUP_INDEXED]
UNORDERED_GROUPS = [UNORDERED_GROUP, UNORDERED_GROUP_INDEXED]
ITEMS = [REGION_REF, REGION_REF_INDEXED]


class UnknownElement(Exception):
    ...


class Item(StubItem):
    """
    Polozka reading orderu
    """

    generated_id = 0

    def __init__(self, item_type, id, index=None, text_region: StubTextRegion=None):
        if id is None:
            id = str(Item.generated_id)
            Item.generated_id += 1

        self.id = id
        self.type = item_type
        self.index = index
        self.text_region = text_region
        self.predecessor = None
        self.successor = None
        self.parent = None

    def get_text(self) -> str:
        """
        Metoda vrati text z polozek posloupnosti
        """

        if self.text_region:
            # postupne konkatenuje text od prvniho prvku
            item = self.get_first()
            text = []

            while item:
                # pres jeho nasledniky
                text.append(item.get_text_region().get_text())
                item = item.successor

            # az k poslednimu prvku
            return ' '.join(text)

        # pro netextove regiony, napr skupiny (group) vraci prazdny string
        return ''

    def get_text_region(self) -> Optional[StubTextRegion]:
        """
        vraci originalni textovy region
        """
        return self.text_region

    def set_parent(self, parent: 'Group'):
        self.parent = parent

    def get_parent(self) -> Optional['Group']:
        return self.parent

    def get_id(self) -> str:
        return self.id

    def _has_index(self):
        return self.index is not None

    def _get_index(self):
        return self.index

    def set_successor(self, successor):
        self.successor = successor

    def set_predecessor(self, predecessor):
        self.predecessor = predecessor

    def get_successor(self) -> Optional['Item']:
        return self.successor

    def get_predecessor(self) -> Optional['Item']:
        return self.predecessor

    def get_first(self):
        """
        Vraci prvni prvek posloupnosti
        """

        if self.predecessor is None:
            return self

        return self.predecessor.get_first()

    def get_last(self):
        """
        Vraci posledni prvek posloupnosti
        """

        if self.successor is None:
            return self

        return self.successor.get_last()

    def print(self, indent: int):
        """
        Pomocna metoda pro tisk struktury reading order
        """

        if self._has_index():
            print(' ' * indent + '(' + str(self.index + 1) + ') ' + self.get_id())
        else:
            print(' ' * indent + self.get_id())

    def get_level(self):
        """
        Vraci hodnotu zanorezni
        """

        level = 0
        parent = self.get_parent()

        while parent:
            level += 1
            parent = parent.get_parent()

        return level

    def get_path(self) -> ['Group']:
        """
        Vraci cestu od korenu stromu k dane instanci
        """

        path = [self]
        parent = self.get_parent()

        while parent:
            path.append(parent)
            parent = parent.get_parent()

        path.reverse()
        return path


class Group(Item):
    """
    Trida, ktera predstavuje skupinu.
    Dedi Item, aby vyuzila funkce predchudce a naslednika (i skupina muze byt v jine [serazene] skupine)
    """

    def __init__(self, group_type, id=None, index=None):
        super().__init__(group_type, id, index)
        self.items = {}

    def _get_symbol(self) -> str:
        """
        Pomocna metoda pro tisk, rozliseni usporadane a neusporadane posloupnosti
        """

        if self.type in UNORDERED_GROUPS:
            return '*'
        else:
            return '='

    def add(self, item: Item):
        """
        Pridani prvku do skupiny
        """

        item.set_parent(self)
        self.items[item.get_id()] = item

    def add_candidates(self, source: StubTextRegion, successor: StubTextRegion):
        """
        Pridani inicializacni sekvece a naslednika do skupiny
        """

        if isinstance(source, StubItem):
            # pokud uz se source sklada z vice prvku (linearni cesta), pak nactu posledni prvek
            source = source.get_last()

        if isinstance(successor, StubItem):
            # pokud uz se successor sklada z vice prvku (linearni cesta), pak nactu prvni prvek
            # to uz by melo byt implicitne, ale pro jistotu jej nactu
            successor = successor.get_first()

        # pokud se jeste nejedna o prvek zarazeny v posloupnosti, pak jej pridam
        if source.get_id() not in self.items:
            self.add(Item(REGION_REF, source.get_id(), text_region=source))

        if successor.get_id() not in self.items:
            self.add(Item(REGION_REF, successor.get_id(), text_region=successor))

        source_item = self.items[source.get_id()]
        successor_item = self.items[successor.get_id()]

        # spojim v linearni sezanm
        source_item.set_successor(successor_item)
        successor_item.set_predecessor(source_item)

    def add_ordered_group(self):
        """
        Prida novou usporadanou skupinu
        """

        group = Group(ORDERED_GROUP)
        self.add(group)

        return group

    def get_beginnings(self):
        """
        Vraci prvky, ktere predstavuji pocatek serazenych skupin
        """
        return [self.items[i] for i in self.items if self.items[i].get_predecessor() is None]

    def print(self, indent: int):
        """
        Pomocna metoda pro tisk
        """

        to_print = self._get_symbol() + ' ' + self.get_id() + ' ' + self._get_symbol()

        if self._has_index():
            print(' ' * indent + '(' + str(self._get_index() + 1) + ') ' + to_print)
        else:
            print(' ' * indent + to_print)

        indent = indent + INDENT

        if self.type in UNORDERED_GROUPS:
            for i in self.items:
                item = self.items[i]
                item.print(indent)
        else:
            beginnings = self.get_beginnings()
            for item in beginnings:
                while item:
                    item.print(indent)
                    item = item.get_successor()

    def is_ordered(self) -> bool:
        return self.type in ORDERED_GROUPS

    def is_unordered(self) -> bool:
        return not self.is_ordered()

    def has_item(self, id) -> bool:
        """
        Zkontroluje, zda se ve skupine nachazi nejaky prvek
        """

        flatten = flatten_items(self)
        ids = [item.get_id() for item in flatten]

        return id in ids

    def get_only_groups(self):
        """
        Vraci vsechny skupiny v teto skupine, pouze skupiny
        """

        return {self.items[i].get_id(): self.items[i] for i in self.items if isinstance(self.items[i], Group)}


class ReadingOrder:
    """
    Koren stromu posloupnosti cteni
    """


    def __init__(self, root: Group = None):
        if root:
            self.root = root
        else:
            self.root = Group(UNORDERED_GROUP)

    def get_all_items(self) -> [Item]:
        """
        Vraci vsechny prvky posloupnosti, pouze prvky
        """
        return flatten_items(self.root)

    def get_by_id(self, id) -> Optional[Item]:
        """
        Vraci prvek dle id
        """

        flatten = self.get_all_items()
        items = {item.get_id(): item for item in flatten}

        return items[id] if id in items else None

    def add_ordered_group(self):
        """
        Pride skupinu pro usporadanou posloupnost
        """
        return self.root.add_ordered_group()

    def get_ordered_groups(self) -> [Group]:
        """
        Vrati vsechny skupiny predstavujici usporadanou posloupnost
        """

        groups = [group for group in flatten_groups(self.root) if group.type in ORDERED_GROUPS]

        if self.root.type in ORDERED_GROUPS:
            groups.append(self.root)

        return groups

    def get_chain_reduction(self):
        """
        Vraci strukturu Chain reduction, dvojice usporadanych prvku
        """

        ordered = self.get_ordered_groups()
        before_in_reading = []

        for group in ordered:
            a = group.get_beginnings()[0]
            b = a.get_successor()

            while b:
                before_in_reading.append((a.get_id(), b.get_id()))
                a = b
                b = b.get_successor()

        return before_in_reading

    def print(self):
        self.root.print(0)


def from_page_xml_element(reading_order_element: ElementTree.Element) -> ReadingOrder:
    """
    Funkce pro sestaveni Reading order instance z PageXML souboru
    """

    element = reading_order_element.find('./')

    if element_type(element) == UNORDERED_GROUP:
        group = process_unordered_group(element)
    else:
        group = process_ordered_group(element)

    return ReadingOrder(group)


def process_unordered_group(element) -> Group:
    """
    Pomocna funkce pro zpracovani neusporadane skupiny PageXML
    """

    el_type = element_type(element)

    if el_type not in UNORDERED_GROUPS:
        raise UnknownElement(el_type)

    # vytovri novou neuspradanou skupinu, instanci
    unordered_group = Group(el_type, element.get('id'), element_index(element))

    for el in element.findall('./'):
        # zpracovani prvku a pridani do posloupnosti
        item = process_element(el)
        unordered_group.add(item)

    return unordered_group


def process_ordered_group(element) -> Group:
    """
    Pomocna funkce pro zpracovani usporadane skupiny PageXML
    """

    el_type = element_type(element)

    if el_type not in ORDERED_GROUPS:
        raise UnknownElement(el_type)

    # vytovri novou uspradanou skupinu, instanci
    ordered_group = Group(el_type, element.get('id'), element_index(element))

    predecessor = None

    for el in element.findall('./'):
        # zpracovani prvku a pridani do skupiny
        item = process_element(el)
        ordered_group.add(item)

        if predecessor:
            # pokud ma prvek v posloupnosti predchudce, je to zohledneno i v programu
            item.set_predecessor(predecessor)
            predecessor.set_successor(item)

        predecessor = item

    return ordered_group


def process_element(element) -> Item:
    """
    Pomocna funkce pro zpracovani PageXML prvku
    """
    el_type = element_type(element)

    if el_type in UNORDERED_GROUPS:
        return process_unordered_group(element)
    elif el_type in ORDERED_GROUPS:
        return process_ordered_group(element)
    elif el_type in ITEMS:
        return Item(el_type, element.get('regionRef'), element_index(element))
    else:
        raise UnknownElement(el_type)


def element_index(el: ElementTree.Element) -> Optional[int]:
    """
    Pomocna funkce pro nacteni id regionu z PageXML
    """
    return int(el.get('index')) if el.get('index') else None


def flatten_items(group: Group) -> [Item]:
    """
    Rekurzivne projde vsechny skupiny a sestavi seznam prvku, nachazejici se v techto skupinach,
    Vrati vsechny prvky v group, rekurzivne
    """

    items = []

    for i in group.items:
        item = group.items[i]

        if isinstance(item, Group):
            items = items + flatten_items(item)
        else:
            items.append(item)

    return items


def flatten_groups(group: Group) -> [Group]:
    """
    Rekurzivne projde vsechny skupiny a sestavi seznam skupin, nachazejici se v techto skupinach,
    Vrati vsechny skupiny v group, rekurzivne
    """

    groups = []

    for i in group.items:
        item = group.items[i]

        if isinstance(item, Group):
            groups.append(item)
            groups = groups + flatten_groups(item)

    return groups


def ancestor_prefix_path(path1: [Item], path2: [Item]) -> [Group]:
    """
    Funkce vrati spolecnou cestu stromu dvou prvku
    """

    prefix = []

    while True:
        x = path1.pop(0)
        y = path2.pop(0)

        if x.get_id() != y.get_id():
            break

        prefix.append(x)

    return prefix


def get_nearest_ancestor(item1: Item, item2: Item) -> Group:
    """
    Funkce vrati prvniho spolecneho predka
    """

    prefix = ancestor_prefix_path(item1.get_path(), item2.get_path())
    return prefix[-1]


def get_group_which_has_item(groups, id) -> Optional[Group]:
    """
    Funkce vrati skupinu, ve ktere se prvek dle id nachazi
    """

    for group in groups:
        if group.has_item(id):
            return group

    return None


def get_ids(items: [Item]):
    """
    Pomocna funkce pro nacte id prvku v seznamu
    """

    return [item.get_id() for item in items]
