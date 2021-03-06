from typing import Optional


class Item(object):
    """
    Rozhrani pro definici prvku reading order
    """

    def get_id(self) -> str: ...

    def set_successor(self, successor: 'Item'): ...

    def get_successor(self) -> Optional['Item']: ...

    def set_predecessor(self, predecessor: 'Item'): ...

    def get_predecessor(self) -> Optional['Item']: ...

    def get_first(self) -> Optional['Item']: ...

    def get_last(self) -> Optional['Item']: ...
