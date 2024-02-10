from enum import Enum, auto


class QueryType(Enum):
    Multi_Query = auto()
    Contextual_Compression = auto()
    Parent_Document = auto()
    Simple_Query = auto()
    Ensembles = auto()
