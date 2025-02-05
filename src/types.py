from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, TypeAlias, TypeVar

# from torch import Tensor
from numpy import ndarray

T = TypeVar("T")
JSONTask: TypeAlias = Dict[str, List[Dict[str, List[List[int]]]]]
Grid = List[List[int]]
GridPairs: TypeAlias = List[Tuple[Grid, Grid]]
AugmentedGrid = List[Grid]
FeatureDict = OrderedDict[str, Tuple[Callable[[Grid], Grid], List[int]]]
NBhood = List[Tuple[int, int]]

# JSONTask_tensor: TypeAlias = Dict[str, List[Dict[str, Tensor]]]
JSONTask_ndarray: TypeAlias = Dict[str, List[Dict[str, ndarray]]]
