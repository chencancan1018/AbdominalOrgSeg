import os
from .AGdataset import AGSegDataset
from .AIVCdataset import AIVCSegDataset
from .Coarsedataset import CoarseSegDataset
from .Esodataset import EsoSegDataset
from .Galldataset import GallSegDataset
from .Lagdataset import LagSegDataset
from .LGdataset import LGSegDataset
from .LSKdataset import LSKSegDataset
from .PDStodataset import PDStoSegDataset
from .Ragdataset import RagSegDataset

def dataset_match(chars):
    if chars == "AGSegDataset":
        return AGSegDataset
    elif chars == "AIVCSegDataset":
        return AIVCSegDataset
    elif chars == "CoarseSegDataset":
        return CoarseSegDataset
    elif chars == "EsoSegDataset":
        return EsoSegDataset
    elif chars == "GallSegDataset":
        return GallSegDataset
    elif chars == "LagSegDataset":
        return LagSegDataset
    elif chars == "LGSegDataset":
        return LGSegDataset
    elif chars == "LSKSegDataset":
        return LSKSegDataset
    elif chars == "PDStoSegDataset":
        return PDStoSegDataset
    elif chars == "RagSegDataset":
        return RagSegDataset
    else:
        raise KeyError("wrong {} match!".format(chars))
