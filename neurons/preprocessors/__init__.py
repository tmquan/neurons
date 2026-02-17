"""
Preprocessor classes for handling multiple connectomics data formats.

All preprocessors inherit from BasePreprocessor and implement:
- load(): Load data from file
- validate(): Check if file is valid for this format
- save(): Save data to file
- supported_extensions: Property listing supported file extensions
"""

from neurons.preprocessors.base import BasePreprocessor
from neurons.preprocessors.hdf5 import HDF5Preprocessor
from neurons.preprocessors.tiff import TIFFPreprocessor
from neurons.preprocessors.nrrd import NRRDPreprocessor
from neurons.preprocessors.nfty import NFTYPreprocessor

__all__ = [
    "BasePreprocessor",
    "HDF5Preprocessor",
    "TIFFPreprocessor",
    "NRRDPreprocessor",
    "NFTYPreprocessor",
]
