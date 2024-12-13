from importlib import metadata

from .localai_embeddings import LocalAIEmbeddings

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "LocalAIEmbeddings",
    "__version__",
]
