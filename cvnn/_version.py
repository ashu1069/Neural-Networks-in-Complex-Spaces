"""Package version helpers."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("neural-networks-in-complex-spaces")
except PackageNotFoundError:
    __version__ = "0.0.0+local"
