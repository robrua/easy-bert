import pkg_resources

from .bert import Bert


__version__ = pkg_resources.resource_string("easybert", "VERSION.txt").decode("UTF-8").strip()


__all__ = [__version__, Bert]
