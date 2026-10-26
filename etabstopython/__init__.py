"""ETABS extraction utilities."""

__all__ = ["EtabsModel"]


def __getattr__(name):
    if name == "EtabsModel":
        from etabstopython.model import EtabsModel

        return EtabsModel
    raise AttributeError(name)
