"""
User Interface components for Monk CLI
Rich terminal interfaces and interactive components.
"""

try:
    from .rich_interface import RichInterface
    __all__ = ['RichInterface']
except ImportError:
    __all__ = []