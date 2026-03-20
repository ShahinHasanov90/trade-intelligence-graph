"""Fraud detection algorithms for trade entity graphs.

This subpackage provides specialized detection algorithms for
identifying common trade fraud patterns in graph structures,
including fraud rings, shell companies, and carousel fraud.
"""

from graph_intel.detection.carousel import CarouselDetector
from graph_intel.detection.rings import FraudRingDetector
from graph_intel.detection.shell_company import ShellCompanyDetector

__all__ = [
    "FraudRingDetector",
    "ShellCompanyDetector",
    "CarouselDetector",
]
