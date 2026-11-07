"""Science layers: rock size-frequency fit (C7) and science-value raster (C9).

Offline and input-agnostic. Real inputs (CRISM mineralogy, the Jezero geologic
map, a trained rock detector) arrive from P2/P1 later; these modules combine
whatever rasters/measurements they produce. No fabricated science values.
"""

from marswalk.science.size_frequency import (
    SFDFit,
    cumulative_number_density,
    expected_count_at_least,
    fit_exponential_sfd,
)
from marswalk.science.value_raster import science_value, top_targets

__all__ = [
    "SFDFit",
    "fit_exponential_sfd",
    "cumulative_number_density",
    "expected_count_at_least",
    "science_value",
    "top_targets",
]
