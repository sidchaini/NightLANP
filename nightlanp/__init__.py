"""NightLANP: Nightly Light curve Analysis with Neural Processes

A light-curve wrapper around keras-neural-processes (``knp``) for applications
to light curves in time-domain astronomy.

Copyright (C) 2026  Siddharth Chaini
-----
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from .wrappers import LightCurveNeuralProcess
from .data import (
    LightCurvePreprocessor,
    get_context_indices,
    create_evaluation_batch,
    prepare_phys_arrays,
    get_phys_updown_errs,
    mag2flux,
    flux2mag,
    MAG_AB_ZP_NJY,
)
from .metrics import (
    flux_to_luptitudes,
    peak_time_fluxmag_error_1d,
    one_obj_metrics,
    one_objd_metrics_df,
    calculate_msse_weighted,
    chisq,
    compute_metrics_for_row,
    process_model,
)
from .CONSTANTS import (
    translate_filters,
    translate_filternos,
    mycolors,
    seed_val,
    get_ADDFLUX_FOR_MAG_CONST,
)
from . import plotting

__version__ = "0.0.1"

__all__ = [
    # high-level wrapper
    "LightCurveNeuralProcess",
    # preprocessing / light-curve wrangling
    "LightCurvePreprocessor",
    "get_context_indices",
    "create_evaluation_batch",
    "prepare_phys_arrays",
    "get_phys_updown_errs",
    "mag2flux",
    "flux2mag",
    "MAG_AB_ZP_NJY",
    # astrophysical metrics
    "flux_to_luptitudes",
    "peak_time_fluxmag_error_1d",
    "one_obj_metrics",
    "one_objd_metrics_df",
    "calculate_msse_weighted",
    "chisq",
    "compute_metrics_for_row",
    "process_model",
    # constants / helpers
    "translate_filters",
    "translate_filternos",
    "mycolors",
    "seed_val",
    "get_ADDFLUX_FOR_MAG_CONST",
    # submodule
    "plotting",
]
