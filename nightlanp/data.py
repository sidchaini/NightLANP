import sys
import numpy as np
import numpy.typing as npt
import joblib
import pandas as pd
import gc
import polars as pl
from tqdm.auto import tqdm
import tensorflow as tf
from keras import ops
from sklearn.preprocessing import MinMaxScaler
from .CONSTANTS import seed_val
from knp.data import get_ragged_tensor, unscale_values


class LightCurvePreprocessor:
    def __init__(
        self,
        time_scaler=None,
        flux_scaler=None,
        fltnum_scaler=None,
        FLUX_MAX=1000,
        FLTNUM_MAX=5,
    ):
        # using sklearn scalers for management but polars for speed
        self.time_scaler = time_scaler if time_scaler else MinMaxScaler()
        self.fltnum_scaler = (
            fltnum_scaler
            if fltnum_scaler
            else MinMaxScaler(feature_range=(0, FLTNUM_MAX))
        )
        self.flux_scaler = (
            flux_scaler if flux_scaler else MinMaxScaler(feature_range=(0, FLUX_MAX))
        )
        self.is_fitted = False

    def fit(self, lc_lazy):
        if self.is_fitted:
            print("WARNING: Overwriting previous scaler fits", file=sys.stderr)

        if isinstance(lc_lazy, pl.DataFrame):
            lc_lazy = lc_lazy.lazy()

        stats = lc_lazy.select(
            [
                pl.col("mjd").list.min().min().alias("mjd_min"),
                pl.col("mjd").list.max().max().alias("mjd_max"),
                pl.col("fltnum").list.min().min().alias("fltnum_min"),
                pl.col("fltnum").list.max().max().alias("fltnum_max"),
                pl.col("flux").list.min().min().alias("flux_min"),
                pl.col("flux").list.max().max().alias("flux_max"),
            ]
        ).collect()

        self.time_scaler.fit([[stats["mjd_min"][0]], [stats["mjd_max"][0]]])
        self.fltnum_scaler.fit([[stats["fltnum_min"][0]], [stats["fltnum_max"][0]]])
        self.flux_scaler.fit([[stats["flux_min"][0]], [stats["flux_max"][0]]])
        self.is_fitted = True

    def transform(self, phot_df):
        def scale_polars(col_name, scaler):
            if isinstance(scaler, MinMaxScaler):
                data_min = scaler.data_min_[0]
                data_max = scaler.data_max_[0]
                feature_min, feature_max = scaler.feature_range

                scale = (feature_max - feature_min) / (data_max - data_min)
                min_val = feature_min - data_min * scale

                return (
                    pl.col(col_name).cast(pl.List(pl.Float32)) * scale + min_val
                ).cast(pl.List(pl.Float32))

            else:
                raise NotImplementedError("Only MinMaxScaler supported currently")

        return phot_df.with_columns(
            [
                scale_polars("mjd", self.time_scaler),
                scale_polars("fltnum", self.fltnum_scaler).cast(pl.List(pl.UInt8)),
                scale_polars("flux", self.flux_scaler),
            ]
        )

    def inverse_transform(self, phot_df):
        # for invert transform, take flux and flux err, calculate upper and lower bounds,
        # then invert transform these bounds rather than the error
        # then, take bounds, and get error as half of distance between them.
        # doing this as an approximation to transform error
        def inverse_scale_polars(col_name, scaler):
            if isinstance(scaler, MinMaxScaler):
                data_min = scaler.data_min_[0]
                data_max = scaler.data_max_[0]
                feature_min, feature_max = scaler.feature_range

                scale = (feature_max - feature_min) / (data_max - data_min)
                min_val = feature_min - data_min * scale

                return (
                    (pl.col(col_name).cast(pl.List(pl.Float32)) - min_val) / scale
                ).cast(pl.List(pl.Float32))

            else:
                raise NotImplementedError("Only MinMaxScaler supported currently")

        # make pl expression for inverse transform
        inverted_expr = [
            inverse_scale_polars("mjd", self.time_scaler),
            inverse_scale_polars("fltnum", self.fltnum_scaler).cast(pl.List(pl.UInt8)),
            inverse_scale_polars("flux", self.flux_scaler),
        ]

        if "flux_err" in phot_df.columns:
            # get upper and lower bounds
            phot_df = phot_df.select(
                pl.all(),
                (pl.col("flux") + pl.col("flux_err")).alias("flux_upper"),
                (pl.col("flux") - pl.col("flux_err")).alias("flux_lower"),
            ).drop("flux_err")

            # add pl expression for inverse-transforming the bounds
            inverted_expr = inverted_expr + [
                inverse_scale_polars("flux_upper", self.flux_scaler).cast(
                    pl.List(pl.Float32)
                ),
                inverse_scale_polars("flux_lower", self.flux_scaler).cast(
                    pl.List(pl.Float32)
                ),
            ]

        # apply pl expressions to light curves
        phot_df = phot_df.with_columns(inverted_expr)

        # calculate standalone error from bounds after transformation
        if "flux_upper" in phot_df.columns and "flux_lower" in phot_df.columns:
            phot_df = phot_df.select(
                pl.all().exclude(["flux_upper", "flux_lower"]),
                (
                    (pl.col("flux_upper") - pl.col("flux_lower"))
                    / pl.lit(2, dtype=pl.Float32)
                ).alias("flux_err"),
                # removed .cast(pl.List(pl.Float32)) at the end
            )
        return phot_df

    def process(self, lc_lazy):

        df = lc_lazy.select(
            [pl.col("mjd"), pl.col("fltnum").cast(pl.List(pl.Float32)), pl.col("flux")]
        ).collect()

        mjd_arrow = df.drop_in_place("mjd").to_arrow()
        flt_arrow = df.drop_in_place("fltnum").to_arrow()
        splits = mjd_arrow.offsets

        X_ragged = np.stack(
            [
                mjd_arrow.values.to_numpy(),
                flt_arrow.values.to_numpy(),
            ],
            axis=1,
        )
        del (mjd_arrow, flt_arrow)

        X_ragged = tf.RaggedTensor.from_row_splits(X_ragged, splits)

        y_ragged = df.drop_in_place("flux").to_arrow().values.to_numpy().reshape(-1, 1)
        del df
        y_ragged = tf.RaggedTensor.from_row_splits(y_ragged, splits)

        return X_ragged, y_ragged

    def get_lightcurves(self, X, y, objids, yerr=None, inverse_transform=True):
        if not self.is_fitted:
            raise ValueError("Processor not fitted! Cannot unscale.")

        X = get_ragged_tensor(X) if not isinstance(X, tf.RaggedTensor) else X
        y = get_ragged_tensor(y) if not isinstance(y, tf.RaggedTensor) else y
        if yerr is not None and not isinstance(yerr, tf.RaggedTensor):
            yerr = get_ragged_tensor(yerr)

        row_lengths = X.row_lengths().numpy()
        if len(objids) != len(row_lengths):
            raise ValueError(
                f"Shape mismatch: {len(objids)} objids vs {len(row_lengths)} rows"
            )

        mjd = X[..., 0].flat_values.numpy()
        fltnum = X[..., 1].flat_values.numpy()
        del X
        flux = y[..., 0].flat_values.numpy()
        del y
        flux_err = yerr[..., 0].flat_values.numpy() if yerr is not None else None
        del yerr
        gc.collect()

        phot_df = {
            "objid": pl.Series(np.repeat(objids, row_lengths)).cast(pl.UInt32),
            "mjd": pl.Series(mjd).cast(pl.Float32),
            "fltnum": pl.Series(fltnum).cast(pl.UInt8),
            "flux": pl.Series(flux).cast(pl.Float32),
        }
        if flux_err is not None:
            phot_df["flux_err"] = pl.Series(flux_err).cast(pl.Float32)

        phot_df = (
            pl.DataFrame(phot_df)
            .group_by("objid", maintain_order=True)
            .agg(pl.exclude("objid"))
        )

        if inverse_transform:
            phot_df = self.inverse_transform(phot_df)

        return phot_df

    def save(self, path):
        joblib.dump(
            {
                "time": self.time_scaler,
                "fltnum": self.fltnum_scaler,
                "flux": self.flux_scaler,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path):
        data = joblib.load(path)
        self.time_scaler = data["time"]
        self.fltnum_scaler = data["fltnum"]
        self.flux_scaler = data["flux"]
        self.is_fitted = data["is_fitted"]


def prepare_phys_arrays(x, y, time_scaler, flux_scaler):
    # prepare physical unit arrays
    x_phys = (np.array(x) if not hasattr(x, "numpy") else x.numpy()).copy()
    y_phys = (np.array(y) if not hasattr(y, "numpy") else y.numpy()).copy()

    x_phys[:, 0] = unscale_values(x_phys[:, 0], time_scaler)
    y_phys[:, 0] = unscale_values(y_phys[:, 0], flux_scaler)
    return x_phys, y_phys


def get_phys_updown_errs(y, y_err):
    # get bounds from errors
    y = (np.array(y) if not hasattr(y, "numpy") else y.numpy()).copy()
    y_err = (np.array(y_err) if not hasattr(y_err, "numpy") else y_err.numpy()).copy()

    y_phys_down_bound = (y[:, 0] - y_err[:, 0]).reshape(y.shape)
    y_phys_up_bound = (y[:, 0] + y_err[:, 0]).reshape(y.shape)

    return y_phys_down_bound, y_phys_up_bound


# old code; phase out soon
def get_context_indices(
    full_mjd,
    peak_mjd,
    num_points,
    strategy="random",
    prepeak_maskcutoff=50,  # 100,
    postpeak_maskcutoff=75,  # 150
    seed=seed_val,
):
    rng = np.random.default_rng(seed)
    total_points = len(full_mjd)
    indices = np.arange(total_points)

    if total_points == 0:
        return np.zeros(num_points, dtype=int)

    chosen_idx = []

    if strategy == "random":
        chosen_idx = rng.choice(
            indices, size=min(num_points, total_points), replace=False
        )

    elif strategy == "peak_focused":
        peak_mask = (full_mjd >= peak_mjd - 2) & (full_mjd <= peak_mjd + 2)
        peak_indices = indices[peak_mask]
        if len(peak_indices) > 0:
            p_idx = rng.choice(peak_indices, 1)
            chosen_idx.append(p_idx[0])

        remaining = num_points - len(chosen_idx)
        if remaining > 0:
            broad_mask = (full_mjd >= peak_mjd - prepeak_maskcutoff) & (
                full_mjd <= peak_mjd + postpeak_maskcutoff
            )
            broad_indices = indices[broad_mask]

            broad_indices = np.setdiff1d(broad_indices, chosen_idx)

            if len(broad_indices) > 0:
                count = min(remaining, len(broad_indices))
                chosen_idx.extend(rng.choice(broad_indices, count, replace=False))

    elif strategy == "extrapolation":
        pre_peak_mask = full_mjd < peak_mjd
        pre_peak_indices = indices[pre_peak_mask]

        count = min(num_points, len(pre_peak_indices))
        if count > 0:
            chosen_idx.extend(rng.choice(pre_peak_indices, count, replace=False))

    chosen_idx = np.array(chosen_idx, dtype=int)

    if len(chosen_idx) < num_points:
        num_needed = num_points - len(chosen_idx)
        available_indices = np.setdiff1d(indices, chosen_idx)
        if len(available_indices) > 0:
            count = min(num_needed, len(available_indices))
            fillers = rng.choice(available_indices, count, replace=False)
            chosen_idx = np.concatenate([chosen_idx, fillers])

    if len(chosen_idx) < num_points:
        num_needed = num_points - len(chosen_idx)
        padding = rng.choice(indices, size=num_needed, replace=True)
        chosen_idx = np.concatenate([chosen_idx, padding])

    return np.sort(chosen_idx).astype(int)


def create_evaluation_batch(
    X_ragged,
    y_ragged,
    lc_lazy,
    num_points,
    strategy,
    time_scaler,
    seed=seed_val,
    pbar=True,
):
    context_x_list, context_y_list = [], []

    peak_rels_lazy = (
        lc_lazy.select(pl.col("objid"), pl.col("mjd").list.min().alias("mjd_min"))
        .sort("objid")
        .collect()
        .iter_rows()
    )

    iterator = enumerate(peak_rels_lazy)
    iterator = (
        tqdm(
            iterator, total=X_ragged.shape[0], leave=False, desc=f"{strategy} class no."
        )
        if pbar
        else iterator
    )

    splits = X_ragged.row_splits.numpy()
    assert (splits == y_ragged.row_splits.numpy()).all()

    for i, (snid, peak_rel) in iterator:
        x_norm = X_ragged._values[splits[i] : splits[i + 1]].numpy()
        y_norm = y_ragged._values[splits[i] : splits[i + 1]].numpy()
        mjd_rel = unscale_values(x_norm[:, 0], time_scaler).flatten()
        # assert np.isclose(mjd_rel, lc_train_og[i].select("mjd").item().to_numpy()).all()

        indices = get_context_indices(mjd_rel, peak_rel, num_points, strategy, seed)

        context_x_list.append(x_norm[indices])
        context_y_list.append(y_norm[indices])

    return np.array(context_x_list, dtype=np.float32), np.array(
        context_y_list, dtype=np.float32
    )


# Code taken verbatim from tdastro: https://github.com/lincc-frameworks/tdastro/
# Thanks to the LINCC Frameworks team!!!

# AB definition is zp=8.9 for 1 Jy
MAG_AB_ZP_NJY = 8.9 + 2.5 * 9


def mag2flux(mag: npt.ArrayLike) -> npt.ArrayLike:
    """Convert AB magnitude to bandflux in nJy

    Parameters
    ----------
    mag : ndarray of float
        The magnitude to convert to bandflux.

    Returns
    -------
    bandflux : ndarray of float
        The bandflux corresponding to the input magnitude.
    """
    return np.power(10.0, -0.4 * (mag - MAG_AB_ZP_NJY))


def flux2mag(flux: npt.ArrayLike) -> npt.ArrayLike:
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = -2.5 * np.log10(flux) + MAG_AB_ZP_NJY
    # mag[flux <= 0] = np.nan
    return mag

##### CODE BELOW IS FOR OPSIM

def poisson_bandflux_std(
    bandflux: npt.ArrayLike,
    *,
    total_exposure_time: npt.ArrayLike,
    exposure_count: npt.ArrayLike,
    footprint: npt.ArrayLike,
    sky: npt.ArrayLike,
    zp: npt.ArrayLike,
    readout_noise: npt.ArrayLike,
    dark_current: npt.ArrayLike,
) -> npt.ArrayLike:
    """Simulate photon noise for bandflux measurements.

    Parameters
    ----------
    bandflux : array_like of float
        Source bandflux in energy units, e.g. nJy.
    total_exposure_time : array_like of float
        Total exposure time of all observation, in time units
        (e.g. seconds).
    exposure_count : array_like of int
        Number of exposures in the observation.
    sky : array_like of float
        Sky background per unit angular area,
        in the units of electrons / pixel^2.
    footprint : array_like of float
        Point spread function effective area, in pixel^2.
    zp : array_like of float
        Zero point bandflux for the observation, i.e. bandflux
        giving a single electron during the total exposure time.
        Units are the same as the input bandflux over electron,
        e.g. nJy / electron.
    readout_noise : array_like of float
        Standard deviation of the readout electrons per pixel per exposure.
    dark_current : array_like of float
        Mean dark current electrons per pixel per unit time.

    Returns
    -------
    array_like
        Simulated bandflux noise, in the same units as the input bandflux.

    Notes
    -----

    1. We do not specify units for the input parameters, but they
    should be consistent with each other.

    2. Here we assume that the sky and source photon noises follow
    Poisson statistics in the limit of large number of photons,
    e.g. they are both considered to be normal distributed with
    variance equal to the number of photons. Readout noise is
    assumed to be Poisson distributed with variance (squared mean)
    equal to the square of the given value. Dark current is assumed
    to be Poisson distributed with variance (squared mean) equal
    to the product of the given value and the exposure time.
    The output is Poisson standard deviation of the sum of all
    these noises converted to the flux units.
    """
    # Get variances, in electrons^2
    source_variance = bandflux / zp
    sky_variance = sky * footprint
    readout_variance = readout_noise**2 * footprint * exposure_count
    dark_variance = dark_current * total_exposure_time * footprint

    total_variance = source_variance + sky_variance + readout_variance + dark_variance

    return np.sqrt(total_variance) * zp

_lsstcam_extinction_coeff = {
    "u": -0.458,
    "g": -0.208,
    "r": -0.122,
    "i": -0.074,
    "z": -0.057,
    "y": -0.095,
}

_lsstcam_zeropoint_per_sec_zenith = {
    "u": 26.524,
    "g": 28.508,
    "r": 28.361,
    "i": 28.171,
    "z": 27.782,
    "y": 26.818,
}

def magnitude_electron_zeropoint(  # noqa: D103
    *,
    band: npt.ArrayLike,
    airmass: npt.ArrayLike,
    exptime: npt.ArrayLike,
    instr_zp: dict[str, float] | None,
    ext_coeff: dict[str, float] | None,
) -> npt.ArrayLike:
    instr_zp = _lsstcam_zeropoint_per_sec_zenith if instr_zp is None else instr_zp
    ext_coeff = _lsstcam_extinction_coeff if ext_coeff is None else ext_coeff

    instr_zp_getter = np.vectorize(instr_zp.get)
    ext_coeff_getter = np.vectorize(ext_coeff.get)

    return instr_zp_getter(band) + ext_coeff_getter(band) * (airmass - 1) + 2.5 * np.log10(exptime)


def flux_electron_zeropoint(  # noqa: D103
    *,
    instr_zp_mag: dict[str, float] | None,
    ext_coeff: dict[str, float] | None,
    band: npt.ArrayLike,
    airmass: npt.ArrayLike,
    exptime: npt.ArrayLike,
) -> npt.ArrayLike:
    mag_zp_electron = magnitude_electron_zeropoint(
        instr_zp=instr_zp_mag, ext_coeff=ext_coeff, band=band, airmass=airmass, exptime=exptime
    )
    return mag2flux(mag_zp_electron)

GAUSS_EFF_AREA2FWHM_SQ = np.pi / (2 * np.log(2))  # ~2.266


import sqlite3
from pathlib import Path

import astropy.units as u
from astropy.coordinates import Latitude, Longitude
from scipy.spatial import KDTree

class ObsTable:
    """A wrapper class around the observations table with helper computation functions and
    cached data for efficiency.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the survey information. Metadata can be included in the
        "tdastro_survey_data" entry of the attributes dictionary.
    colmap : dict, optional
        A mapping of standard column names to their names in the input table.
        For example, in Rubin's OpSim we might have the column "observationStartMJD"
        which maps to "time". In that case we would have an entry with key="time"
        and value="observationStartMJD".
    **kwargs : dict
        Additional keyword arguments to pass to the constructor. This can include
        overrides of any of the survey values.

    Attributes
    ----------
    survey_values : dict, optional
        A mapping for constant values for the survey used in various computations, such
        as readout noise and dark current.
    _table : pandas.core.frame.DataFrame
        The table with all the observation information mapped to standard column names.
    _colmap : dict
        A mapping of standard column names to their names in the input table.
    _inv_colmap : dict
        A dictionary mapping the custom column names back to the standard names.
    _kd_tree : scipy.spatial.KDTree or None
        A kd_tree of the survey pointings for fast spatial queries. We use the scipy
        kd-tree instead of astropy's functions so we can directly control caching.
    """

    _required_columns = ["ra", "dec", "time"]

    # Default survey values. These are all None for the abstract base class.
    _default_survey_values = {
        "dark_current": None,
        "ext_coeff": None,
        "pixel_scale": None,
        "radius": None,
        "read_noise": None,
        "zp_per_sec": None,
        "survey_name": "Unknown",
    }

    def __init__(
        self,
        table,
        *,
        colmap=None,
        **kwargs,
    ):
        # Create a copy of the table.
        if isinstance(table, dict):
            self._table = pd.DataFrame(table)
        else:
            self._table = table.copy()

        # Remap the columns to standard names. Start with the existing names (from the table)
        # and overwrite anything provided by the column map. Save the inverse mapping.
        name_map = {col: col for col in self._table.columns}
        self._inv_colmap = {}
        self._colmap = colmap if colmap is not None else {}
        if colmap is not None:
            for key, value in colmap.items():
                if value in name_map:
                    # Check for collisions (mapping a column to an existing column)
                    if key in self._table.columns and key != value:
                        raise ValueError(f"Trying to map {value} to {key}, but {key} is already a column.")

                    # Add this entry to the list of column names that need to be remapped.
                    name_map[value] = key

                # Save the inverse mapping as well
                self._inv_colmap[value] = key
        self._table.rename(columns=name_map, inplace=True)

        # Check that we have the required columns.
        for col in self._required_columns:
            if col not in self._table.columns:
                raise KeyError(f"Missing required column: {col}")

        # Save the survey values, overwriting anything that is manually specified
        # as a keyword argument or provided in the table's metadata.
        self.survey_values = self._default_survey_values.copy()
        if "tdastro_survey_data" in self._table.attrs:
            metadata = self._table.attrs["tdastro_survey_data"]
            if not isinstance(metadata, dict):
                raise TypeError("Got unexpected type for tdastro_survey_data")
            for key, value in metadata.items():
                self.survey_values[key] = value
        for key, value in kwargs.items():
            self.survey_values[key] = value

        # If we are not given zero point data, try to derive it from the other columns.
        if "zp" not in self:
            self._assign_zero_points()

        # Build the kd-tree.
        self._kd_tree = None
        self._build_kd_tree()

    def __len__(self):
        return len(self._table)

    def __getitem__(self, key):
        """Access the underlying observation table by column name."""
        if key in self._table.columns:
            return self._table[key]
        if key in self._inv_colmap and self._inv_colmap[key] in self._table.columns:
            return self._table[self._inv_colmap[key]]
        raise KeyError(f"Column not found: {key}")

    def __contains__(self, key):
        """Check if a column exists in the survey table."""
        if key in self._table.columns:
            return True
        if key in self._inv_colmap and self._inv_colmap[key] in self._table.columns:
            return True
        return False

    def safe_get_survey_value(self, key):
        """Get a survey value by key, checking that it is not None.

        Parameters
        ----------
        key : str
            The key of the survey value to retrieve.
        """
        value = self.survey_values.get(key, None)
        if value is None:
            raise ValueError(
                f"Survey value for {key} is not defined. This should be set when creating the object."
            )
        return value

    @property
    def columns(self):
        """Get the column names."""
        return self._table.columns

    @classmethod
    def from_db(cls, filename, sql_query="SELECT * FROM observations", **kwargs):
        """Create an ObsTable object from the data in an db file. Reads data matching
        what is produced by write_db (and matching the RubinOpsim table).

        Parameters
        ----------
        filename : str
            The name of the db file.
        sql_query : str
            The SQL query to use when loading the table.
            Default: "SELECT * FROM observations"
        kwargs : dict, optional
            Additional keyword arguments to pass to the Survey constructor.

        Returns
        -------
        ObsTable
            A table with all of the pointing data.

        Raise
        -----
        FileNotFoundError if the file does not exist.
        ValueError if unable to load the table.
        """
        if not Path(filename).is_file():
            raise FileNotFoundError(f"db file {filename} not found.")
        con = sqlite3.connect(f"file:{filename}?mode=ro", uri=True)

        # Read the table.
        try:
            survey_data = pd.read_sql_query(sql_query, con)
        except Exception:
            raise ValueError("Database read failed.") from None

        # Close the connection.
        con.close()

        return cls(survey_data, **kwargs)

    @classmethod
    def from_parquet(cls, filename):
        """Create an ObsTable object from a parquet file.

        Parameters
        ----------
        filename : str
            The name of the parquet file to read.

        Returns
        -------
        ObsTable
            A table with all of the pointing data.
        """
        if not Path(filename).is_file():
            raise FileNotFoundError(f"File {filename} not found.")
        survey_data = pd.read_parquet(filename)
        return cls(survey_data)

    def get_filters(self):
        """Get the unique filters in the ObsTable."""
        if "filter" not in self._table.columns:
            raise KeyError("No filters column found in ObsTable.")
        return np.unique(self._table["filter"])


    def _build_kd_tree(self):
        """Construct the KD-tree from the ObsTable."""
        ra_rad = np.radians(self._table["ra"].to_numpy())
        dec_rad = np.radians(self._table["dec"].to_numpy())
        # Convert the pointings to Cartesian coordinates on a unit sphere.
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        cart_coords = np.array([x, y, z]).T

        # Construct the kd-tree.
        self._kd_tree = KDTree(cart_coords)

    def _assign_zero_points(self):
        """Assign instrumental zero points in nJy to the data table.

        Default implementation does not produce a zeropoint column. Subclasses
        should override this method with a survey specific computation.
        """
        pass

    def add_column(self, colname, values, *, overwrite=False):
        """Add a column to the current data table.

        Parameters
        ----------
        colname : str
            The name of the new column.
        values : int, float, str, list, or numpy.ndarray
            The value(s) to add.
        overwrite : bool
            Overwrite the column is it already exists.
            Default: False
        """
        if colname in self._table.columns and not overwrite:
            raise KeyError(f"Column {colname} already exists.")

        # If the input is a scalar, turn it into an array of the correct length
        if np.isscalar(values):
            values = np.full((len(self._table)), values)
        self._table[colname] = values

    def write_db(self, filename, *, tablename="observations", overwrite=False):
        """Write out the observation table as a database to a given SQL table.

        Parameters
        ----------
        filename : str
            The name of the db file.
        tablename : str
            The table to which to write.
            Default: "observations"
        overwrite : bool
            Overwrite the existing DB file.
            Default: False

        Raise
        -----
        FileExistsError if the file already exists and overwrite is False.
        """
        if_exists = "replace" if overwrite else "fail"

        con = sqlite3.connect(filename)
        try:
            self._table.to_sql(tablename, con, if_exists=if_exists)
        except Exception:
            raise ValueError("Database write failed.") from None

        con.close()

    def write_parquet(self, filename, *, overwrite=False):
        """Write out the observation table as a parquet file.

        Parameters
        ----------
        filename : str
            The name of the parquet file.
        overwrite : bool
            Overwrite the existing parquet file.
            Default: False

        Raise
        -----
        FileExistsError if the file already exists and overwrite is False.
        """
        if not overwrite and Path(filename).is_file():
            raise FileExistsError(f"File {filename} already exists.")

        # Save all the survey data as metadata.
        self._table.attrs["tdastro_survey_data"] = self.survey_values
        self._table.to_parquet(filename)

    def time_bounds(self):
        """Returns the min and max times for all observations in the ObsTable.

        Returns
        -------
        t_min, t_max : float, float
            The min and max times for all observations in the ObsTable.
        """
        t_min = self._table["time"].min()
        t_max = self._table["time"].max()
        return t_min, t_max

    def filter_rows(self, rows):
        """Filter the rows in the ObsTable to only include those indices that are provided
        in a list of row indices (integers) or marked True in a mask.

        Parameters
        ----------
        rows : numpy.ndarray
            Either a Boolean array of the same length as the table or list of integer
            row indices to keep.

        Returns
        -------
        self : ObsTable
            The filtered ObsTable object.
        """
        # Check if we are dealing with a mask of a list of indices.
        rows = np.asarray(rows)
        if rows.dtype == bool:
            if len(rows) != len(self._table):
                raise ValueError(
                    f"Mask length mismatch. Expected {len(self._table)} rows, but found {len(rows)}."
                )
            mask = rows
        else:
            mask = np.full((len(self._table),), False)
            mask[rows] = True

        # Filter the rows in-place and build a new kd-tree.
        self._table = self._table[mask]
        self._kd_tree = None
        self._build_kd_tree()

        return self

    def is_observed(self, query_ra, query_dec, radius=None, t_min=None, t_max=None):
        """Check if the query point(s) fall within the field of view of any
        pointing in the ObsTable.

        Parameters
        ----------
        query_ra : float or numpy.ndarray
            The query right ascension (in degrees).
        query_dec : float or numpy.ndarray
            The query declination (in degrees).
        radius : float or None, optional
            The angular radius of the observation (in degrees).
        t_min : float or None, optional
            The minimum time (in MJD) for the observations to consider.
            If None, no time filtering is applied.
        t_max : float or None, optional
            The maximum time (in MJD) for the observations to consider.
            If None, no time filtering is applied.

        Returns
        -------
        seen : bool or list[bool]
            Depending on the input, this is either a single bool to indicate
            whether the query point is observed or a list of bools for an array
            of query points.
        """
        inds = self.range_search(query_ra, query_dec, radius, t_min=t_min, t_max=t_max)
        if np.isscalar(query_ra):
            return len(inds) > 0
        return [len(entry) > 0 for entry in inds]

    def range_search(self, query_ra, query_dec, radius=None, t_min=None, t_max=None):
        """Return the indices of the pointings that fall within the field
        of view of the query point(s).

        Parameters
        ----------
        query_ra : float or numpy.ndarray
            The query right ascension (in degrees).
        query_dec : float or numpy.ndarray
            The query declination (in degrees).
        radius : float or None, optional
            The angular radius of the observation (in degrees). If None
            uses the default radius for the ObsTable.
        t_min : float, numpy.ndarray or None, optional
            The minimum time (in MJD) for the observations to consider.
            If None, no time filtering is applied.
        t_max : float, numpy.ndarray or None, optional
            The maximum time (in MJD) for the observations to consider.
            If None, no time filtering is applied.

        Returns
        -------
        inds : list[int] or list[numpy.ndarray]
            Depending on the input, this is either a list of indices for a single query point
            or a list of arrays (of indices) for an array of query points.
        """
        if query_ra is None or query_dec is None:
            raise ValueError("Query RA and dec must be provided for range search, but got None.")

        # Fallback to the preset radius if None is provided.
        radius = radius if radius is not None else self.survey_values.get("radius", None)
        if radius is None:
            raise ValueError("Radius must be provided for range search or as a default. Got None.")

        # If the points are scalars, make them into length 1 arrays.
        is_scalar = np.isscalar(query_ra) and np.isscalar(query_dec)
        query_ra = np.atleast_1d(query_ra)
        query_dec = np.atleast_1d(query_dec)

        # Confirm the query RA and Dec have the same length.
        if len(query_ra) != len(query_dec):
            raise ValueError("Query RA and Dec must have the same length.")
        if np.any(query_ra == None) or np.any(query_dec == None):  # noqa: E711
            raise ValueError("Query RA and dec cannot contain None.")

        # Transform the query point(s) to 3-d Cartesian coordinate(s).
        ra_rad = np.radians(query_ra)
        dec_rad = np.radians(query_dec)
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        cart_query = np.array([x, y, z]).T

        # Adjust the angular radius to a cartesian search radius and perform the search.
        adjusted_radius = 2.0 * np.sin(0.5 * np.radians(radius))
        inds = self._kd_tree.query_ball_point(cart_query, adjusted_radius)

        if t_min is not None or t_max is not None:
            num_queries = len(query_ra)
            times = self._table["time"].to_numpy()

            if t_min is None:
                t_min = np.full(num_queries, -np.inf)
            else:
                t_min = np.atleast_1d(t_min)
            if len(t_min) != num_queries:
                raise ValueError(f"t_min must be a scalar or an array of length {num_queries}.")

            if t_max is None:
                t_max = np.full(num_queries, np.inf)
            else:
                t_max = np.atleast_1d(t_max)
            if len(t_max) != num_queries:
                raise ValueError(f"t_max must be a scalar or an array of length {num_queries}.")

            # Run through each list of indices and filter by time. We need to do this
            # iteratively, because the lists can have different lengths.
            for idx, subinds in enumerate(inds):
                if len(subinds) == 0:
                    continue
                time_mask = (times[subinds] >= t_min[idx]) & (times[subinds] <= t_max[idx])
                inds[idx] = np.asarray(subinds)[time_mask]

        # If the query was a scalar, we return a single list of indices.
        if is_scalar:
            inds = inds[0]
        return inds

    def get_observations(self, query_ra, query_dec, radius=None, t_min=None, t_max=None, cols=None):
        """Return the observation information when the query point falls within
        the field of view of a pointing in the ObsTable.

        Parameters
        ----------
        query_ra : float
            The query right ascension (in degrees).
        query_dec : float
            The query declination (in degrees).
        radius : float or None, optional
            The angular radius of the observation (in degrees). If None
            uses the default radius for the ObsTable.
        t_min : float or None, optional
            The minimum time (in MJD) for the observations to consider.
            If None, no time filtering is applied.
        t_max : float or None, optional
            The maximum time (in MJD) for the observations to consider.
            If None, no time filtering is applied.
        cols : list or str
            A list of the names of columns to extract or a single column name.
            If None returns all the columns.

        Returns
        -------
        results : dict
            A dictionary mapping the given column name to a numpy array of values.
        """
        neighbors = self.range_search(query_ra, query_dec, radius, t_min=t_min, t_max=t_max)

        results = {}
        if cols is None:
            cols = self._table.columns.to_list()
        elif isinstance(cols, str):
            cols = [cols]

        for col in cols:
            # Allow the user to specify either the original or mapped column names,
            # by using the class accessor (__getitem__), instead of the table one.
            if col not in self:
                raise KeyError(f"Unrecognized column name {col}")
            results[col] = self[col][neighbors].to_numpy()
        return results

    def bandflux_error_point_source(self, bandflux, index):
        """Compute observational bandflux error for a point source.

        Parameters
        ----------
        bandflux : array_like of float
            Band bandflux of the point source in nJy.
        index : array_like of int
            The index of the observation in the ObsTable table.

        Returns
        -------
        flux_err : array_like of float
            Simulated bandflux noise in nJy.
        """
        raise NotImplementedError

import logging

import pooch

logger = logging.getLogger(__name__)

def download_data_file_if_needed(data_path, data_url, force_download=False):
    """Download a data file from a URL and save it to a specified path.

    Parameters
    ----------
    data_path : str or Path
        The path to the data file. This is where the downloaded file will be written.
    data_url : str
        The URL to download the data file.
    force_download : bool, optional
        If True, the file will be downloaded even if it already exists. Default is False.

    Returns
    -------
    bool
        True if the download was successful, False otherwise.
    """
    # Start by checking if the file already exists and if we are not forcing a download.
    data_path = Path(data_path)
    if not force_download and data_path.exists():
        logger.info(f"Data file {data_path} already exists. Skipping download.")
        return True

    # Check that there is a valid URL for the download.
    if data_url is None or len(data_url) == 0:
        raise ValueError("No URL given for table download.")
    logger.info(f"Downloading data file from {data_url} to {data_path}")

    # Create the directory in which to save the file if it does not already exist.
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Use pooch to download the data files and extract them to the data directory.
    full_path = pooch.retrieve(
        url=data_url,
        known_hash=None,
        fname=data_path.name,
        path=data_path.parent,
    )

    if full_path is None or not Path(full_path).exists():
        logger.error(f"Transmission table not downloaded from {data_url}.")
        return False
    return True


LSSTCAM_PIXEL_SCALE = 0.2
"""The pixel scale for the LSST camera in arcseconds per pixel."""

_lsstcam_readout_noise = 8.8
"""The standard deviation of the count of readout electrons per pixel for the LSST camera.

The value is from https://smtn-002.lsst.io/v/OPSIM-1171/index.html
"""

_lsstcam_dark_current = 0.2
"""The dark current for the LSST camera in electrons per second per pixel.

The value is from https://smtn-002.lsst.io/v/OPSIM-1171/index.html
"""

_lsstcam_view_radius = 1.75
"""The angular radius of the observation field (in degrees)."""


class OpSim(ObsTable):
    """A wrapper class around the opsim table with cached data for efficiency.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the OpSim information.
    colmap : dict
        A mapping of short column names to their names in the underlying table.
        Defaults to the Rubin OpSim column names, stored in the class variable
        _opsim_colnames.
    **kwargs : dict
        Additional keyword arguments to pass to the constructor. This includes overrides
        for survey parameters such as:
        - dark_current : The dark current for the camera in electrons per second per pixel.
        - ext_coeff: Mapping of filter names to extinction coefficients.
        - pixel_scale: The pixel scale for the camera in arcseconds per pixel.
        - radius: The angular radius of the observations (in degrees).
        - read_noise: The readout noise for the camera in electrons per pixel.
        - zp_per_sec: Mapping of filter names to zeropoints at zenith.
    """

    _required_names = ["ra", "dec", "time"]

    # Default column names for the Rubin OpSim.
    _default_colnames = {
        "airmass": "airmass",
        "dec": "fieldDec",
        "exptime": "visitExposureTime",
        "filter": "filter",
        "ra": "fieldRA",
        "time": "observationStartMJD",
        "zp": "zp_nJy",  # We add this column to the table
        "seeing": "seeingFwhmEff",
        "skybrightness": "skyBrightness",
        "nexposure": "numExposures",
    }

    # Default survey values.
    _default_survey_values = {
        "dark_current": _lsstcam_dark_current,
        "ext_coeff": _lsstcam_extinction_coeff,
        "pixel_scale": LSSTCAM_PIXEL_SCALE,
        "radius": _lsstcam_view_radius,
        "read_noise": _lsstcam_readout_noise,
        "zp_per_sec": _lsstcam_zeropoint_per_sec_zenith,
        "survey_name": "LSST",
    }

    # Class constants for the column names.
    def __init__(
        self,
        table,
        colmap=None,
        **kwargs,
    ):
        colmap = self._default_colnames if colmap is None else colmap
        super().__init__(table, colmap=colmap, **kwargs)

    def _assign_zero_points(self):
        """Assign instrumental zero points in nJy to the OpSim tables."""
        cols = self._table.columns.to_list()
        if not ("filter" in cols and "airmass" in cols and "exptime" in cols):
            raise ValueError(
                "OpSim does not include the columns needed to derive zero point "
                "information. Required columns: filter, airmass, and exptime."
            )

        zp_values = flux_electron_zeropoint(
            ext_coeff=self.safe_get_survey_value("ext_coeff"),
            instr_zp_mag=self.safe_get_survey_value("zp_per_sec"),
            band=self._table["filter"],
            airmass=self._table["airmass"],
            exptime=self._table["exptime"],
        )
        self.add_column("zp", zp_values, overwrite=True)

    @classmethod
    def from_url(cls, opsim_url, force_download=False):
        """Construct an OpSim object from a URL to a predefined opsim data file.

        For Rubin OpSim data, you will typically use the latest baseline data set in:
        https://s3df.slac.stanford.edu/data/rubin/sim-data/
        such as:
        https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs3.4/baseline/baseline_v3.4_10yrs.db

        Parameters
        ----------
        opsim_url : str
            The URL to the opsim data file.
        force_download : bool, optional
            If True, the OpSim data will be downloaded even if it already exists locally.
            Default is False.

        Returns
        -------
        opsim : OpSim
            An OpSim object containing the data from the specified URL.
        """
        data_file_name = opsim_url.split("/")[-1]
        data_path = _TDASTRO_BASE_DATA_DIR / "opsim" / data_file_name

        if not download_data_file_if_needed(data_path, opsim_url, force_download=force_download):
            raise RuntimeError(f"Failed to download opsim data from {opsim_url}.")
        return cls.from_db(data_path)

    def bandflux_error_point_source(self, bandflux, index):
        """Compute observational bandflux error for a point source

        Parameters
        ----------
        bandflux : array_like of float
            Band bandflux of the point source in nJy.
        index : array_like of int
            The index of the observation in the OpSim table.

        Returns
        -------
        flux_err : array_like of float
            Simulated bandflux noise in nJy.
        """
        observations = self._table.iloc[index]

        # By the effective FWHM definition, see
        # https://smtn-002.lsst.io/v/OPSIM-1171/index.html
        # We need it in pixel^2
        pixel_scale = self.safe_get_survey_value("pixel_scale")
        footprint = GAUSS_EFF_AREA2FWHM_SQ * (observations["seeing"] / pixel_scale) ** 2
        zp = observations["zp"]

        # Table value is in mag/arcsec^2
        sky_njy_angular = mag2flux(observations["skybrightness"])
        # We need electrons per pixel^2
        sky = sky_njy_angular * pixel_scale**2 / zp

        return poisson_bandflux_std(
            bandflux,
            total_exposure_time=observations["exptime"],
            exposure_count=observations["nexposure"],
            footprint=footprint,
            sky=sky,
            zp=zp,
            readout_noise=self.safe_get_survey_value("read_noise"),
            dark_current=self.safe_get_survey_value("dark_current"),
        )


def create_random_opsim(num_obs, seed=None):
    """Create a random OpSim pointings drawn uniformly from (RA, dec).

    Parameters
    ----------
    num_obs : int
        The size of the OpSim to generate.
    seed : int
        The seed to used for random number generation. If None then
        uses a default random number generator.
        Default: None

    Returns
    -------
    opsim_data : OpSim
        The OpSim data structure.
    seed : int, optional
        The seed for the random number generator.
    """
    if num_obs <= 0:
        raise ValueError("Number of observations must be greater than zero.")

    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed=seed)

    # Generate the (RA, dec) pairs uniformly on the surface of a sphere.
    ra = np.degrees(rng.uniform(0.0, 2.0 * np.pi, size=num_obs))
    dec = np.degrees(np.arccos(2.0 * rng.uniform(0.0, 1.0, size=num_obs) - 1.0) - (np.pi / 2.0))

    # Generate the information needed to compute zeropoint.
    airmass = rng.uniform(1.3, 1.7, size=num_obs)
    filter = rng.choice(["u", "g", "r", "i", "z", "y"], size=num_obs)

    input_data = {
        "observationStartMJD": 0.05 * np.arange(num_obs),
        "fieldRA": ra,
        "fieldDec": dec,
        "airmass": airmass,
        "filter": filter,
        "visitExposureTime": 29.0 * np.ones(num_obs),
    }

    opsim = OpSim(input_data)
    return opsim


def opsim_add_random_data(opsim_data, colname, min_val=0.0, max_val=1.0):
    """Add a column composed of random uniform data. Used for testing.

    Parameters
    ----------
    opsim_data : OpSim
        The OpSim data structure to modify.
    colname : str
        The name of the new column to add.
    min_val : float
        The minimum value of the uniform range.
        Default: 0.0
    max_val : float
        The maximum value of the uniform range.
        Default: 1.0
    """
    values = np.random.uniform(low=min_val, high=max_val, size=len(opsim_data))
    opsim_data.add_column(colname, values)


def oversample_opsim(
    opsim: OpSim,
    *,
    pointing: tuple[float, float] = (200, -50),
    search_radius: float = 1.75,
    delta_t: float = 0.01,
    time_range: tuple[float | None, float | None] = (None, None),
    bands: list[str] | None = None,
    strategy: str = "darkest_sky",
):
    """Single-pointing oversampled OpSim table.

    It includes observations for a single pointing only,
    but with very high time resolution. The observations
    would alternate between the bands.

    Parameters
    ----------
    opsim : OpSim
        The OpSim table to oversample.
    pointing : tuple of RA and Dec in degrees
        The pointing to use for the oversampled table.
    search_radius : float, optional
        The search radius for the oversampled table in degrees.
        The default is the half of the LSST's field of view.
    delta_t : float, optional
        The time between observations in days.
    time_range : tuple or floats or Nones, optional
        The start and end times of the observations in MJD.
        None means to use the minimum (maximum) time in
        all the observations found for the given pointing.
        Time is being samples as np.arange(*time_range, delta_t).
    bands : list of str or None, optional
        The list of bands to include in the oversampled table.
        The default is to include all bands found for the given pointing.
    strategy : str, optional
        The strategy to select prototype observations.
        - "darkest_sky" selects the observations with the minimal sky brightness
          (maximum "skyBrightness" value) in each band. This is the default.
        - "random" selects the observations randomly. Fixed seed is used.

    """
    ra, dec = pointing
    observations = opsim._table.iloc[opsim.range_search(ra, dec, search_radius)]
    if len(observations) == 0:
        raise ValueError("No observations found for the given pointing.")

    time_min, time_max = time_range
    if time_min is None:
        time_min = np.min(observations["time"])
    if time_max is None:
        time_max = np.max(observations["time"])
    if time_min >= time_max:
        raise ValueError(f"Invalid time_range: start > end: {time_min} > {time_max}")

    uniq_bands = np.unique(observations["filter"])
    if bands is None:
        bands = uniq_bands
    elif not set(bands).issubset(uniq_bands):
        raise ValueError(f"Invalid bands: {bands}")

    new_times = np.arange(time_min, time_max, delta_t)
    n = len(new_times)
    if n < len(bands):
        raise ValueError("Not enough time points to cover all bands.")

    new_table = pd.DataFrame(
        {
            # Just in case, to not have confusion with the original table
            "observationId": opsim._table["observationId"].max() + 1 + np.arange(n),
            "time": new_times,
            "ra": ra,
            "dec": dec,
            "filter": np.tile(bands, n // len(bands)),
        }
    )
    other_columns = [column for column in observations.columns if column not in new_table.columns]

    if strategy == "darkest_sky":
        for band in bands:
            # MAXimum magnitude is MINimum brightness (darkest sky)
            idxmax = observations["skybrightness"][observations["filter"] == band].idxmax()
            idx = new_table.index[new_table["filter"] == band]
            darkest_sky_obs = pd.DataFrame.from_records([observations.loc[idxmax]] * idx.size, index=idx)
            new_table.loc[idx, other_columns] = darkest_sky_obs[other_columns]
    elif strategy == "random":
        rng = np.random.default_rng(0)
        for band in bands:
            single_band_obs = observations[observations["filter"] == band]
            idx = new_table.index[new_table["filter"] == band]
            random_obs = single_band_obs.sample(idx.size, replace=True, random_state=rng).set_index(idx)
            new_table.loc[idx, other_columns] = random_obs[other_columns]
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    return OpSim(
        new_table,
        colmap=opsim._colmap,
        **opsim.survey_values,
    )

