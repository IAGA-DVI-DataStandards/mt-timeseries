"""
Numerical prewhitening and Fourier-domain recoloring utilities.
"""

from typing import Mapping

import numpy as np
import xarray as xr
from loguru import logger
from mt_metadata.processing.fourier_coefficients import FCChannel
from mt_metadata.processing.prewhitening import Prewhitening, PrewhiteningTypeEnum


def _fit_ar_residuals(
    channel_data: np.ndarray,
    ar_order: int,
    channel_id: str,
) -> tuple[np.ndarray, FCChannel]:
    """Fit a stable AR model and return its innovation sequence."""
    if channel_data.ndim != 1:
        msg = f"Channel {channel_id} must have one-dimensional time-series data"
        raise ValueError(msg)
    if channel_data.size <= ar_order + 1:
        msg = (
            f"Channel {channel_id} has {channel_data.size} samples, which is too "
            f"short for AR order {ar_order}"
        )
        raise ValueError(msg)
    if not np.isfinite(channel_data).all():
        msg = f"Channel {channel_id} contains missing or non-finite samples"
        raise ValueError(msg)

    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(
        channel_data,
        order=(ar_order, 0, 0),
        trend="n",
        enforce_stationarity=True,
    )
    fitted_model = model.fit()
    if np.any(np.abs(fitted_model.arroots) <= 1):
        msg = f"Channel {channel_id} produced an unstable AR filter"
        raise ValueError(msg)

    ar_coefficients = [float(value) for value in fitted_model.arparams]
    result = FCChannel(
        component=channel_id,
        prewhitening=Prewhitening(
            type=PrewhiteningTypeEnum.arma,
            ar_order=len(ar_coefficients),
            ma_order=0,
            fit_domain="full run",
            response_convention="A(z) = 1 - sum(phi_k z^-k)",
            ar_coefficients=ar_coefficients,
            ma_coefficients=[],
        ),
    )
    return np.asarray(fitted_model.resid), result


def _coefficient_attrs(result: FCChannel) -> dict[str, list[float]]:
    """Return the transient coefficient keys used by the xarray transport."""
    return {
        "ar_coefficients": result.prewhitening.ar_coefficients,
        "ma_coefficients": result.prewhitening.ma_coefficients,
    }


def apply_prewhitening(
    stft_config,
    run_xrds_input: xr.Dataset,
) -> tuple[xr.Dataset, dict[str, FCChannel]]:
    """
    Apply configured prewhitening over the full input run.

    Returns the whitened time series and fitted per-channel filter metadata. The
    metadata must be passed to :func:`apply_recoloring`; filters are never fit
    from Fourier data.
    """
    prewhitening = stft_config.prewhitening
    if not prewhitening.type.value:
        logger.info("No prewhitening specified - skipping this step")
        return run_xrds_input, {}

    if prewhitening.type == PrewhiteningTypeEnum.first_difference:
        return run_xrds_input.diff("time", label="upper"), {}

    if prewhitening.type != PrewhiteningTypeEnum.arma:
        msg = f"{prewhitening.type.value} pre-whitening not implemented"
        logger.error(msg)
        raise NotImplementedError(msg)

    run_xrds = run_xrds_input.copy(deep=True)
    results: dict[str, FCChannel] = {}
    for channel_id, channel in run_xrds_input.data_vars.items():
        residuals, result = _fit_ar_residuals(
            np.asarray(channel.data),
            prewhitening.ar_order,
            channel_id,
        )
        run_xrds[channel_id].data = residuals
        run_xrds[channel_id].attrs["prewhitening"] = _coefficient_attrs(result)
        results[channel_id] = result
    return run_xrds, results


def _whitening_response(
    prewhitening: Prewhitening,
    frequencies: np.ndarray,
    sample_rate: float,
    result: FCChannel | None,
) -> np.ndarray:
    """Compute the exact discrete-time whitening response."""
    z_inverse = np.exp(-2.0j * np.pi * frequencies / sample_rate)
    if prewhitening.type == PrewhiteningTypeEnum.first_difference:
        return 1.0 - z_inverse

    if prewhitening.type == PrewhiteningTypeEnum.arma:
        if result is None:
            msg = "ARMA recoloring requires fitted per-channel coefficients"
            raise ValueError(msg)
        ar_polynomial = np.ones_like(z_inverse, dtype=complex)
        for order, coefficient in enumerate(
            result.prewhitening.ar_coefficients, start=1
        ):
            ar_polynomial -= coefficient * z_inverse**order

        ma_polynomial = np.ones_like(z_inverse, dtype=complex)
        for order, coefficient in enumerate(
            result.prewhitening.ma_coefficients, start=1
        ):
            ma_polynomial += coefficient * z_inverse**order
        return ar_polynomial / ma_polynomial

    msg = f"{prewhitening.type.value} recoloring not implemented"
    raise NotImplementedError(msg)


def apply_recoloring(
    stft_config,
    stft_obj: xr.Dataset,
    sample_rate: float,
    results: Mapping[str, FCChannel],
) -> xr.Dataset:
    """
    Invert configured prewhitening in the Fourier domain channel by channel.

    Frequencies where the whitening response is zero are set to zero, including
    DC for first differences, instead of emitting infinities.
    """
    prewhitening = stft_config.prewhitening
    if not prewhitening.type.value:
        logger.debug("No prewhitening was defined -- skipping recoloring")
        return stft_obj

    stft_obj = stft_obj.copy(deep=True)
    frequencies = np.asarray(stft_obj.frequency.data)
    for channel_id in stft_obj.data_vars:
        response = _whitening_response(
            prewhitening,
            frequencies,
            sample_rate,
            results.get(channel_id),
        )
        valid = ~np.isclose(response, 0.0)
        data = np.asarray(stft_obj[channel_id].data)
        recolored = np.zeros_like(data)
        np.divide(data, response, out=recolored, where=valid)
        stft_obj[channel_id].data = recolored
        if channel_id in results:
            stft_obj[channel_id].attrs["prewhitening"] = _coefficient_attrs(
                results[channel_id]
            )
    return stft_obj
