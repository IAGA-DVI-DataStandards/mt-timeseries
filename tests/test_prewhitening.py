import numpy as np
import pytest
import xarray as xr
from mt_metadata.processing import ShortTimeFourierTransform
from mt_metadata.processing.fourier_coefficients import FCChannel
from mt_metadata.processing.prewhitening import Prewhitening, PrewhiteningTypeEnum
from pydantic import ValidationError

from mt_timeseries.spectre.prewhitening import apply_prewhitening, apply_recoloring


@pytest.fixture
def first_difference_config():
    return ShortTimeFourierTransform(
        prewhitening=Prewhitening(type=PrewhiteningTypeEnum.first_difference),
    )


@pytest.fixture
def ar_config():
    return ShortTimeFourierTransform(
        prewhitening=Prewhitening(
            type=PrewhiteningTypeEnum.arma,
            ar_order=1,
            ma_order=0,
        ),
    )


def _time_series_dataset(**channels):
    size = len(next(iter(channels.values())))
    return xr.Dataset(
        {channel_id: ("time", values) for channel_id, values in channels.items()},
        coords={"time": np.arange(size, dtype="timedelta64[s]")},
    )


def _fourier_dataset(channels, frequencies):
    return xr.Dataset(
        {
            channel_id: (("time", "frequency"), values[np.newaxis, :])
            for channel_id, values in channels.items()
        },
        coords={
            "time": np.array([0], dtype="timedelta64[s]"),
            "frequency": frequencies,
        },
    )


def _ar1_samples(coefficient, innovations):
    samples = np.empty_like(innovations)
    samples[0] = innovations[0]
    for index in range(1, len(samples)):
        samples[index] = coefficient * samples[index - 1] + innovations[index]
    return samples


def test_first_difference_uses_discrete_difference(first_difference_config):
    dataset = _time_series_dataset(hx=np.array([2.0, 5.0, 9.0, 14.0]))

    whitened, results = apply_prewhitening(first_difference_config, dataset)

    np.testing.assert_array_equal(whitened.hx.data, np.array([3.0, 4.0, 5.0]))
    assert whitened.time.data[0] == np.timedelta64(1, "s")
    assert results == {}


def test_first_difference_recoloring_uses_exact_response_and_zeroes_dc(
    first_difference_config,
):
    frequencies = np.array([0.0, 0.125, 0.25, 0.375])
    original = np.array([3.0 + 1.0j, 2.0 - 4.0j, -1.0 + 2.0j, 5.0 - 3.0j])
    response = 1.0 - np.exp(-2.0j * np.pi * frequencies)
    whitened = _fourier_dataset({"hx": original * response}, frequencies)

    recolored = apply_recoloring(
        first_difference_config,
        whitened,
        sample_rate=1.0,
        results={},
    )

    assert recolored.hx.data[0, 0] == 0.0
    np.testing.assert_allclose(recolored.hx.data[0, 1:], original[1:])


def test_ar_prewhitening_recovers_coefficients_and_recolors_each_channel(ar_config):
    rng = np.random.default_rng(42)
    hx = _ar1_samples(0.7, rng.normal(size=512))
    hy = _ar1_samples(-0.4, rng.normal(size=512))
    dataset = _time_series_dataset(hx=hx, hy=hy)

    whitened, results = apply_prewhitening(ar_config, dataset)

    assert abs(np.corrcoef(whitened.hx.data[:-1], whitened.hx.data[1:])[0, 1]) < abs(
        np.corrcoef(hx[:-1], hx[1:])[0, 1]
    )
    assert abs(np.corrcoef(whitened.hy.data[:-1], whitened.hy.data[1:])[0, 1]) < abs(
        np.corrcoef(hy[:-1], hy[1:])[0, 1]
    )
    assert isinstance(results["hx"], FCChannel)
    assert results["hx"].prewhitening.ar_coefficients[0] == pytest.approx(0.7, abs=0.1)
    assert results["hy"].prewhitening.ar_coefficients[0] == pytest.approx(-0.4, abs=0.1)
    assert results["hx"].prewhitening.ma_coefficients == []
    assert results["hy"].prewhitening.ma_coefficients == []

    frequencies = np.array([0.1, 0.2, 0.3])
    original = {
        "hx": np.array([2.0 + 3.0j, -1.0 + 4.0j, 5.0 - 2.0j]),
        "hy": np.array([-3.0 + 2.0j, 4.0 + 1.0j, -2.0 - 5.0j]),
    }
    whitened_fourier = {
        channel_id: values
        * (
            1.0
            - results[channel_id].prewhitening.ar_coefficients[0]
            * np.exp(-2.0j * np.pi * frequencies)
        )
        for channel_id, values in original.items()
    }

    recolored = apply_recoloring(
        ar_config,
        _fourier_dataset(whitened_fourier, frequencies),
        sample_rate=1.0,
        results=results,
    )

    for channel_id, values in original.items():
        np.testing.assert_allclose(recolored[channel_id].data[0], values)
        assert recolored[channel_id].attrs["prewhitening"] == {
            "ar_coefficients": results[channel_id].prewhitening.ar_coefficients,
            "ma_coefficients": results[channel_id].prewhitening.ma_coefficients,
        }


def test_ar_prewhitening_rejects_non_finite_samples(ar_config):
    values = np.arange(16, dtype=float)
    values[3] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        apply_prewhitening(
            ar_config,
            _time_series_dataset(hx=values),
        )


def test_ar_prewhitening_rejects_nonzero_ma_order():
    with pytest.raises(NotImplementedError, match="ma_order"):
        Prewhitening(
            type=PrewhiteningTypeEnum.arma,
            ar_order=1,
            ma_order=1,
        )


def test_prewhitening_metadata_validates_arma_order():
    with pytest.raises(ValueError, match="ar_order"):
        Prewhitening(type=PrewhiteningTypeEnum.arma)

    with pytest.raises(ValidationError, match="ar_order"):
        Prewhitening(ar_order=-1)
