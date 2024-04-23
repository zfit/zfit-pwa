from __future__ import annotations

import types

import numpy as np
import zfit  # suppress tf warnings
import zfit.z.numpy as znp
from zfit import supports, z

from zfit_pwa.compwa.variables import obs_from_frame


def patched_call(self, data) -> np.ndarray:
    extended_data = {**self.__parameters, **data}  # type: ignore[arg-type]
    return self.__function(extended_data)  # type: ignore[arg-type]


class ComPWAPDF(zfit.pdf.BasePDF):
    def __init__(
        self, intensity, norm, obs=None, params=None, extended=None, name="ComPWA"
    ):
        """ComPWA intensity normalized over the *norm* dataset."""
        if params is None:
            params = {
                name: zfit.param.convert_to_parameter(
                    val, name=name, prefer_constant=False
                )
                for name, val in intensity.parameters.items()
            }
        if obs is None:
            obs = obs_from_frame(norm)
        intensity.__call__ = types.MethodType(patched_call, intensity)
        super().__init__(obs, params=params, name=name, extended=extended)
        self.intensity = intensity
        norm = {ob: znp.array(ar) for ob, ar in zip(self.obs, z.unstack_x(norm))}
        self.norm_sample = norm

    @supports(norm=True)
    def _pdf(self, x, norm):
        data = {ob: znp.array(ar) for ob, ar in zip(self.obs, z.unstack_x(x))}
        params = {p.name: znp.array(p) for p in self.params.values()}
        data |= params

        unnormalized_pdf = self._jitted_unnormalized_pdf(data)

        if norm is False:
            return unnormalized_pdf
        else:
            norm_sample = self.norm_sample | params
            return unnormalized_pdf / self._jitted_normalization(norm_sample)

    @z.function(wraps="tensorwaves")
    def _jitted_unnormalized_pdf(self, data):
        unnormalized_pdf = self.intensity(data)

        return unnormalized_pdf

    @z.function(wraps="tensorwaves")
    def _jitted_normalization(self, norm_sample):
        return znp.mean(self._jitted_unnormalized_pdf(norm_sample))
