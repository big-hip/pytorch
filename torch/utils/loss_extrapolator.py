# mypy: allow-untyped-defs
r"""Extrapolate the final training loss from a few early observations.

Following the empirical neural scaling laws of Kaplan et al., 2020
(https://arxiv.org/abs/2001.08361), the training loss as a function of the
optimization step ``t`` is well described by a power law with an irreducible
offset::

    L(t) = L_inf + (t_c / t) ** alpha

where ``L_inf`` is the irreducible loss the run converges to, ``t_c`` is a
characteristic step and ``alpha`` is the scaling exponent. Fitting these three
parameters on the first few ``(step, loss)`` pairs gives a cheap estimate of the
loss the run will reach after many more steps, which is handy to print alongside
the live loss during training.
"""

import math
from typing import Optional

import torch


__all__ = ["LossExtrapolator"]


class LossExtrapolator:
    r"""Fit a neural-scaling-law power law to ``(step, loss)`` observations.

    The loss is modelled as ``L(t) = L_inf + coefficient * t ** -exponent``.
    For a fixed ``L_inf`` the remaining parameters are linear in log-log space,
    so the fit is a closed-form least-squares line; ``L_inf`` itself is found
    with a golden-section search over ``[0, min(loss))``. The procedure is
    deterministic and requires no gradient-based optimizer.

    Typical usage during a training loop::

        extrapolator = LossExtrapolator()
        for step, loss in enumerate(training_steps):
            ...
            extrapolator.update(step + 1, loss.item())
            final = extrapolator.estimate_final_loss(total_steps)
            if final is not None:
                print(f"step {step + 1} loss {loss.item():.4f} "
                      f"estimated final loss {final:.4f}")

    Args:
        irreducible_loss (float, optional): if given, ``L_inf`` is fixed to this
            value instead of being fit. Useful when the asymptotic loss (for
            example an entropy floor) is known in advance. Default: ``None``.
        min_observations (int): minimum number of recorded observations before a
            fit is attempted. Power-law fitting with an unknown irreducible loss
            needs at least 3 points; a few more makes the estimate far less
            noisy. Default: ``4``.
    """

    def __init__(
        self,
        irreducible_loss: Optional[float] = None,
        min_observations: int = 4,
    ) -> None:
        if irreducible_loss is not None and irreducible_loss < 0:
            raise ValueError("irreducible_loss must be non-negative")
        min_required = 2 if irreducible_loss is not None else 3
        if min_observations < min_required:
            raise ValueError(
                f"min_observations must be at least {min_required} "
                f"when irreducible_loss is "
                f"{'set' if irreducible_loss is not None else 'unknown'}"
            )
        self._fixed_irreducible_loss = irreducible_loss
        self.min_observations = min_observations
        self.steps: list[float] = []
        self.losses: list[float] = []
        self.irreducible_loss: Optional[float] = None
        self.coefficient: Optional[float] = None
        self.exponent: Optional[float] = None
        self._fitted = False

    def update(self, step, loss) -> None:
        """Record a single ``(step, loss)`` observation.

        Args:
            step: positive optimization step (1-indexed). Accepts a Python
                number or a scalar :class:`torch.Tensor`.
            loss: loss value at ``step``; must be strictly positive, as required
                by the power-law model. Accepts a Python number or a scalar
                :class:`torch.Tensor`.
        """
        step = float(step.item() if isinstance(step, torch.Tensor) else step)
        loss = float(loss.item() if isinstance(loss, torch.Tensor) else loss)
        if not math.isfinite(step) or step <= 0:
            raise ValueError(f"step must be a positive finite number, got {step}")
        # The power law L_inf + coefficient * t ** -exponent with L_inf >= 0 and
        # coefficient > 0 is strictly positive, so a non-positive loss cannot be
        # fit and would otherwise corrupt the log-space search.
        if not math.isfinite(loss) or loss <= 0:
            raise ValueError(f"loss must be a positive finite number, got {loss}")
        self.steps.append(step)
        self.losses.append(loss)
        self._fitted = False

    def reset(self) -> None:
        """Forget all observations and the current fit."""
        self.steps.clear()
        self.losses.clear()
        self.irreducible_loss = None
        self.coefficient = None
        self.exponent = None
        self._fitted = False

    def _can_fit(self) -> bool:
        if len(self.steps) < self.min_observations:
            return False
        # A non-degenerate log-log line needs at least two distinct steps.
        return len(set(self.steps)) >= 2

    def _line_fit(self, irreducible_loss: float):
        # Least-squares line log(loss - L_inf) = intercept - exponent * log(step).
        # Returns (exponent, log_coefficient, residual_sum_of_squares).
        x = torch.log(torch.tensor(self.steps, dtype=torch.float64))
        y = torch.log(torch.tensor(self.losses, dtype=torch.float64) - irreducible_loss)
        mx = x.mean()
        my = y.mean()
        dx = x - mx
        slope = (dx * (y - my)).sum() / (dx * dx).sum()
        intercept = my - slope * mx
        residual = ((y - (intercept + slope * x)) ** 2).sum()
        return -slope.item(), intercept.item(), residual.item()

    def fit(self) -> bool:
        """Fit the power law to the recorded observations.

        Returns:
            ``True`` if a fit was produced, ``False`` if there are not yet
            enough observations (see ``min_observations``).
        """
        if not self._can_fit():
            return False

        if self._fixed_irreducible_loss is not None:
            irreducible_loss = self._fixed_irreducible_loss
            if min(self.losses) <= irreducible_loss:
                raise ValueError(
                    "all observed losses must exceed irreducible_loss "
                    f"({irreducible_loss}); got minimum {min(self.losses)}"
                )
        else:
            irreducible_loss = self._search_irreducible_loss()

        exponent, log_coefficient, _ = self._line_fit(irreducible_loss)
        self.irreducible_loss = irreducible_loss
        self.coefficient = math.exp(log_coefficient)
        self.exponent = exponent
        self._fitted = True
        return True

    def _search_irreducible_loss(self) -> float:
        # Golden-section search for the L_inf that minimizes the line-fit
        # residual. The bracket stays strictly below min(loss) so that
        # log(loss - L_inf) is always defined.
        lo = 0.0
        hi = min(self.losses) * (1.0 - 1e-6)
        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        c = hi - inv_phi * (hi - lo)
        d = lo + inv_phi * (hi - lo)
        fc = self._line_fit(c)[2]
        fd = self._line_fit(d)[2]
        for _ in range(100):
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - inv_phi * (hi - lo)
                fc = self._line_fit(c)[2]
            else:
                lo, c, fc = c, d, fd
                d = lo + inv_phi * (hi - lo)
                fd = self._line_fit(d)[2]
            if hi - lo < 1e-12:
                break
        return 0.5 * (lo + hi)

    def predict(self, step) -> Optional[float]:
        """Predict the loss at ``step`` from the current fit.

        Fits lazily if new observations have been recorded since the last fit.

        Args:
            step: optimization step to predict the loss at.

        Returns:
            The predicted loss, or ``None`` if there are not yet enough
            observations to fit.
        """
        step = float(step.item() if isinstance(step, torch.Tensor) else step)
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        if not self._fitted and not self.fit():
            return None
        assert (
            self.irreducible_loss is not None
            and self.coefficient is not None
            and self.exponent is not None
        )
        return self.irreducible_loss + self.coefficient * step ** (-self.exponent)

    def estimate_final_loss(self, total_steps) -> Optional[float]:
        """Estimate the loss at the end of training.

        Convenience wrapper around :meth:`predict` for the final step.

        Args:
            total_steps: total number of optimization steps the run will take.

        Returns:
            The estimated final loss, or ``None`` if there are not yet enough
            observations to fit.
        """
        return self.predict(total_steps)
