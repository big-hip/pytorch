# Owner(s): ["module: nn"]

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils.loss_extrapolator import LossExtrapolator


def _power_law(steps, irreducible_loss, coefficient, exponent):
    return [irreducible_loss + coefficient * s ** (-exponent) for s in steps]


class TestLossExtrapolator(TestCase):
    def test_recovers_known_power_law(self):
        steps = [10, 20, 40, 80, 160, 320]
        losses = _power_law(steps, 1.5, 8.0, 0.4)
        ex = LossExtrapolator()
        for step, loss in zip(steps, losses):
            ex.update(step, loss)
        self.assertTrue(ex.fit())
        self.assertEqual(ex.irreducible_loss, 1.5, atol=1e-6, rtol=0)
        self.assertEqual(ex.coefficient, 8.0, atol=1e-5, rtol=0)
        self.assertEqual(ex.exponent, 0.4, atol=1e-6, rtol=0)

    def test_estimate_final_loss_extrapolates(self):
        steps = [10, 20, 40, 80, 160, 320]
        losses = _power_law(steps, 1.5, 8.0, 0.4)
        ex = LossExtrapolator()
        for step, loss in zip(steps, losses):
            ex.update(step, loss)
        expected = 1.5 + 8.0 * 10000 ** (-0.4)
        self.assertEqual(ex.estimate_final_loss(10000), expected, atol=1e-4, rtol=0)

    def test_returns_none_before_enough_observations(self):
        ex = LossExtrapolator(min_observations=4)
        ex.update(1, 5.0)
        ex.update(2, 4.0)
        self.assertFalse(ex.fit())
        self.assertIsNone(ex.estimate_final_loss(1000))
        self.assertIsNone(ex.predict(1000))

    def test_fixed_irreducible_loss(self):
        steps = [10, 20, 40, 80]
        losses = _power_law(steps, 1.5, 8.0, 0.4)
        ex = LossExtrapolator(irreducible_loss=1.5, min_observations=2)
        for step, loss in zip(steps, losses):
            ex.update(step, loss)
        self.assertTrue(ex.fit())
        self.assertEqual(ex.irreducible_loss, 1.5)
        expected = 1.5 + 8.0 * 10000 ** (-0.4)
        self.assertEqual(ex.predict(10000), expected, atol=1e-5, rtol=0)

    def test_fixed_irreducible_loss_above_observations_raises(self):
        ex = LossExtrapolator(irreducible_loss=10.0, min_observations=2)
        ex.update(1, 5.0)
        ex.update(2, 4.0)
        with self.assertRaisesRegex(ValueError, "must exceed irreducible_loss"):
            ex.fit()

    def test_accepts_tensor_inputs(self):
        steps = [10, 20, 40, 80, 160]
        losses = _power_law(steps, 0.5, 4.0, 0.5)
        ex = LossExtrapolator()
        for step, loss in zip(steps, losses):
            ex.update(torch.tensor(step), torch.tensor(loss))
        self.assertEqual(
            ex.estimate_final_loss(torch.tensor(5000)),
            0.5 + 4.0 * 5000 ** (-0.5),
            atol=1e-4,
            rtol=0,
        )

    def test_reset_clears_state(self):
        ex = LossExtrapolator()
        for step, loss in zip([1, 2, 3, 4], [4.0, 3.0, 2.5, 2.2]):
            ex.update(step, loss)
        self.assertTrue(ex.fit())
        ex.reset()
        self.assertEqual(ex.steps, [])
        self.assertIsNone(ex.irreducible_loss)
        self.assertFalse(ex.fit())

    def test_lazy_refit_after_update(self):
        steps = [10, 20, 40, 80, 160, 320]
        losses = _power_law(steps, 1.5, 8.0, 0.4)
        ex = LossExtrapolator()
        for step, loss in zip(steps, losses):
            ex.update(step, loss)
        first = ex.predict(10000)
        ex.update(640, 1.5 + 8.0 * 640 ** (-0.4))
        second = ex.predict(10000)
        self.assertEqual(first, second, atol=1e-4, rtol=0)

    @parametrize("bad_step", [0, -1])
    def test_invalid_step_raises(self, bad_step):
        ex = LossExtrapolator()
        with self.assertRaisesRegex(ValueError, "step must be"):
            ex.update(bad_step, 1.0)

    def test_invalid_loss_raises(self):
        ex = LossExtrapolator()
        with self.assertRaisesRegex(ValueError, "loss must be finite"):
            ex.update(1, float("nan"))

    def test_invalid_construction_raises(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            LossExtrapolator(irreducible_loss=-1.0)
        with self.assertRaisesRegex(ValueError, "min_observations"):
            LossExtrapolator(min_observations=2)

    def test_recovers_noisy_power_law_within_tolerance(self):
        torch.manual_seed(0)
        steps = [2**i for i in range(2, 12)]
        clean = _power_law(steps, 1.2, 6.0, 0.35)
        noise = torch.empty(len(steps)).uniform_(-0.005, 0.005).tolist()
        losses = [c * (1 + n) for c, n in zip(clean, noise)]
        ex = LossExtrapolator()
        for step, loss in zip(steps, losses):
            ex.update(step, loss)
        expected = 1.2 + 6.0 * 100000 ** (-0.35)
        self.assertEqual(ex.estimate_final_loss(100000), expected, atol=5e-2, rtol=0)


instantiate_parametrized_tests(TestLossExtrapolator)


if __name__ == "__main__":
    run_tests()
