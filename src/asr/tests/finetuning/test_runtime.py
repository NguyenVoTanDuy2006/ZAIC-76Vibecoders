import unittest
import torch


class RuntimeTests(unittest.TestCase):
    def test_partial_accumulation_uses_actual_group_size(self):
        from asr.finetuning.optimization import OptimizerController
        from asr.finetuning.runtime import DistributedRuntime

        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1.0)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        controller = OptimizerController(
            opt, None, None, 100.0, DistributedRuntime("cpu")
        )
        for x in [1.0, 2.0]:
            controller.backward(model(torch.tensor([[x]])).sum(), 2)
        result = controller.step(model)
        self.assertTrue(result["successful"])
        self.assertAlmostEqual(float(model.weight.detach()), 0.85, places=6)

    def test_nonfinite_gradients_skip_scheduler_and_optimizer(self):
        from asr.finetuning.optimization import OptimizerController
        from asr.finetuning.runtime import DistributedRuntime

        model = torch.nn.Linear(1, 1, bias=False)
        old = model.weight.detach().clone()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, 0.5)
        controller = OptimizerController(
            opt, scheduler, None, 1.0, DistributedRuntime("cpu")
        )
        model.weight.grad = torch.full_like(model.weight, float("inf"))
        result = controller.step(model)
        self.assertFalse(result["successful"])
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertTrue(torch.equal(old, model.weight))


if __name__ == "__main__":
    unittest.main()
