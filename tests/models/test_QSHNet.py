import unittest

import torch

from models.QSHNet import HypergraphLearner, Model, SpikeRouter
from utils.configs import get_configs


class TestQSHNet(unittest.TestCase):

    def test_spike_router_starts_as_identity_with_silent_event_path(self):
        router = SpikeRouter(d_model=8)
        obs = torch.randn(2, 3, 8)
        mask_d = torch.ones_like(obs)
        variable_incidence_matrix = torch.tensor([
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ])
        variable_indices_flattened = torch.tensor([
            [0, 0, 1],
            [0, 1, 0],
        ])

        obs_base, obs_event, route_state = router(
            obs, mask_d, variable_incidence_matrix, variable_indices_flattened
        )

        self.assertEqual(route_state["retain_gate"].shape, torch.Size((2, 3)))
        self.assertEqual(route_state["event_gate"].shape, torch.Size((2, 3)))
        self.assertEqual(route_state["route_logit"].shape, torch.Size((2, 3)))
        self.assertTrue(torch.allclose(obs_base, obs, atol=1e-6))
        self.assertTrue(torch.allclose(obs_event, torch.zeros_like(obs), atol=1e-6))
        self.assertGreater(route_state["retain_gate"].min().item(), 0.99)
        self.assertLess(route_state["event_gate"].max().item(), 0.01)

    def test_hypergraph_learner_uses_nodewise_conditioned_quaternion_gate(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        linear_out = torch.randn(2, 3, 8)
        event_gate = torch.zeros(2, 3)

        quat_gate = learner.compute_quaternion_gate(0, linear_out, event_gate)

        self.assertEqual(quat_gate.shape, torch.Size((2, 3, 1)))
        self.assertTrue(torch.all(quat_gate >= 0.0))
        self.assertTrue(torch.all(quat_gate <= 1.0))
        self.assertLess(quat_gate.mean().item(), 0.1)

    def test_model_forward_runs_with_default_configs(self):
        configs = get_configs(args=["--model_name", "QSHNet", "--model_id", "QSHNet"])
        model = Model(configs)

        x = torch.randn(configs.batch_size, configs.seq_len, configs.enc_in)
        result_dict = model(**{"x": x, "exp_stage": "test"})

        self.assertEqual(result_dict["pred"].shape, torch.Size((configs.batch_size, configs.pred_len, configs.c_out)))
        self.assertEqual(result_dict["true"].shape, torch.Size((configs.batch_size, configs.pred_len, configs.c_out)))
