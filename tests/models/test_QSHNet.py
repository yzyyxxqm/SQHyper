import unittest

import torch

from models.QSHNet import HypergraphLearner, Model, SpikeRouter
from utils.configs import get_configs


class TestQSHNet(unittest.TestCase):

    def test_spike_router_starts_as_identity_with_moderate_event_gate(self):
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
        self.assertGreater(route_state["event_gate"].mean().item(), 0.001)
        self.assertLess(route_state["event_gate"].mean().item(), 0.01)

    def test_hypergraph_learner_uses_nodewise_conditioned_quaternion_gate(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        linear_out = torch.randn(2, 3, 8)
        event_gate = torch.zeros(2, 3)

        quat_gate = learner.compute_quaternion_gate(0, linear_out, event_gate)

        self.assertEqual(quat_gate.shape, torch.Size((2, 3, 1)))
        self.assertTrue(torch.all(quat_gate >= 0.0))
        self.assertTrue(torch.all(quat_gate <= 1.0))
        self.assertLess(quat_gate.mean().item(), 0.1)

    def test_quaternion_residual_is_bounded_relative_to_linear_path(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        linear_out = torch.ones(2, 3, 8)
        quat_out = torch.full((2, 3, 8), 20.0)
        alpha = torch.ones(2, 3, 1)

        bounded_residual = learner.bound_quaternion_residual(
            linear_out=linear_out,
            quat_out=quat_out,
            alpha=alpha,
        )

        residual_norm = bounded_residual.norm(dim=-1)
        linear_norm = linear_out.norm(dim=-1)

        self.assertTrue(torch.all(residual_norm <= 0.25 * linear_norm + 1e-6))

    def test_event_delta_normalization_centers_each_hyperedge_feature_vector(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        event_delta = torch.randn(2, 3, 8) * 5.0 + 7.0

        normalized_delta = learner.normalize_event_delta(0, event_delta, target="temporal")

        self.assertEqual(normalized_delta.shape, torch.Size((2, 3, 8)))
        self.assertTrue(
            torch.allclose(
                normalized_delta.mean(dim=-1),
                torch.zeros(2, 3),
                atol=1e-5,
            )
        )

    def test_event_injection_adds_bounded_delta_to_both_hyperedge_paths(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        event_delta = torch.full((2, 3, 8), 0.5)
        main_state = torch.randn(2, 3, 8)
        event_scale = torch.tensor(0.1)

        temporal_injected = learner.apply_event_injection(
            layer_idx=0,
            main_state=main_state,
            event_delta=event_delta,
            event_scale=event_scale,
            target="temporal",
        )
        variable_injected = learner.apply_event_injection(
            layer_idx=0,
            main_state=main_state,
            event_delta=event_delta,
            event_scale=event_scale,
            target="variable",
        )

        expected = main_state + 0.05
        self.assertTrue(torch.allclose(temporal_injected, expected, atol=1e-6))
        self.assertTrue(torch.allclose(variable_injected, expected, atol=1e-6))

    def test_event_scale_is_capped_without_disturbing_small_initial_value(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)

        initial_scale = learner.compute_event_scale(0).item()
        with torch.no_grad():
            learner.event_residual_scale[0].fill_(10.0)
        capped_scale = learner.compute_event_scale(0).item()

        self.assertGreater(initial_scale, 0.05)
        self.assertLess(initial_scale, 0.12)
        self.assertLessEqual(capped_scale, 0.12 + 1e-6)

    def test_event_scale_density_modulation_keeps_base_scale_at_baseline_density(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        base_scale = learner.compute_event_scale(0)
        route_density = torch.full((2, 3, 1), learner.event_density_baseline)

        modulated_scale = learner.modulate_event_scale(base_scale, route_density, target="variable")

        expected_scale = torch.ones_like(route_density) * base_scale
        self.assertTrue(torch.allclose(modulated_scale, expected_scale))

    def test_event_scale_density_modulation_reduces_scale_for_dense_routes_on_variable_path(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        base_scale = torch.tensor(0.1)
        route_density = torch.ones(2, 3, 1)

        modulated_scale = learner.modulate_event_scale(base_scale, route_density, target="variable")

        expected_scale = base_scale * (1.0 - learner.variable_event_density_penalty_max)
        self.assertTrue(torch.allclose(modulated_scale, torch.full_like(route_density, expected_scale)))

    def test_event_scale_density_modulation_keeps_temporal_path_unchanged_even_for_dense_routes(self):
        learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
        base_scale = torch.tensor(0.1)
        route_density = torch.ones(2, 3, 1)

        modulated_scale = learner.modulate_event_scale(base_scale, route_density, target="temporal")

        self.assertTrue(torch.allclose(modulated_scale, torch.full_like(route_density, base_scale)))

    def test_spike_router_caps_retain_gate_drop_under_large_scale(self):
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

        with torch.no_grad():
            router.membrane_proj.weight.zero_()
            router.membrane_proj.bias.fill_(-10.0)
            router.retain_log_scale.fill_(5.0)

        _, _, route_state = router(
            obs, mask_d, variable_incidence_matrix, variable_indices_flattened
        )

        self.assertGreaterEqual(route_state["retain_gate"].min().item(), 0.9)
        self.assertLessEqual(route_state["retain_gate"].max().item(), 1.0)

    def test_hypergraph_learner_initializes_event_residual_with_small_nonzero_scale(self):
        learner = HypergraphLearner(n_layers=2, d_model=8, n_heads=1, time_length=4)

        event_scales = [
            torch.sigmoid(scale.detach()).item()
            for scale in learner.event_residual_scale
        ]

        for event_scale in event_scales:
            self.assertGreater(event_scale, 0.05)
            self.assertLess(event_scale, 0.15)

    def test_model_forward_runs_with_default_configs(self):
        configs = get_configs(args=["--model_name", "QSHNet", "--model_id", "QSHNet"])
        model = Model(configs)

        x = torch.randn(configs.batch_size, configs.seq_len, configs.enc_in)
        result_dict = model(**{"x": x, "exp_stage": "test"})

        self.assertEqual(result_dict["pred"].shape, torch.Size((configs.batch_size, configs.pred_len, configs.c_out)))
        self.assertEqual(result_dict["true"].shape, torch.Size((configs.batch_size, configs.pred_len, configs.c_out)))
