"""Regression tests for public motion loading and online SPGP prediction."""

import contextlib
import io
import pickle
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run_ws"))

from SparseIAAL.dataset import list_motion_files, resolve_motion_directory
from SparseIAAL.sparse_IAAL_te import EPSILON, SparseIAALClass
from SparseIAAL.sparse_IAAL import SparseIAALClass as PlainSparseIAALClass
from estimator_options import parse_estimator_args


class SPGPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.checkout = Path(self.temporary.name).resolve() / "dataset"
        self.motion_dir = self.checkout / "data" / "whole_motions"
        self.motion_dir.mkdir(parents=True)
        self.cache_path = Path(self.temporary.name) / "cache" / "kernels.pkl"
        self.write_motion("motion_variation_2.pkl", 0.5)
        self.write_motion("motion_normal_10.pkl", 0.2)
        self.write_motion("motion_normal_2.pkl", 0.0)
        (self.motion_dir / ".DS_Store").write_text("ignored")
        (self.motion_dir / "README.md").write_text("ignored")

    def write_motion(self, filename, offset):
        t = np.linspace(0., 1., 12)
        receiver = np.zeros((len(t), 34, 3))
        receiver[:, 0, :] = np.column_stack((2.5 - 2. * t, offset + .1 * t, np.ones(len(t))))
        obj = np.column_stack((.45 + .1 * t, -.3 + offset + .02 * t, 1.2 + .1 * t))
        with (self.motion_dir / filename).open("wb") as stream:
            pickle.dump({"pose_receiver": receiver, "pose_object": obj}, stream)

    def model(self, **overrides):
        model_class = overrides.pop("model_class", SparseIAALClass)
        options = dict(
            filepath=self.checkout, kernel_data_path=self.cache_path,
            motion_type="all", sum_sample=3, use_te=True,
            chunking_size=4, sliding_window=3, inducing_rate=.4,
        )
        options.update(overrides)
        if model_class is PlainSparseIAALClass:
            options.pop("use_te")
            options.pop("chunking_size")
        with contextlib.redirect_stdout(io.StringIO()):
            return model_class(**options)

    def cache(self):
        with self.cache_path.open("rb") as stream:
            return pickle.load(stream)

    def save_cache(self, cache):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as stream:
            pickle.dump(cache, stream)

    def test_dataset_locations_and_deterministic_filtering(self):
        for path in (self.checkout, self.checkout / "data", self.motion_dir):
            self.assertEqual(resolve_motion_directory(path), self.motion_dir)
        self.assertEqual(
            list_motion_files(self.motion_dir),
            ["motion_normal_2.pkl", "motion_normal_10.pkl", "motion_variation_2.pkl"],
        )
        self.assertEqual(list_motion_files(self.motion_dir, "variation"), ["motion_variation_2.pkl"])
        self.assertEqual(list_motion_files(self.motion_dir, "normal", 1), ["motion_normal_2.pkl"])

    def test_cli_defaults_and_explicit_options(self):
        args = parse_estimator_args(["--dataset-dir", str(self.checkout)])
        self.assertEqual(args.motion_dir, self.motion_dir)
        self.assertEqual(args.motion_type, "normal")
        args = parse_estimator_args([
            "--dataset-dir", str(self.motion_dir), "--motion-type", "variation",
            "--max-data-num", "1", "--no-kernel-cache", "--no-render",
        ])
        self.assertEqual(args.motion_type, "variation")
        self.assertIsNone(args.kernel_cache)
        self.assertFalse(args.render)
        self.assertEqual(self.model(motion_type=args.motion_type, max_data_num=args.max_data_num).N, 1)

    def test_cli_default_uses_the_dataset_submodule(self):
        repository = Path(self.temporary.name).resolve() / "repository"
        submodule = repository / "run_ws" / "data" / "dataset"
        submodule.parent.mkdir(parents=True)
        submodule.symlink_to(self.checkout, target_is_directory=True)
        with mock.patch("estimator_options.REPOSITORY_DIR", repository):
            options = parse_estimator_args([])
        self.assertEqual(options.motion_dir, self.motion_dir)

    def test_missing_dataset_and_empty_group_have_clear_errors(self):
        with self.assertRaisesRegex(FileNotFoundError, "--dataset-dir"):
            resolve_motion_directory(self.checkout / "missing")
        empty = self.checkout / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "No normal motion files"):
            list_motion_files(empty, "normal")

    def test_cache_reuse_is_independent_of_dictionary_order(self):
        original = self.model()
        cache = self.cache()
        cache["models"] = dict(reversed(list(cache["models"].items())))
        self.save_cache(cache)
        with mock.patch.object(SparseIAALClass, "make_gpr_kernel", side_effect=AssertionError("unexpected rebuild")):
            restored = self.model()
        for first, second in zip(original.gpr_matrix_list, restored.gpr_matrix_list):
            np.testing.assert_array_equal(first["K_tilda_mu"], second["K_tilda_mu"])
        position = original.receiver_pos[0][0]
        with contextlib.redirect_stdout(io.StringIO()):
            np.testing.assert_allclose(original.estimate_target(position), restored.estimate_target(position))

    def test_legacy_and_mismatched_models_are_rebuilt(self):
        original = self.model()
        self.save_cache(original.gpr_matrix_list)  # Old, unnamed cache format.
        with self.assertWarnsRegex(UserWarning, "rebuilding"):
            self.model()
        cache = self.cache()
        names = list(cache["models"])
        cache["models"][names[0]], cache["models"][names[1]] = cache["models"][names[1]], cache["models"][names[0]]
        self.save_cache(cache)
        with self.assertWarnsRegex(UserWarning, "rebuilding"):
            restored = self.model()
        for first, second in zip(original.gpr_matrix_list, restored.gpr_matrix_list):
            np.testing.assert_allclose(first["K_tilda_mu"], second["K_tilda_mu"])

    def test_data_and_kernel_changes_invalidate_cache(self):
        self.model()
        original_metadata = self.cache()["metadata"]
        self.write_motion("motion_normal_2.pkl", 0.8)
        with self.assertWarnsRegex(UserWarning, "rebuilding"):
            self.model()
        self.assertNotEqual(original_metadata["motions"], self.cache()["metadata"]["motions"])
        with self.assertWarnsRegex(UserWarning, "rebuilding"):
            changed = self.model(inducing_rate=.6)
        self.assertEqual(self.cache()["metadata"]["inducing_rate"], .6)
        self.assertEqual(len(changed.gpr_matrix_list[0]["inducing_data_pos"]), 7)
        with self.assertWarnsRegex(UserWarning, "rebuilding"):
            self.model(inducing_rate=.6, hz=60)
        self.assertEqual(self.cache()["metadata"]["hz"], 60)
        with self.assertWarnsRegex(UserWarning, "rebuilding"):
            limited = self.model(inducing_rate=.6, hz=60, max_data_num=1)
        self.assertEqual(set(self.cache()["models"]), set(limited.motion_files))

    def test_stationary_receiver_and_zero_flow_remain_finite(self):
        model = self.model(kernel_data_path=None)
        zero = np.zeros(2)
        for flow in (np.zeros((3, 2)), np.ones((3, 2)), np.full((3, 2), EPSILON * .01)):
            distances = model.compute_traj_dist(.1, flow, np.zeros(3), zero)
            self.assertTrue(np.isfinite(distances).all())
        position = model.receiver_pos[0][0].copy()
        with np.errstate(divide="raise", invalid="raise"), contextlib.redirect_stdout(io.StringIO()):
            for _ in range(10):
                self.assertTrue(np.isfinite(model.estimate_target(position)).all())
            self.assertTrue(np.isfinite(model.estimate_target(position + [-.1, 0.])).all())
        self.assertTrue(np.isfinite(model.d_list_obs).all())

    def test_reset_matches_a_fresh_episode(self):
        used = self.model(kernel_data_path=None)
        fresh = self.model(kernel_data_path=None)
        with contextlib.redirect_stdout(io.StringIO()):
            for position in used.receiver_pos[0][4:8]:
                used.estimate_target(position)
            used.reset_obs()
            self.assertTrue(all(np.isnan(chunk).all() for chunk in used.action_deq))
            self.assertEqual(used.traj_obs, [])
            self.assertEqual(used.d_list_obs, [])
            target = fresh.receiver_pos[0][0]
            np.testing.assert_allclose(used.estimate_target(target), fresh.estimate_target(target))

    def test_equal_distances_without_ensemble_and_single_motion(self):
        model = self.model(kernel_data_path=None, use_te=False)
        with np.errstate(divide="raise", invalid="raise"):
            target = model.get_target_pos(
                model.receiver_pos[0][0], np.arange(model.N), np.ones(model.N),
                model.receiver_pos, model.object_pos, 0,
            )
        self.assertTrue(np.isfinite(target).all())
        solo = self.model(kernel_data_path=None, max_data_num=1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(np.isfinite(solo.estimate_target(solo.receiver_pos[0][0])).all())

    def test_observed_velocity_uses_observation_frequency(self):
        model = self.model(kernel_data_path=None, hz_obs=10)
        first = model.receiver_pos[0][0].copy()
        with contextlib.redirect_stdout(io.StringIO()):
            model.estimate_target(first)
            original = model.compute_traj_dist
            with mock.patch.object(model, "compute_traj_dist", wraps=original) as compute:
                model.estimate_target(first + [.1, 0.])
        np.testing.assert_allclose(compute.call_args[0][3], [1., 0.])

    def test_plain_model_matches_shared_predictions_and_reuses_cache(self):
        shared = self.model(use_te=False)
        with mock.patch.object(SparseIAALClass, "make_gpr_kernel", side_effect=AssertionError("unexpected rebuild")):
            plain = self.model(model_class=PlainSparseIAALClass)
        self.assertFalse(plain.use_te)
        self.assertEqual(plain.motion_files, shared.motion_files)
        positions = list(shared.receiver_pos[0][:4]) + [shared.receiver_pos[0][3]] * 5
        with np.errstate(divide="raise", invalid="raise"), contextlib.redirect_stdout(io.StringIO()):
            for position in positions:
                expected = shared.estimate_target(position)
                actual = plain.estimate_target(position)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected)
            plain.reset_obs()
            shared.reset_obs()
            self.assertEqual(plain.traj_obs, [])
            self.assertEqual(plain.d_list_obs, [])
            np.testing.assert_allclose(
                plain.estimate_target(positions[0]), shared.estimate_target(positions[0]),
            )

    def test_plain_original_positional_arguments_and_regression_api(self):
        with contextlib.redirect_stdout(io.StringIO()):
            # Preserve the original order: sum_sample immediately follows hz_obs.
            plain = PlainSparseIAALClass(self.checkout, 30, 10, 2, .1, False, .2, 1, 3, .4)
        self.assertEqual(plain.sum_sample, 2)
        self.assertEqual(plain.hz_obs, 10)
        self.assertEqual(plain.forecasting_step, 6)
        self.assertEqual(plain.N, 1)
        model = plain.gpr_matrix_list[0]
        test_pos = plain.receiver_pos[0][:2]
        legacy = plain.sparse_gaussian_process_regression(
            data_pos_inducing=plain.receiver_pos_inducing[0], test_pos=test_pos, kernel_data=model,
        )
        shared = plain.sparse_gaussian_process_regression(test_pos, model)
        for original, current in zip(legacy, shared):
            np.testing.assert_allclose(original, current)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(np.isfinite(plain.estimate_target(test_pos[0])).all())

    def test_plain_cosine_uses_each_trajectory_norm(self):
        plain = self.model(model_class=PlainSparseIAALClass, kernel_data_path=None)
        np.testing.assert_allclose(
            plain.cosine_distance(np.array([[1., 0.], [10., 0.], [-1., 0.]]), np.array([1., 0.])),
            [0., 0., 2.],
        )
        self.assertTrue(np.isfinite(plain.cosine_distance(np.zeros((3, 2)), np.zeros(2))).all())

    def test_forecast_clamps_to_motion_end_in_both_models(self):
        for model_class in (SparseIAALClass, PlainSparseIAALClass):
            with self.subTest(model=model_class.__module__):
                model = self.model(
                    model_class=model_class, kernel_data_path=None, use_te=False,
                    max_data_num=1, forecasting_time=60.,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    target = model.estimate_target(model.receiver_pos[0][-1])
                np.testing.assert_allclose(target, model.object_pos[0][-1])

    def test_pooled_flow_works_and_keeps_all_demo_candidates(self):
        for model_class in (SparseIAALClass, PlainSparseIAALClass):
            with self.subTest(model=model_class.__module__):
                model = self.model(
                    model_class=model_class, kernel_data_path=None, use_te=False,
                    use_gpr_whole=True,
                )
                self.assertEqual(model.N, 3)
                position = model.receiver_pos[0][2]
                mu, var = model.get_mu_var_list(position)
                self.assertEqual(mu.shape, (3, 2))
                self.assertEqual(var.shape, (3,))
                self.assertTrue(np.isfinite(mu).all())
                self.assertTrue(np.isfinite(var).all())
                expected = np.mean([
                    giver[np.argmin(np.linalg.norm(receiver - position, axis=1))]
                    for receiver, giver in zip(model.receiver_pos, model.object_pos)
                ], axis=0)
                with contextlib.redirect_stdout(io.StringIO()):
                    np.testing.assert_allclose(model.estimate_target(position), expected)
                    self.assertTrue(np.isfinite(model.estimate_target(position)).all())


if __name__ == "__main__":
    unittest.main()
