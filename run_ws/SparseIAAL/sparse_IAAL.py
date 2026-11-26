"""SPGP trajectory prediction without temporal ensembling.

Keep the original constructor's positional arguments while sharing the maintained
SPGP implementation with sparse_IAAL_te.
"""

from .sparse_IAAL_te import SparseIAALClass as _SparseIAALClass


class SparseIAALClass(_SparseIAALClass):
    def __init__(
        self,
        filepath,
        hz=30,
        hz_obs=None,
        sum_sample=5,
        kappa=0.1,
        use_gpr_whole=False,
        forecasting_time=0.,
        max_data_num=None,
        sliding_window=None,
        inducing_rate=0.3,
        *,
        kernel_data_path=None,
        save_kernel_data=False,
        motion_type="all",
    ):
        super().__init__(
            filepath=filepath,
            hz=hz,
            hz_obs=hz_obs,
            kernel_data_path=kernel_data_path,
            save_kernel_data=save_kernel_data,
            sum_sample=sum_sample,
            kappa=kappa,
            use_gpr_whole=use_gpr_whole,
            forecasting_time=forecasting_time,
            max_data_num=max_data_num,
            sliding_window=sliding_window,
            inducing_rate=inducing_rate,
            use_te=False,
            motion_type=motion_type,
        )

    def sparse_gaussian_process_regression(self, data_pos_inducing, test_pos, kernel_data=None):
        """Accept both the original three-argument call and the shared engine call."""
        if kernel_data is None:
            return super().sparse_gaussian_process_regression(data_pos_inducing, test_pos)
        model = dict(kernel_data, inducing_data_pos=data_pos_inducing)
        return super().sparse_gaussian_process_regression(test_pos, model)
