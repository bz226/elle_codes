from elle_jax_model.simulation import GrainGrowthConfig, initialize_order_parameters, run_simulation


def test_initialization_is_normalized() -> None:
    cfg = GrainGrowthConfig(nx=16, ny=12, num_grains=4, seed=123)
    phi = initialize_order_parameters(cfg)
    sums = phi.sum(axis=0)
    assert phi.shape == (4, 16, 12)
    assert ((sums - 1.0) ** 2).mean() < 1e-10


def test_run_simulation_shapes() -> None:
    cfg = GrainGrowthConfig(nx=12, ny=8, num_grains=3, seed=1)
    final_state, snapshots = run_simulation(cfg, steps=10, save_every=5)
    assert final_state.shape == (3, 12, 8)
    assert len(snapshots) == 2
