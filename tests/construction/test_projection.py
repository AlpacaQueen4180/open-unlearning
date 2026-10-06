import pytest
import torch

from construction.projection import ProjectionConfig, project_block, project_gradients


@pytest.mark.parametrize("shape", [(), (5,), (4, 3), (2, 3, 4)])
def test_zero_safety_and_missing_block_preserve_task(shape):
    task = torch.randn(shape)
    assert torch.equal(project_block(task, torch.zeros_like(task)), task)
    assert torch.equal(project_block(task, None), task)


def test_global_conflict_gates_all_blocks_and_does_not_add_safety_only_gradient():
    task = {"a": torch.tensor([1.]), "b": torch.tensor([3.])}
    safety = {"a": torch.tensor([-1.]), "b": torch.tensor([1.]), "unused": torch.ones(2)}
    result, dot = project_gradients(task, safety)
    assert dot == 2
    assert result.keys() == task.keys()
    assert all(torch.equal(task[k], result[k]) for k in task)


def test_exact_projection_orthogonality_and_norm():
    torch.manual_seed(31)
    safety, task = torch.randn(9, 7), torch.randn(9, 7)
    projected = project_block(task, safety, ProjectionConfig(rank=3, method="exact"))
    basis = torch.linalg.svd(safety, full_matrices=False).U[:, :3]
    torch.testing.assert_close(basis.T @ projected, torch.zeros(3, 7), atol=2e-6, rtol=0)
    assert projected.norm() <= task.norm() + 1e-6


def test_rank_clamped_and_conflicting_scalar_removed():
    result, dot = project_gradients({"x": torch.tensor(2.)}, {"x": torch.tensor(-3.)})
    assert dot < 0 and result["x"] == 0


def test_randomized_svd_deterministic_and_does_not_consume_rng():
    task, safety = torch.randn(30, 20), torch.randn(30, 20)
    rng = torch.random.get_rng_state().clone()
    config = ProjectionConfig(rank=3)
    a = project_block(task, safety, config, step=7, name="layer")
    assert torch.equal(torch.random.get_rng_state(), rng)
    b = project_block(task, safety, config, step=7, name="layer")
    assert torch.equal(a, b)


def test_invalid_gradient_and_config_rejected():
    with pytest.raises(ValueError):
        ProjectionConfig(rank=0)
    with pytest.raises(ValueError):
        project_block(torch.tensor([float("nan")]), torch.ones(1))
