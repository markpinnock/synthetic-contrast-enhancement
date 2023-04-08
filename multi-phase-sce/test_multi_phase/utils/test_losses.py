import numpy as np
import pytest
from typing import Callable, Tuple

from multi_phase.utils.losses import focused_mae, FocusedMetric


#-------------------------------------------------------------------------

@pytest.fixture
def create_test_tensor() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Create test tensor for MAE and metric tests """

    a = np.zeros((1, 2, 4, 4), dtype="float32")
    b = np.ones((1, 2, 4, 4), dtype="float32")
    b[:, 0, 1, 1] = 2
    b[:, 1, 1, 1] = 4

    m = np.zeros((1, 2, 4, 4), dtype="float32")
    m[:, :, 1, 1] = 1

    return a, b, m


#-------------------------------------------------------------------------

def test_focused_mae(create_test_tensor: Callable) -> None:
    """ Test focused mean absolute error """

    a, b, m = create_test_tensor
    assert np.allclose(np.array(focused_mae(a, b, m)), np.array([1.125, 3.0]))


#-------------------------------------------------------------------------

def test_focused_metric(create_test_tensor: Callable):
    """ Test focused metric """

    metric = FocusedMetric()
    a, b, m = create_test_tensor
    
    metric.update_state(a, b, m)
    assert np.allclose(metric.result().numpy(), np.array([1.125, 3.0]))
    metric.reset_states()
    assert np.allclose(metric.result().numpy(), np.array([0.0, 0.0]))


#-------------------------------------------------------------------------

def test_metric_loop(create_test_tensor: Callable):
    """ Test focused metric operating in training loop """

    metric = FocusedMetric()
    a, b, m = create_test_tensor
    
    for _ in range(5):
        metric.update_state(a, b, m)

    assert np.allclose(metric.result().numpy(), np.array([1.125, 3.0]))
    metric.reset_states()
    assert np.allclose(metric.result().numpy(), np.array([0.0, 0.0]))
