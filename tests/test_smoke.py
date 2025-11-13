import pytest
from agents.trainer import Trainer


def test_smoke_run():
    t = Trainer(episodes=2)
    results = t.run()
    assert len(results) == 2
    # each result is (reward, steps)
    for r, steps in results:
        assert isinstance(r, float)
        assert isinstance(steps, int)
        assert steps > 0
