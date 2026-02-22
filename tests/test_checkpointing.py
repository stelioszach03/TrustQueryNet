from pathlib import Path

import torch

from trustquerynet.training.checkpointing import load_checkpoint


def _payload_for(model, optimizer):
    return {
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "extra": {"history_entry": {"epoch": 2}},
    }


def test_load_checkpoint_requests_weights_only_false(monkeypatch):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    calls = []

    def fake_load(path, map_location=None, **kwargs):
        calls.append({"path": Path(path), "map_location": map_location, **kwargs})
        return _payload_for(model, optimizer)

    monkeypatch.setattr(torch, "load", fake_load)

    payload = load_checkpoint("dummy.ckpt", model, optimizer, map_location="cpu")

    assert payload["epoch"] == 2
    assert calls[0]["path"] == Path("dummy.ckpt")
    assert calls[0]["map_location"] == "cpu"
    assert calls[0]["weights_only"] is False


def test_load_checkpoint_falls_back_when_weights_only_kw_is_unsupported(monkeypatch):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    calls = []

    def fake_load(path, map_location=None, **kwargs):
        calls.append({"path": Path(path), "map_location": map_location, **kwargs})
        if "weights_only" in kwargs:
            raise TypeError("unexpected keyword argument 'weights_only'")
        return _payload_for(model, optimizer)

    monkeypatch.setattr(torch, "load", fake_load)

    payload = load_checkpoint("dummy.ckpt", model, optimizer, map_location="cpu")

    assert payload["epoch"] == 2
    assert len(calls) == 2
    assert calls[0]["weights_only"] is False
    assert "weights_only" not in calls[1]
