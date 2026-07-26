import torch

from muscriptor.transcription_model import _ModelConfig, _build_model


def test_generate_runs_inside_inference_mode(monkeypatch):
    model = _build_model(
        torch.device("cpu"),
        _ModelConfig(dim=8, num_heads=2, num_layers=1, card=8),
    )
    model.eval()
    inference_mode_states = []

    def sample_next_token(*_args, **_kwargs):
        inference_mode_states.append(torch.is_inference_mode_enabled())
        return torch.tensor([1])

    monkeypatch.setattr(model, "_sample_next_token", sample_next_token)
    generated = list(
        model.generate(
            num_samples=1,
            max_gen_len=2,
            use_sampling=False,
            early_stop_on_token=1,
        )
    )

    assert [step.tolist() for step in generated] == [[1]]
    assert inference_mode_states == [True]
