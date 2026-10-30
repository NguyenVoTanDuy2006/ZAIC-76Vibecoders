from __future__ import annotations


def get_huggingface_token(label: str = "HF_TOKEN") -> str:
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        raise RuntimeError("Kaggle secrets are unavailable; configure the HF_TOKEN secret in a Kaggle notebook") from exc
    token = UserSecretsClient().get_secret(label)
    if not token:
        raise RuntimeError(f"Kaggle secret {label!r} is empty or unavailable")
    return token
