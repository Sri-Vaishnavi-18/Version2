import pytest

from contract_lens.sources import _validate_public_http_url


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/privacy",
        "http://127.0.0.1/privacy",
        "https://user:pass@example.com/privacy",
    ],
)
def test_rejects_non_public_or_invalid_policy_urls(url):
    with pytest.raises(ValueError):
        _validate_public_http_url(url)
