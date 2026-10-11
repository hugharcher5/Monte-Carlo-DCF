"""FMP errors must never reveal the API key (the error text is shown in the app)."""
from __future__ import annotations

import pytest

from valuation import analysts


class FakeResponse:
    def __init__(self, status):
        self.status_code, self.ok = status, status < 400

    def json(self):
        return [{"dcf": 123.4}]


@pytest.mark.parametrize("status", [402, 500])
def test_fmp_errors_hide_the_key(monkeypatch, status):
    monkeypatch.setenv("FMP_API_KEY", "SECRET123")
    monkeypatch.setattr(analysts.requests, "get", lambda *a, **k: FakeResponse(status))
    with pytest.raises(RuntimeError) as e:
        analysts.fmp_dcf("TM")
    assert "SECRET123" not in str(e.value)


def test_fmp_dcf_reads_value(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "SECRET123")
    monkeypatch.setattr(analysts.requests, "get", lambda *a, **k: FakeResponse(200))
    assert analysts.fmp_dcf("TM") == 123.4
