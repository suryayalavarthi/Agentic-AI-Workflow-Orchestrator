from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.server import web_scraper


# web_scraper is wrapped by @mcp.tool(); access the underlying function via .fn
_web_scraper_fn = web_scraper.fn


def test_web_scraper_success(mock_settings):
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Hello World</p></body></html>"
    mock_response.raise_for_status.return_value = None

    with patch("src.server._fetch_url", return_value=mock_response):
        result = _web_scraper_fn("https://example.com")

    assert "Hello World" in result


def test_web_scraper_strips_scripts(mock_settings):
    mock_response = MagicMock()
    mock_response.text = (
        "<html><body>"
        "<script>alert('x')</script>"
        "<style>.x{}</style>"
        "<p>Content here</p>"
        "</body></html>"
    )
    mock_response.raise_for_status.return_value = None

    with patch("src.server._fetch_url", return_value=mock_response):
        result = _web_scraper_fn("https://example.com")

    assert "Content here" in result
    assert "alert" not in result


def test_web_scraper_truncates(mock_settings):
    mock_response = MagicMock()
    mock_response.text = f"<html><body><p>{'a' * 10000}</p></body></html>"
    mock_response.raise_for_status.return_value = None

    with patch("src.server._fetch_url", return_value=mock_response):
        result = _web_scraper_fn("https://example.com")

    assert len(result) <= 5000
