"""
Unit tests for the webhook alerting helper.
All HTTP calls are mocked — no network required.
"""

import requests
import pytest

from main import send_alert


class TestSendAlert:

    def test_no_op_when_webhook_url_is_none(self, caplog):
        send_alert("Test Subject", "Test body", webhook_url=None)
        assert "suppressed" in caplog.text

    def test_posts_to_correct_url(self, mocker):
        mock_post = mocker.patch("main.requests.post")
        mock_post.return_value.raise_for_status = mocker.MagicMock()

        send_alert("Subject", "Body", "https://webhook.example.com/abc")

        mock_post.assert_called_once()
        call_url = mock_post.call_args.args[0]
        assert call_url == "https://webhook.example.com/abc"

    def test_payload_contains_subject_and_body(self, mocker):
        mock_post = mocker.patch("main.requests.post")
        mock_post.return_value.raise_for_status = mocker.MagicMock()

        send_alert("Pipeline FAILED", "run_id: abc123\nerror: boom", "https://hook.example.com")

        payload = mock_post.call_args.kwargs["json"]
        assert "Pipeline FAILED" in payload["text"]
        assert "run_id: abc123" in payload["text"]
        assert "boom" in payload["text"]

    def test_payload_key_is_text(self, mocker):
        """Teams and Slack both consume a top-level 'text' key."""
        mock_post = mocker.patch("main.requests.post")
        mock_post.return_value.raise_for_status = mocker.MagicMock()

        send_alert("S", "B", "https://hook.example.com")

        payload = mock_post.call_args.kwargs["json"]
        assert "text" in payload

    def test_timeout_is_set(self, mocker):
        mock_post = mocker.patch("main.requests.post")
        mock_post.return_value.raise_for_status = mocker.MagicMock()

        send_alert("S", "B", "https://hook.example.com")

        assert mock_post.call_args.kwargs.get("timeout") is not None

    def test_connection_error_is_non_fatal(self, mocker):
        mocker.patch("main.requests.post", side_effect=requests.ConnectionError("refused"))
        # Must not raise
        send_alert("Alert", "Body", "https://webhook.example.com")

    def test_http_error_is_non_fatal(self, mocker):
        mock_post = mocker.patch("main.requests.post")
        mock_post.return_value.raise_for_status.side_effect = requests.HTTPError("500")
        send_alert("Alert", "Body", "https://webhook.example.com")

    def test_timeout_error_is_non_fatal(self, mocker):
        mocker.patch("main.requests.post", side_effect=requests.Timeout())
        send_alert("Alert", "Body", "https://webhook.example.com")

    def test_not_called_with_empty_url(self, mocker):
        mock_post = mocker.patch("main.requests.post")
        send_alert("S", "B", webhook_url=None)
        mock_post.assert_not_called()
