import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api_client import HelpdeskClient, HelpdeskAPIError


class TestHelpdeskClient(unittest.TestCase):
    def setUp(self):
        self.client = HelpdeskClient(
            base_url="https://fake.helpdesk/api",
            api_key="fake_key",
            candidate_email="test@example.com",
            candidate_name="Test Candidate",
        )

    @patch("requests.Session.request")
    def test_handles_409_already_handled_gracefully(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 409
        mock_resp.text = '{"error": "already_handled", "message": "Ticket already handled"}'
        mock_req.return_value = mock_resp

        res = self.client.post_reply("T001", "nl", "question", "P4", "reason", "reply text")
        self.assertEqual(res.get("status"), "already_handled")
        self.assertEqual(res.get("status_code"), 409)
        # Should NOT retry 409
        self.assertEqual(mock_req.call_count, 1)

    @patch("time.sleep")
    @patch("requests.Session.request")
    def test_retries_on_503_service_unavailable(self, mock_req, mock_sleep):
        resp_503 = MagicMock()
        resp_503.status_code = 503

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.content = b'{"status": "ok"}'
        resp_200.json.return_value = {"status": "ok"}

        mock_req.side_effect = [resp_503, resp_200]

        res = self.client.get_status()
        self.assertEqual(res, {"status": "ok"})
        self.assertEqual(mock_req.call_count, 2)
        mock_sleep.assert_called()

    @patch("time.sleep")
    @patch("requests.Session.request")
    def test_retries_on_429_rate_limited(self, mock_req, mock_sleep):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "1.5"}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.content = b'{"tickets": []}'
        resp_200.json.return_value = {"tickets": []}

        mock_req.side_effect = [resp_429, resp_200]

        res, next_cursor = self.client.get_tickets()
        self.assertEqual(res, [])
        self.assertEqual(mock_req.call_count, 2)


if __name__ == "__main__":
    unittest.main()
