"""
Robust HTTP client for the Roosterly Helpdesk API.

Implements production-grade reliability:
- Handles random 503s with exponential backoff retries.
- Respects rate limits (< 5 req/s) with proactive throttling and 429 Retry-After handling.
- Gracefully handles 409 already_handled errors without retrying.
- Supports cursor-based pagination for ticket ingestion.
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("helpdesk_client")

DEFAULT_BASE_URL = "https://nakywgkztvzkihdjnwbc.supabase.co/functions/v1/helpdesk"
MAX_RETRIES = 4
MIN_REQUEST_INTERVAL = 0.25  # Max 4 requests/sec (well below the 5 req/sec 429 threshold)


class HelpdeskAPIError(Exception):
    def __init__(self, status_code: int, error_code: str, message: str, details: Any = None):
        super().__init__(f"Helpdesk API Error {status_code} ({error_code}): {message}")
        self.status_code = status_code
        self.error_code = error_code
        self.details = details


class HelpdeskClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        candidate_email: Optional[str] = None,
        candidate_name: Optional[str] = None,
    ):
        self.base_url = (base_url or os.environ.get("HELPDESK_API_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("HELPDESK_API_KEY")
        self.candidate_email = candidate_email or os.environ.get("CANDIDATE_EMAIL")
        self.candidate_name = candidate_name or os.environ.get("CANDIDATE_NAME")

        if not self.api_key:
            raise ValueError("HELPDESK_API_KEY must be provided via environment variable or argument.")
        if not self.candidate_email:
            raise ValueError("CANDIDATE_EMAIL must be provided via environment variable or argument.")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "X-Candidate-Email": self.candidate_email,
            "Content-Type": "application/json",
            "User-Agent": "Nordavo-Support-Agent/2.0",
        })
        if self.candidate_name:
            self.session.headers["X-Candidate-Name"] = self.candidate_name

        self._last_request_time = 0.0

    def _throttle(self) -> None:
        """Proactively throttle calls to ensure we stay under 5 req/sec."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                response = self.session.request(method, url, timeout=15, **kwargs)
                
                # Success
                if response.status_code in (200, 201):
                    return response.json() if response.content else {}

                # 409 Already Handled: Customer must never get two answers; do not retry
                if response.status_code == 409:
                    logger.warning("Ticket already handled (409): %s", response.text)
                    return {"status": "already_handled", "status_code": 409}

                # 429 Rate Limited: Wait Retry-After (or back off), then retry
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 2.0))
                    logger.warning("Rate limited (429). Backing off for %.2fs (attempt %d/%d)", retry_after, attempt + 1, MAX_RETRIES)
                    time.sleep(retry_after)
                    continue

                # 503 Service Unavailable: Simulated random downtime; retry with exponential backoff
                if response.status_code == 503:
                    backoff = min(0.5 * (2 ** attempt), 4.0)
                    logger.warning("Helpdesk 503 unavailable. Retrying in %.2fs (attempt %d/%d)", backoff, attempt + 1, MAX_RETRIES)
                    time.sleep(backoff)
                    continue

                # Client error or unrecoverable error
                try:
                    err_json = response.json()
                    err_code = err_json.get("error", "api_error")
                    err_msg = err_json.get("message", response.text)
                except Exception:
                    err_code = "http_error"
                    err_msg = response.text

                raise HelpdeskAPIError(response.status_code, err_code, err_msg)

            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                backoff = min(0.5 * (2 ** attempt), 4.0)
                logger.warning("Network connection issue (%s). Retrying in %.2fs", exc, backoff)
                time.sleep(backoff)

        raise HelpdeskAPIError(503, "max_retries_exceeded", f"Failed {method} {path} after {MAX_RETRIES} attempts.")

    def get_status(self) -> Dict[str, Any]:
        """GET /status: Account mode, handled/open ticket counts."""
        return self._request("GET", "/status")

    def get_examples(self) -> List[Dict[str, Any]]:
        """GET /examples: 10 ground-truth examples labeled by Nordavo."""
        res = self._request("GET", "/examples")
        return res.get("examples", [])

    def get_tickets(
        self,
        status: str = "open",
        limit: int = 10,
        cursor: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """GET /tickets?status=...&limit=...&cursor=...: Paginated ticket retrieval."""
        params = {"status": status, "limit": min(limit, 10)}
        if cursor:
            params["cursor"] = cursor
        res = self._request("GET", "/tickets", params=params)
        return res.get("tickets", []), res.get("next_cursor")

    def get_all_open_tickets(self) -> List[Dict[str, Any]]:
        """Fetch all open tickets by traversing pagination cursors."""
        tickets: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        
        while True:
            batch, next_cursor = self.get_tickets(status="open", limit=10, cursor=cursor)
            tickets.extend(batch)
            if not next_cursor or not batch:
                break
            cursor = next_cursor
            
        return tickets

    def get_ticket(self, ticket_id: str) -> Dict[str, Any]:
        """GET /tickets/{id}: Retrieve a single ticket."""
        return self._request("GET", f"/tickets/{ticket_id}")

    def post_reply(
        self,
        ticket_id: str,
        language: str,
        category: str,
        priority: str,
        reason: str,
        draft_reply: str,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """POST /tickets/{id}/reply: Send reply directly to customer."""
        payload: Dict[str, Any] = {
            "language": language,
            "category": category,
            "priority": priority,
            "reason": reason,
            "draft_reply": draft_reply,
        }
        if confidence is not None:
            payload["confidence"] = round(float(confidence), 2)
        return self._request("POST", f"/tickets/{ticket_id}/reply", json=payload)

    def post_handoff(
        self,
        ticket_id: str,
        language: str,
        category: str,
        priority: str,
        reason: str,
        handoff_note: str,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """POST /tickets/{id}/handoff: Escalate ticket to human team."""
        payload: Dict[str, Any] = {
            "language": language,
            "category": category,
            "priority": priority,
            "reason": reason,
            "handoff_note": handoff_note,
        }
        if confidence is not None:
            payload["confidence"] = round(float(confidence), 2)
        return self._request("POST", f"/tickets/{ticket_id}/handoff", json=payload)

    def post_close(
        self,
        ticket_id: str,
        language: str,
        category: str,
        priority: str,
        reason: str,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """POST /tickets/{id}/close: Close without action (spam/empty)."""
        payload: Dict[str, Any] = {
            "language": language,
            "category": category,
            "priority": priority,
            "reason": reason,
        }
        if confidence is not None:
            payload["confidence"] = round(float(confidence), 2)
        return self._request("POST", f"/tickets/{ticket_id}/close", json=payload)

    def reset_practice(self) -> Dict[str, Any]:
        """POST /practice/reset: Re-opens all practice tickets for a fresh run."""
        return self._request("POST", "/practice/reset")
