from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.core.config import get_settings
from src.services import calendar_service


class FakeRequest:
    def __init__(self, response, observed, name, **kwargs):
        self.response = response
        self.observed = observed
        self.name = name
        self.kwargs = kwargs

    def execute(self):
        self.observed.append((self.name, self.kwargs))
        return self.response


class FakeFreeBusy:
    def __init__(self, observed, response):
        self.observed = observed
        self.response = response

    def query(self, *, body):
        return FakeRequest(
            self.response,
            self.observed,
            "freebusy",
            body=body,
        )


class FakeEvents:
    def __init__(self, observed, response):
        self.observed = observed
        self.response = response

    def insert(self, **kwargs):
        return FakeRequest(self.response, self.observed, "insert", **kwargs)

    def delete(self, **kwargs):
        return FakeRequest({}, self.observed, "delete", **kwargs)


class FakeCalendarService:
    def __init__(self, availability, event):
        self.observed = []
        self.availability = availability
        self.event = event

    def freebusy(self):
        return FakeFreeBusy(self.observed, self.availability)

    def events(self):
        return FakeEvents(self.observed, self.event)


@pytest.fixture
def configured_calendar(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "google_client_id", "client-id")
    monkeypatch.setattr(settings, "google_client_secret", "client-secret")
    monkeypatch.setattr(settings, "google_refresh_token", "refresh-token")
    monkeypatch.setattr(settings, "google_calendar_id", "primary")
    monkeypatch.setattr(settings, "calendar_timezone", "Asia/Kathmandu")
    monkeypatch.setattr(settings, "interview_slot_minutes", 30)


def test_available_slot_creates_event_with_configured_timezone(
    monkeypatch,
    configured_calendar,
):
    service = FakeCalendarService(
        {"calendars": {"primary": {"busy": []}}},
        {
            "id": "event-1",
            "htmlLink": "https://calendar.google.com/event?eid=event-1",
        },
    )
    monkeypatch.setattr(calendar_service, "_get_calendar_service", lambda: service)

    result = calendar_service.create_interview_event(
        name="Alex Candidate",
        email="alex@example.com",
        interview_date="2099-12-01",
        interview_time="14:30",
    )

    assert result is not None
    assert result.event_id == "event-1"
    assert result.html_link == "https://calendar.google.com/event?eid=event-1"
    assert len(service.observed) == 2
    freebusy_body = service.observed[0][1]["body"]
    assert freebusy_body["timeZone"] == "Asia/Kathmandu"
    assert freebusy_body["items"] == [{"id": "primary"}]
    insert = service.observed[1][1]
    assert insert["sendUpdates"] == "none"
    assert insert["body"]["start"]["timeZone"] == "Asia/Kathmandu"
    assert insert["body"]["start"]["dateTime"].endswith("+05:45")
    start = datetime.fromisoformat(insert["body"]["start"]["dateTime"])
    end = datetime.fromisoformat(insert["body"]["end"]["dateTime"])
    assert (end - start).total_seconds() == 30 * 60


def test_busy_slot_does_not_create_event(monkeypatch, configured_calendar):
    service = FakeCalendarService(
        {
            "calendars": {
                "primary": {
                    "busy": [
                        {
                            "start": "2099-12-01T14:30:00+05:45",
                            "end": "2099-12-01T15:00:00+05:45",
                        }
                    ]
                }
            }
        },
        {},
    )
    monkeypatch.setattr(calendar_service, "_get_calendar_service", lambda: service)

    result = calendar_service.create_interview_event(
        name="Alex Candidate",
        email="alex@example.com",
        interview_date="2099-12-01",
        interview_time="14:30",
    )

    assert result is None
    assert [operation for operation, _ in service.observed] == ["freebusy"]


def test_missing_google_credentials_raise_actionable_error(
    monkeypatch,
    configured_calendar,
):
    monkeypatch.setattr(get_settings(), "google_client_secret", "")
    calendar_service._get_calendar_service.cache_clear()
    monkeypatch.setattr(calendar_service, "build", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="GOOGLE_CLIENT_SECRET"):
        calendar_service._get_calendar_service()


def test_past_slot_is_rejected(monkeypatch, configured_calendar):
    class UnexpectedCalendarCall:
        def __call__(self):
            pytest.fail("calendar must not be called for a past slot")

    monkeypatch.setattr(
        calendar_service,
        "_get_calendar_service",
        UnexpectedCalendarCall(),
    )
    past_date = datetime.now(ZoneInfo("Asia/Kathmandu")).date().replace(year=2000)

    with pytest.raises(ValueError, match="must be in the future"):
        calendar_service.create_interview_event(
            name="Alex Candidate",
            email="alex@example.com",
            interview_date=past_date.isoformat(),
            interview_time="14:30",
        )

