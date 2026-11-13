from dataclasses import dataclass
from datetime import datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from google.auth.exceptions import GoogleAuthError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.core.config import get_settings


CALENDAR_SCOPES = (
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.freebusy",
)


@dataclass(frozen=True)
class CalendarEvent:
    event_id: str
    html_link: str | None
    start: datetime
    end: datetime


@lru_cache
def _get_calendar_service():
    settings = get_settings()
    required_settings = {
        "GOOGLE_CLIENT_ID": settings.google_client_id,
        "GOOGLE_CLIENT_SECRET": settings.google_client_secret,
        "GOOGLE_REFRESH_TOKEN": settings.google_refresh_token,
    }
    missing_settings = [
        name for name, value in required_settings.items() if not value
    ]
    if missing_settings:
        raise RuntimeError(
            "Google Calendar is not configured. Missing: "
            + ", ".join(missing_settings)
        )

    credentials = Credentials(
        token=None,
        refresh_token=settings.google_refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        scopes=CALENDAR_SCOPES,
    )
    try:
        return build("calendar", "v3", credentials=credentials, cache_discovery=False)
    except GoogleAuthError as exc:
        raise RuntimeError("Unable to authenticate with Google Calendar.") from exc


def _calendar_id() -> str:
    calendar_id = get_settings().google_calendar_id.strip()
    if not calendar_id:
        raise RuntimeError("GOOGLE_CALENDAR_ID is not configured.")
    return calendar_id


def _slot_interval(
    interview_date: str,
    interview_time: str,
) -> tuple[datetime, datetime, ZoneInfo]:
    settings = get_settings()
    try:
        timezone = ZoneInfo(settings.calendar_timezone)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(
            f"Unknown calendar timezone: {settings.calendar_timezone}"
        ) from exc

    try:
        start = datetime.combine(
            datetime.strptime(interview_date, "%Y-%m-%d").date(),
            time.fromisoformat(interview_time),
        ).replace(tzinfo=timezone)
    except ValueError as exc:
        raise ValueError("Interview date or time has an invalid format.") from exc

    if start <= datetime.now(timezone):
        raise ValueError("Interview date and time must be in the future.")

    end = start + timedelta(minutes=settings.interview_slot_minutes)
    return start, end, timezone


def create_interview_event(
    *,
    name: str,
    email: str,
    interview_date: str,
    interview_time: str,
) -> CalendarEvent | None:
    settings = get_settings()
    if settings.interview_slot_minutes < 1:
        raise RuntimeError("INTERVIEW_SLOT_MINUTES must be positive.")
    start, end, timezone = _slot_interval(interview_date, interview_time)
    calendar_id = _calendar_id()
    service = _get_calendar_service()

    try:
        availability = (
            service.freebusy()
            .query(
                body={
                    "timeMin": start.isoformat(),
                    "timeMax": end.isoformat(),
                    "timeZone": settings.calendar_timezone,
                    "items": [{"id": calendar_id}],
                }
            )
            .execute()
        )
        calendar = availability.get("calendars", {}).get(calendar_id)
        if not calendar or calendar.get("errors"):
            raise RuntimeError(
                "Google Calendar could not provide availability for the configured calendar."
            )
        if calendar.get("busy"):
            return None

        event = (
            service.events()
            .insert(
                calendarId=calendar_id,
                sendUpdates="none",
                body={
                    "summary": f"Interview - {name}",
                    "description": f"Candidate: {name}\nContact: {email}",
                    "start": {
                        "dateTime": start.isoformat(),
                        "timeZone": settings.calendar_timezone,
                    },
                    "end": {
                        "dateTime": end.isoformat(),
                        "timeZone": settings.calendar_timezone,
                    },
                },
            )
            .execute()
        )
    except HttpError as exc:
        raise RuntimeError("Google Calendar request failed.") from exc
    except GoogleAuthError as exc:
        raise RuntimeError("Google Calendar authentication failed.") from exc

    event_id = event.get("id")
    if not isinstance(event_id, str) or not event_id:
        raise RuntimeError("Google Calendar created an event without an ID.")
    html_link = event.get("htmlLink")
    return CalendarEvent(
        event_id=event_id,
        html_link=html_link if isinstance(html_link, str) else None,
        start=start,
        end=end,
    )


def delete_interview_event(event_id: str) -> None:
    try:
        service = _get_calendar_service()
        service.events().delete(
            calendarId=_calendar_id(),
            eventId=event_id,
        ).execute()
    except HttpError as exc:
        raise RuntimeError("Unable to remove the uncommitted calendar event.") from exc
    except GoogleAuthError as exc:
        raise RuntimeError(
            "Unable to authenticate while removing the uncommitted calendar event."
        ) from exc
