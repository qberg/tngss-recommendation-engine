import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from src.utils.common import key_to_label


class Tag(BaseModel):
    name: str


class Schedule(BaseModel):
    from_date: str
    to_date: str


class Location(BaseModel):
    city: str = "Coimbatore"
    country: str = "India"


class Agenda(BaseModel):
    time: Optional[str] = None
    description: Optional[str] = None


class SpeakerExperience(BaseModel):
    company: Optional[str] = None
    designation: Optional[str] = None
    location: Optional[str] = None


class Speaker(BaseModel):
    name: str
    designation: Optional[str] = None
    organization: Optional[str] = None
    speaker_type: str
    location: Optional[str] = None
    summary: Optional[str] = None


class AlmaMatter(BaseModel):
    college: Optional[str]
    degree: Optional[str]


class Event(BaseModel):
    id: str
    title: str
    main_or_partner: str = "Main Event"
    schedule: Optional[Schedule] = None
    zone: str
    about: str
    format: Optional[str] = None
    agenda: List[Agenda] = []
    speakers: List[Speaker] = []
    tags: List[Tag] = []
    current_registerations: int = 0

    @classmethod
    def from_api_response(cls, event_data: dict) -> "Event":
        """Convert raw API response to Event"""
        speakers = []
        for speaker_data in event_data.get("speakers", []):

            speaker = speaker_data.get("speaker", {})

            if not speaker:
                continue
            speakers.append(
                Speaker(
                    name=speaker.get("name", ""),
                    designation=speaker.get("designation", ""),
                    organization=speaker.get("organization", ""),
                    speaker_type=(
                        speaker.get("speaker_type", {}).get("name")
                        if speaker.get("speaker_type")
                        else "Domestic"
                    ),
                    location=(
                        speaker.get("location", {}).get("country")
                        if speaker.get("location")
                        else "India"
                    ),
                    summary=speaker.get("summary", ""),
                )
            )

        agenda_items = []
        for agenda_data in event_data.get("agenda", []):
            agenda_items.append(
                Agenda(
                    time=agenda_data.get("time"),
                    description=agenda_data.get("description"),
                )
            )

        tags = [Tag(name=tag.get("name", "")) for tag in event_data.get("tags", [])]

        schedule = None
        schedule_data = event_data.get("schedule", {})
        if schedule_data:
            schedule = Schedule(
                from_date=schedule_data.get("from_date", ""),
                to_date=schedule_data.get("to_date", ""),
            )

        return cls(
            id=event_data.get("id", ""),
            title=event_data.get("title", ""),
            main_or_partner=key_to_label(
                event_data.get("main_or_partner", "Main Event")
            ),
            schedule=schedule,
            zone=(
                event_data.get("zone", {}).get("name")
                if event_data.get("zone")
                else "Summit Stage"
            ),
            about=event_data.get("about", ""),
            agenda=agenda_items,
            format=(
                event_data.get("format", {}).get("name")
                if event_data.get("format")
                else "Spotlight"
            ),
            speakers=speakers,
            tags=tags,
            current_registerations=event_data.get("current_registerations", 0),
        )


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    TESTS_DIR = ROOT / "tests"
    sample_event_file = TESTS_DIR / "sample_event.json"
    print(f"{sample_event_file}")

    if sample_event_file.exists():
        with open(sample_event_file, "r") as f:
            sample_event_data = json.load(f)

        event = Event.from_api_response(sample_event_data)
        print("#" * 50)
        print(f"Event: {event.speakers}")
        print("#" * 50)
    else:
        print("[FAILED] Sample event file not found for testing")
