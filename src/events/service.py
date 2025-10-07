"""
Core Event service for processing text of events
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List

import pytz

from src.embeddings.utils import num_tokens_from_string
from src.events.client import EventsClient
from src.events.schemas import Event
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/events_service.log")


class EventService:
    """Format event data for embedding and display"""

    @staticmethod
    def format_datetime(iso_string: str) -> str:
        """Convert ISO datetime to readable format"""
        try:
            dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
            ist_tz = pytz.timezone("Asia/Kolkata")
            ist_dt = dt.astimezone(ist_tz)
            return ist_dt.strftime("%B %d, %Y at %I:%M %p IST")

        except Exception as e:
            logger.error(
                f"[FAILED] Conversion of iso_string for event failed: {e}, returning the iso_string as is"
            )
            return iso_string

    @staticmethod
    def extract_event_info(event: Event) -> str:
        """Extract key info from raw event data"""
        code_start_time = time.time()
        segments = []

        event_type = event.format or "session"
        event_category = (
            event.main_or_partner.replace("_", " ").title()
            if event.main_or_partner
            else "summit"
        )
        segments.append(
            f"{event.title} is a {event_type} at the Tamil Nadu Global Startup Summit 2025, categorized as a {event_category}"
        )

        if event.schedule:
            start_time = EventService.format_datetime(event.schedule.from_date)
            end_time = EventService.format_datetime(event.schedule.to_date)
            segments.append(f"The event is scheduled from {start_time} to {end_time}")

        if event.zone:
            segments.append(
                f"This session takes place at the {event.zone} within the summit venue."
            )

        if event.about:
            about_text = event.about.strip()
            if not about_text.endswith("."):
                about_text += "."
            segments.append(
                f"The session content focuses on the following: {about_text}"
            )

        if event.speakers:
            speaker_details = []
            for speaker in event.speakers[:5]:
                speaker_info = speaker.name
                if speaker.designation and speaker.organization:
                    speaker_info += f", who serves as {speaker.designation} at {speaker.organization}"
                elif speaker.organization:
                    speaker_info += f" from {speaker.organization}"
                elif speaker.designation:
                    speaker_info += f", {speaker.designation}"

                speaker_info += f". Summary: {speaker.summary}."

                speaker_details.append(speaker_info)

            if len(speaker_details) == 1:
                segments.append(f"The featured speaker is {speaker_details[0]}")
            elif len(speaker_details) > 1:
                if len(speaker_details) == 2:
                    segments.append(
                        f"The featured speakers are {speaker_details[0]} and {speaker_details[1]}"
                    )
                else:
                    last_speaker = speaker_details.pop()
                    segments.append(
                        f"The featured speakers include {', '.join(speaker_details)}, and {last_speaker}"
                    )

        if event.tags:
            tag_names = [tag.name.lower() for tag in event.tags]
            if len(tag_names) == 1:
                segments.append(
                    f"This event covers {tag_names[0]} related topics and innovations"
                )
            else:
                segments.append(
                    f"The session addresses themes including {', '.join(tag_names)}"
                )

        if event.agenda:
            agenda_descriptions = []
            for agenda_item in event.agenda[:3]:
                if agenda_item.description and agenda_item.time:
                    agenda_descriptions.append(
                        f"{agenda_item.description} at {agenda_item.time}"
                    )
                elif agenda_item.description:
                    agenda_descriptions.append(agenda_item.description)

            if agenda_descriptions:
                segments.append(
                    f"The agenda includes: {'. '.join(agenda_descriptions)}"
                )

        if event.current_registerations > 0:
            segments.append(
                f"This session has attracted {event.current_registerations} registered participants, indicating community interest and engagement"
            )

        event_embedding_text = ". ".join(segments) + "."
        num_tokens = num_tokens_from_string(event_embedding_text)

        code_end_time = time.time()
        execution_time = code_end_time - code_start_time

        logger.info(
            f"[COMPLETE] Time taken: {execution_time:.2f} seconds for creating embedding texts"
        )

        logger.info(
            f"[SUCCESS] Event embedding text created [{len(event_embedding_text)} chracters] [{num_tokens} tokens]"
        )

        return event_embedding_text

    @staticmethod
    def format_event_for_embedding(event_data: dict) -> str:
        """Extract key info from raw event dict"""
        event = Event.from_api_response(event_data)
        return EventService.extract_event_info(event)

    @staticmethod
    def format_events_for_embedding(events: List[Dict[str, Any]]) -> List[str]:
        """Format events list into markdown text"""
        if not events:
            logger.info("No public events are currently available.")
            return []

        start_time = time.time()
        event_objects = []
        for event_data in events:
            try:
                event = Event.from_api_response(event_data)
                event_objects.append(event)
            except Exception as e:
                logger.error(f"Could not parse event: {e} {event_data['id']}")
                continue

        formatted_events = []
        for event in event_objects:
            event_text = EventService.extract_event_info(event)
            formatted_events.append(event_text)

        end_time = time.time()
        exeution_time = end_time - start_time

        logger.info(
            f"[COMPLETE] Total events: {len(events)} for text generation | Time taken: {exeution_time:.2f} seconds"
        )
        logger.info(f"[STATS] Speed: {len(events)/exeution_time:.1f} events/second")

        return formatted_events


async def main():
    """Test the event embedding text generation"""
    client = EventsClient()
    try:
        logger.info("[***] Testing EventService with real API data")
        batch_size = 100

        logger.info(f"[***] Fetching {batch_size} events from api for testing")
        events_data = await client.get_all_public_events(batch_size=batch_size)

        if not events_data:
            logger.info("[!!!] No events found")

        logger.info(
            f"[SUCCESS] Found {len(events_data)} events. Testing the text generation"
        )

        print("#" * 50)
        print("# Test 1")
        print("#" * 50)
        first_event_data = events_data[100]
        logger.info(f"Testing with event: {first_event_data.get('title', 'Unknown')}")
        event = Event.from_api_response(first_event_data)

        formatted_text = EventService.extract_event_info(event)

        print("+" * 50)
        print("FORMATTED OUTPUT:")
        print("+" * 50)
        print(formatted_text)
        print("+" * 50)

        logger.info(
            "[SUCCESS] Test for single event embedding text format generation successful"
        )

        # print("+" * 50)
        # print("+ Test 2")
        # print("+" * 50)
        # events_text = EventService.format_events_for_embedding(events_data)

        # logger.info(f"Created embedding ready texts for {len(events_text)} events")

        # for event_text in events_text[:10]:
        #    print("+" * 50)
        #    print("Event text preview")
        #    print(f"{event_text[:-1]}...")
        #    print("+" * 50)

        # logger.info(
        #    "[SUCCESS] Test for multiple events embedding text format generation successful"
        # )

    except Exception as e:
        logger.error(f"Test failed: {e}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
