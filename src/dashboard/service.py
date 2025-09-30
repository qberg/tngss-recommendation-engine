from typing import Any, Dict

from src.events.client import EventsClient
from src.events.schemas import Event
from src.recommendations.service import RecommendationService


class DashboardService:
    def __init__(self, recommendation_service: RecommendationService):
        self.rec_service = recommendation_service

    async def get_user_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """Get all data needed for user dashboard"""

        # Use UserEmbeddingService to get user data
        user_data = await self.rec_service.user_service.get_user_data(user_id)

        # Access profile_service through user_service
        user_texts = self.rec_service.user_service.profile_service.create_all_texts(
            user_data
        )

        # Generate scores
        event_scores = await self.rec_service.generate_event_scores_for_user_with_cache(
            user_id, max_events=-1
        )

        # Fetch event details
        events_client = EventsClient()
        event_details = {}

        for score in event_scores:
            event_id = score["target_id"]
            try:
                event_data = await events_client.get_event(event_id)
                if not event_data:
                    event_details[event_id] = {
                        "title": "Event not found",
                        "format": "Unknown",
                        "about": "Event data unavailable",
                    }
                    continue

                event = Event.from_api_response(event_data)
                event_details[event_id] = {
                    "title": event.title,
                    "format": event.format,
                    "about": event.about,
                }
            except Exception as e:
                print(f"Error loading event {event_id}: {e}")
                event_details[event_id] = {
                    "title": f"Error loading event {event_id}",
                    "format": "Unknown",
                    "about": f"Error: {str(e)}",
                }

        await events_client.close()

        return {
            "user_id": user_id,
            "user_data": {
                "name": (
                    f"{user_data.login.first_name} {user_data.login.last_name}"
                    if user_data.login
                    else "Unknown"
                ),
                "designation": (
                    user_data.profile.designation if user_data.profile else "Unknown"
                ),
                "organization": (
                    user_data.org.organisation_name if user_data.org else "Unknown"
                ),
            },
            "user_texts": {
                "personal": user_texts[0],
                "organizational": user_texts[1],
                "intent": user_texts[2],
            },
            "event_scores": event_scores,
            "event_details": event_details,
        }
