"""
Script to enrich user registration data and save to CSV.
Includes: registration data, user profile, org profile, profile management, attendee pass, event registrations.
Excludes: connections, recommendations, user settings, context builder.
"""

import asyncio
import csv
from datetime import datetime
from typing import Any, Dict, List

import httpx
from bson import ObjectId

from src.config import settings
from src.database import close_mongo_connection, connect_to_mongo, get_database
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/user_enrichment_csv.log")


class UserDataCSVEnricher:
    def __init__(self):
        self.db = None

    async def setup(self):
        """Initialize database connection"""
        await connect_to_mongo()
        self.db = get_database()
        logger.info("[SETUP] Database connection established")

    async def fetch_registrations_from_api(
        self, event_id: str, bearer_token: str
    ) -> List[Dict]:
        """Fetch user registrations from the API"""
        api_url = f"https://tngss.startuptn.in/event-service/v2/event/user-registrations/list_new?event_id={event_id}"

        logger.info(f"[API] Fetching registrations from: {api_url}")

        headers = {"Authorization": f"Bearer {bearer_token}"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(api_url, headers=headers)
            response.raise_for_status()
            data = response.json()

            registrations = data.get("data", [])
            logger.info(f"[API] Fetched {len(registrations)} registrations")

            return registrations

    async def get_user_profile(self, email: str) -> Dict:
        """Get user profile data"""
        user = await self.db["login_info"].find_one({"email_id": email})
        if not user:
            return {}

        profile = await self.db[settings.USERS_PROFILE_COLLECTION].find_one(
            {"user_id": user["_id"]}
        )
        return profile or {}

    async def get_organisation_profile(self, email: str) -> Dict:
        """Get organisation profile data"""
        user = await self.db["login_info"].find_one({"email_id": email})
        if not user:
            return {}

        org_profile = await self.db[settings.ORGANISATION_PROFILE_COLLECTION].find_one(
            {"user_id": user["_id"]}
        )
        return org_profile or {}

    async def get_profile_management(self, email: str) -> Dict:
        """Get profile management data"""
        user = await self.db["login_info"].find_one({"email_id": email})
        if not user:
            return {}

        profile = await self.db["profile_management"].find_one({"user_id": user["_id"]})
        return profile or {}

    async def get_attendee_pass_info(self, email: str) -> Dict:
        """Get attendee pass information"""
        pass_info = await self.db["attendee-passes"].find_one({"email": email})
        return pass_info or {}

    async def get_event_registrations_count(self, email: str) -> int:
        """Get count of other event registrations"""
        user = await self.db["login_info"].find_one({"email_id": email})
        if not user:
            return 0

        count = await self.db["event_registrations"].count_documents(
            {"user_id": user["_id"], "is_deleted": False}
        )
        return count

    async def enrich_single_user(self, registration_data: Dict) -> Dict:
        """Enrich a single user's data"""
        email = registration_data.get("email_id")

        if not email:
            logger.warning(
                f"[SKIP] No email found for registration {registration_data.get('_id')}"
            )
            return self._flatten_for_csv(registration_data, {}, {}, {}, {}, 0)

        logger.info(f"[ENRICH] Processing {email}")

        # Gather data in parallel
        user_profile, org_profile, profile_mgmt, attendee_pass, event_count = (
            await asyncio.gather(
                self.get_user_profile(email),
                self.get_organisation_profile(email),
                self.get_profile_management(email),
                self.get_attendee_pass_info(email),
                self.get_event_registrations_count(email),
                return_exceptions=True,
            )
        )

        # Handle exceptions
        user_profile = user_profile if not isinstance(user_profile, Exception) else {}
        org_profile = org_profile if not isinstance(org_profile, Exception) else {}
        profile_mgmt = profile_mgmt if not isinstance(profile_mgmt, Exception) else {}
        attendee_pass = (
            attendee_pass if not isinstance(attendee_pass, Exception) else {}
        )
        event_count = event_count if not isinstance(event_count, Exception) else 0

        return self._flatten_for_csv(
            registration_data,
            user_profile,
            org_profile,
            profile_mgmt,
            attendee_pass,
            event_count,
        )

    def _flatten_for_csv(
        self,
        registration: Dict,
        user_profile: Dict,
        org_profile: Dict,
        profile_mgmt: Dict,
        attendee_pass: Dict,
        event_count: int,
    ) -> Dict:
        """Flatten nested data into a single row for CSV"""

        def safe_str(val):
            """Convert value to string, handling ObjectId and None"""
            if val is None:
                return ""
            if isinstance(val, ObjectId):
                return str(val)
            if isinstance(val, (list, dict)):
                return str(val)
            return str(val)

        # Start with registration data
        row = {
            "registration_id": safe_str(registration.get("_id")),
            "email": safe_str(registration.get("email_id")),
            "name": safe_str(registration.get("name")),
            "phone": safe_str(registration.get("phone_number")),
            "gender": safe_str(registration.get("gender")),
            "designation": safe_str(registration.get("designation")),
            "organization": safe_str(registration.get("organization_name")),
            "ticket_type": safe_str(registration.get("ticket")),
            "sector": safe_str(registration.get("sector")),
            "registration_status": safe_str(registration.get("registration_status")),
            "registration_city": safe_str(registration.get("registration_city")),
            "registration_state": safe_str(registration.get("registration_state")),
            "registration_country": safe_str(registration.get("registration_country")),
            "profile_type": safe_str(registration.get("profile_type")),
            "organisation_type": safe_str(registration.get("organisation_type")),
            "why_attending": safe_str(registration.get("why_attending")),
            "website": safe_str(registration.get("website")),
            "checked_in": safe_str(registration.get("checked_in")),
        }

        # Add user profile data
        row.update(
            {
                "user_profile_exists": "Yes" if user_profile else "No",
                "user_profile_linkedin": safe_str(user_profile.get("linkedin_url")),
                "user_profile_twitter": safe_str(user_profile.get("twitter_url")),
                "user_profile_github": safe_str(user_profile.get("github_url")),
            }
        )

        # Add org profile data
        row.update(
            {
                "org_profile_exists": "Yes" if org_profile else "No",
                "org_company_name": safe_str(org_profile.get("company_name")),
                "org_industry": safe_str(org_profile.get("industry")),
                "org_website": safe_str(org_profile.get("website")),
                "org_employee_count": safe_str(org_profile.get("employee_count")),
            }
        )

        # Add profile management data
        row.update(
            {
                "profile_mgmt_exists": "Yes" if profile_mgmt else "No",
                "profile_mgmt_bio": safe_str(profile_mgmt.get("bio")),
                "profile_mgmt_designation": safe_str(profile_mgmt.get("designation")),
                "profile_mgmt_org": safe_str(profile_mgmt.get("organization_name")),
                "profile_mgmt_focused_sector": safe_str(
                    profile_mgmt.get("focused_sector")
                ),
            }
        )

        # Add attendee pass data
        row.update(
            {
                "attendee_pass_exists": "Yes" if attendee_pass else "No",
                "pass_type": safe_str(attendee_pass.get("pass_type")),
                "pass_type_id": safe_str(attendee_pass.get("pass_type_id")),
                "pass_organisation_type": safe_str(
                    attendee_pass.get("organisation_type")
                ),
                "pass_sector_interested": safe_str(
                    attendee_pass.get("sector_interested")
                ),
            }
        )

        # Add event count
        row.update({"total_event_registrations": safe_str(event_count)})

        return row

    async def enrich_and_save_csv(
        self, event_id: str, bearer_token: str, output_filename: str = None
    ):
        """Enrich all users and save to CSV"""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"enriched_users_{timestamp}.csv"

        # Fetch registrations
        registrations = await self.fetch_registrations_from_api(event_id, bearer_token)

        logger.info(f"[ENRICH] Starting enrichment for {len(registrations)} users")

        # Enrich users
        enriched_rows = []
        for i, registration in enumerate(registrations, 1):
            if i % 10 == 0:
                logger.info(f"[PROGRESS] {i}/{len(registrations)}")

            try:
                row = await self.enrich_single_user(registration)
                enriched_rows.append(row)
            except Exception as e:
                logger.error(
                    f"[ERROR] Failed to enrich {registration.get('email_id')}: {e}"
                )

        # Write to CSV
        if enriched_rows:
            fieldnames = enriched_rows[0].keys()

            with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(enriched_rows)

            logger.info(f"[SAVED] Data saved to {output_filename}")
            print(
                f"\n✓ Successfully saved {len(enriched_rows)} enriched users to {output_filename}"
            )
        else:
            logger.warning("[WARNING] No data to save")
            print("\n✗ No data to save")

        return output_filename, len(enriched_rows)


async def main():
    """Main execution function"""
    EVENT_ID = "68e25704f93e895880818cbe"
    BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNjhiMzMxNzMxZDRlYmVmNzliOGM3ODZkIiwiaWF0IjoxNzU2NTc1Mzc1LCJleHAiOjE3NjUyMTUzNzV9.n8s-3SZfWQJy-lMkWX6jLWSC1cPjnlLWBjjJMc3vxBY"

    enricher = UserDataCSVEnricher()

    try:
        await enricher.setup()

        print(f"\n{'='*70}")
        print(f"ENRICHING USER REGISTRATIONS")
        print(f"Event ID: {EVENT_ID}")
        print(f"{'='*70}\n")

        filename, count = await enricher.enrich_and_save_csv(EVENT_ID, BEARER_TOKEN)

        print(f"\n{'='*70}")
        print(f"COMPLETE")
        print(f"{'='*70}")
        print(f"Total users: {count}")
        print(f"Output file: {filename}")
        print(f"{'='*70}\n")

    finally:
        await close_mongo_connection()


if __name__ == "__main__":
    asyncio.run(main())
