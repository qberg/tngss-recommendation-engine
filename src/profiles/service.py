"""
Core profile service for processing text of users
"""

import asyncio
from typing import Dict, List

from src.embeddings.utils import num_tokens_from_string
from src.profiles.schemas import (ContextBuilder, LoginInfo,
                                  OrganisationProfile, UserData, UserProfile)
from src.utils.common import key_to_label
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/profile_service.log")


class ProfileService:
    """Service for creating text representations from user profiles."""

    def __init__(self):
        pass

    def get_pronouns(self, gender: str) -> Dict[str, str]:
        """Returns pronouns based on gender input"""
        gender = gender.lower().strip()

        pronoun_map = {
            "male": {"subject": "he", "object": "him", "possessive": "his"},
            "man": {"subject": "he", "object": "him", "possessive": "his"},
            "female": {"subject": "she", "object": "her", "possessive": "her"},
            "woman": {"subject": "she", "object": "her", "possessive": "her"},
            "transgender": {"subject": "they", "object": "them", "possessive": "their"},
            "other": {"subject": "they", "object": "them", "possessive": "their"},
        }

        return pronoun_map.get(
            gender, {"subject": "they", "object": "them", "possessive": "their"}
        )

    def create_personal_text(self, user_data: UserData) -> str:
        """Create personal text for embedding from user data"""
        try:
            login = user_data.login
            profile = user_data.profile
            org = user_data.org

            ##############################
            # Perosnal Info Extraction ##
            ##############################
            name = ""
            if login:
                name = f"{login.first_name} {login.last_name}".strip()

            designation = profile.designation if profile else ""
            bio = profile.bio if profile else ""
            location = ""
            gender = ""
            gender_pronoun = {}
            if profile:
                gender = profile.gender
                gender_pronoun = self.get_pronouns(gender)

                city = profile.city or ""
                country = profile.country or ""
                if city and country:
                    location = f"{city}, {country}"
                elif city:
                    location = city
                elif country:
                    location = country

            ##############################
            # Compnay Info Extraction   ##
            ##############################

            org_name = org.organisation_name if org else ""
            org_type = org.profile_type if org else ""
            sector = org.sector if org else ""

            text_parts = []

            if name and designation and org_name:
                text_parts.append(
                    f"{gender_pronoun.get('subject', 'They').title()} works as {designation} at {org_name}"
                )
            elif name and designation:
                text_parts.append(f"{name} works as {designation}")

            if org_type and sector:
                clean_sector = key_to_label(sector)
                clean_org_type = key_to_label(org_type)
                text_parts.append(
                    f"Their organization focusses in {clean_sector} sector as {clean_org_type}"
                )

            if bio:
                text_parts.append(f"Background: {bio}")

            # if location:
            # text_parts.append(f"Based out of {location}")

            if gender_pronoun:
                text_parts.append(
                    f"{gender_pronoun['subject'].title()} is currently exploring opportunities in {sector}."
                )

            personal_text = ". ".join(text_parts)

            num_tokens = num_tokens_from_string(personal_text)

            logger.info(
                f"[SUCCESS][ProfileService] Personal text created [{len(personal_text)} chracters] [{num_tokens} tokens]"
            )

            return personal_text

        except Exception as e:
            logger.error(
                f"[FAILED][ProfileService] Personal text creation for embedding failed: {e}"
            )
            return ""

    def create_org_text(self, user_data: UserData) -> str:
        """Create organizational text for embedding"""
        try:
            org = user_data.org
            if not org:
                return "No organization information provided."

            text_parts = []

            # 1. Organisation intro
            intro = []
            if org.organisation_name:
                intro.append(org.organisation_name)
            if org.profile_type:
                intro.append(key_to_label(org.profile_type))
            if org.sub_type:
                intro.append(key_to_label(org.sub_type))
            if org.sector:
                intro.append(f"in the {key_to_label(org.sector)} sector")

            if intro:
                location = []
                if org.city:
                    location.append(org.city)
                if org.country:
                    location.append(org.country)
                loc_str = f" based in {', '.join(location)}" if location else ""
                # text_parts.append(" ".join(intro) + loc_str + ".")

            # 2. About / narrative description
            if org.about_organisation:
                text_parts.append(
                    f"{org.organisation_name} focuses on {org.about_organisation}."
                )

            # 3. Products / business model
            if org.product_offering_details:
                text_parts.append(
                    f"The company offers {', '.join(org.product_offering_details)}."
                )
            if org.bussiness_model:
                text_parts.append(
                    f"It operates using {', '.join(org.bussiness_model)} business models."
                )

            # 4. Startup-specific info
            startup_bits = []
            if org.stage:
                startup_bits.append(f"is at {org.stage} stage")
            if org.team_size:
                startup_bits.append(f"with a team of {org.team_size} people")
            if org.fund_rise_till_date:
                startup_bits.append(f"having raised {org.fund_rise_till_date} so far")
            if org.revenue:
                startup_bits.append(f"and reporting revenue of {org.revenue},")
            if org.ticket_size:
                startup_bits.append(f"with tickets size around {org.ticket_size}.")
            if startup_bits:
                text_parts.append(f"The organization {' '.join(startup_bits)}.")

            # 5. Investor-specific info
            investor_bits = []
            if org.funding_stage:
                investor_bits.append(f"invests at {org.funding_stage} stage")
            if org.investment_instrument:
                investor_bits.append(
                    f"using instruments such as {org.investment_instrument}"
                )
            if org.revenue_stage:
                investor_bits.append(
                    f"and prefers companies at {org.revenue_stage} revenue stage"
                )
            if org.ticket_size:
                investor_bits.append(f"with ticket sizes around {org.ticket_size}")
            if investor_bits:
                text_parts.append(
                    f"As an investor, the organization {' '.join(investor_bits)}."
                )

            org_text = " ".join(text_parts)
            num_tokens = num_tokens_from_string(org_text)

            logger.info(
                f"[SUCCESS][ProfileService] Org text created [{len(org_text)} characters] [{num_tokens} tokens]"
            )
            return org_text

        except Exception as e:
            logger.error(f"[FAILED] Org text creation for embedding failed: {e}")
            return "No organization information provided."

    def create_intent_text(self, user_data: UserData) -> str:
        """Create intent text for embedding from user data"""
        try:
            org = user_data.org
            context = user_data.context

            text_parts = []

            if org and org.looking_for:
                looking_for_clean = [key_to_label(item) for item in org.looking_for]
                text_parts.append(
                    f"They are currently seeking {', '.join(looking_for_clean)}."
                )

            if context and context.looking_to_connect:
                connect_clean = [
                    key_to_label(item) for item in context.looking_to_connect
                ]
                text_parts.append(
                    f"They want to connect with {', '.join(connect_clean)}."
                )

            if context and context.looking_to_meet:
                meet_clean = [key_to_label(item) for item in context.looking_to_meet]
                text_parts.append(
                    f"They are particularly interested in meeting {', '.join(meet_clean)}."
                )

            if context and context.sector:
                sector_clean = [key_to_label(item) for item in context.sector]
                text_parts.append(
                    f"Their main sector interests include {', '.join(sector_clean)}."
                )

            if org and org.product_offering_details:
                text_parts.append(
                    f"They currently offer {', '.join(org.product_offering_details)}."
                )

            if not text_parts:
                return "No explicit intent provided."

            intent_text = " ".join(text_parts)

            num_tokens = num_tokens_from_string(intent_text)

            logger.info(
                f"[SUCCESS][ProfileService] Intent text created [{len(intent_text)} characters] [{num_tokens} tokens]"
            )
            return intent_text

        except Exception as e:
            logger.error(f"[FAILED][ProfileService] Failed to create intent text: {e}")
            return "No explicit intent provided."

    def create_all_texts(self, user_data: UserData) -> List[str]:
        """Create all three text types for a user."""

        personal = self.create_personal_text(user_data)
        org = self.create_org_text(user_data)
        intent = self.create_intent_text(user_data)

        texts = [
            personal if personal.strip() else "User profile incomplete.",
            org if org.strip() else "No organization information provided.",
            intent if intent.strip() else "No explicit intent provided.",
        ]

        return texts


async def main():
    """Test the ProfileService."""
    try:
        logger.info("[***] Testing ProfileService")

        from bson import ObjectId

        from src.database import connect_to_mongo, get_database

        await connect_to_mongo()
        db = get_database()

        user = await db["fake_login_info"].find_one({})
        if not user:
            logger.error("[FAILED] No users found in database")
            return

        # user_id = str(user["_id"])
        user_id = str("68d2a8d7d84108aa7b141514")
        logger.info(f"[***] Testing with user: {user_id}")

        user_obj_id = ObjectId(user_id)
        org_id = user.get("organisation_profile_id")

        profile = await db["fake_profile_management"].find_one({"user_id": user_obj_id})
        org = (
            await db["fake_organisation_profile_management"].find_one({"_id": org_id})
            if org_id
            else None
        )
        context = await db["fake_context_builder_management"].find_one(
            {"user_id": user_obj_id}
        )

        user_data = UserData(
            login=LoginInfo(**user) if user else None,
            profile=UserProfile(**profile) if profile else None,
            org=OrganisationProfile(**org) if org else None,
            context=ContextBuilder(**context) if context else None,
        )

        # Test the ProfileService
        profile_service = ProfileService()

        # Test individual text creation
        personal_text = profile_service.create_personal_text(user_data)
        logger.info(f"[SUCCESS] Personal text: {personal_text[:-1]}...")

        org_text = profile_service.create_org_text(user_data)
        logger.info(f"[SUCCESS] Org text: {org_text[:-1]}...")

        intent_text = profile_service.create_intent_text(user_data)
        logger.info(f"[SUCCESS] Intent text: {intent_text[:-1]}...")

        # Test all texts together
        all_texts = profile_service.create_all_texts(user_data)
        logger.info(f"[SUCCESS] Created {len(all_texts)} texts")

        logger.info("[SUCCESS] All ProfileService tests passed!")

    except Exception as e:
        logger.error(f"[FAILED] ProfileService test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
