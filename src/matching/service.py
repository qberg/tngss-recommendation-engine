"""
Core recommendation service business logic for multi-vector matching system.
"""

import asyncio
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
from bson import ObjectId
from openai import OpenAI
from pymongo.asynchronous.database import AsyncDatabase

sys.path.append(str(Path(__file__).parent.parent))
from database import connect_to_mongo, get_database
from src.config import settings
from src.matching.constants import VectorType
from src.matching.schemas import (ContextBuilder, LoginInfo,
                                  OrganisationProfile, UserData, UserProfile)
from src.matching.utils import (cosine_similarity, num_tokens_from_string,
                                truncate_text)
from src.utils.common import key_to_label
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/recommendation_service.log")


class RecommendationService:
    """Core service for generating recommendations using multi-vector approach."""

    def __init__(self, database: AsyncDatabase):
        self.db = database
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

        logger.info("[SUCCESS] Recommendation service initialized")

    def test_openai_connection(self):
        """Test OpenAI API connection"""
        try:
            logger.info("[...] Testing OpenAI Connection")

            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small", input=["Test Connection"]
            )

            if response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                logger.info("[SUCCESS] OpenAI connection sucessfull")
                logger.info(f"[INFO] Embedding Dimensions: {len(embedding)}")
            else:
                logger.error("[FAILED] OpenAI API returned empty response")

        except Exception as e:
            raise e

    async def check_database_collecitons(self):
        """Check database collections and returns counts"""
        try:
            logger.info("[***] Checking database collections")

            collections = {
                "login_info": settings.LOGIN_COLLECTION,
                "user_profiles": settings.USERS_PROFILE_COLLECTION,
                "org_profiles": settings.ORGANISATION_PROFILE_COLLECTION,
                "context_builder": settings.CONTEXT_BUILDER_COLLECTION,
                "events": settings.EVENTS_COLLECTION,
            }

            for name, collection in collections.items():
                count = await self.db[collection].count_documents({})
                logger.info(f"[INFO] {name}: {count} documents")

            logger.info("[SUCCESS] Database connection checked")

        except Exception as e:
            logger.error(f"[FAILED] Database check failed: {e}")

    async def get_raw_user_data(self, user_id: str) -> Dict[str, Any]:
        """Fetch user data across four collections"""
        try:
            logger.info(f"[***] Fetching data for user: {user_id}")
            start_time = time.perf_counter()

            user_obj_id = ObjectId(user_id)

            user = await self.db[settings.LOGIN_COLLECTION].find_one(
                {"_id": user_obj_id}
            )
            if not user:
                raise ValueError(f"User not found: {user_id}")

            org_id = user.get("organisation_profile_id")

            profile = await self.db[settings.USERS_PROFILE_COLLECTION].find_one(
                {"user_id": user_obj_id}
            )
            org = (
                await self.db[settings.ORGANISATION_PROFILE_COLLECTION].find_one(
                    {"_id": org_id}
                )
                if org_id
                else {}
            )
            context = await self.db[settings.CONTEXT_BUILDER_COLLECTION].find_one(
                {"user_id": user_obj_id}
            )

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000

            logger.info(
                f"[SUCCESS] User data fetched in {execution_time:.2f}ms - Profile: {'Yes' if profile else 'No'}, Org: {'Yes' if org else 'No'}, Context: {'Yes' if context else 'No'}"
            )

            return {"login": user, "profile": profile, "org": org, "context": context}

        except Exception as e:
            logger.error(
                f"[FAILED] Fetching user data across multiple collections failed: {e}"
            )
            raise e

    async def get_user_data(self, user_id: str) -> UserData:
        try:
            raw_data = await self.get_raw_user_data(user_id)

            return UserData(
                login=LoginInfo(**raw_data["login"]) if raw_data["login"] else None,
                profile=(
                    UserProfile(**raw_data["profile"]) if raw_data["profile"] else None
                ),
                org=OrganisationProfile(**raw_data["org"]) if raw_data["org"] else None,
                context=(
                    ContextBuilder(**raw_data["context"])
                    if raw_data["context"]
                    else None
                ),
            )

        except Exception as e:
            logger.error(f"Failed to create UserData model: {e}")
            raise e

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
            if profile:
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
                text_parts.append(f"{name} is {designation} at {org_name}")
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

            if location:
                text_parts.append(f"Based out of {location}")

            personal_text = ". ".join(text_parts)

            num_tokens = num_tokens_from_string(personal_text)

            logger.info(
                f"[SUCCESS] Personal text created [{len(personal_text)} chracters] [{num_tokens} tokens]"
            )

            return personal_text

        except Exception as e:
            logger.error(f"[FAILED] Personal text creation for embedding failed: {e}")
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
                text_parts.append(" ".join(intro) + loc_str + ".")

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
                startup_bits.append(f"and reporting revenue of {org.revenue}")
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
                f"[SUCCESS] Org text created [{len(org_text)} characters] [{num_tokens} tokens]"
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
                f"[SUCCESS] Intent text created [{len(intent_text)} characters] [{num_tokens} tokens]"
            )
            return intent_text

        except Exception as e:
            logger.error(f"[FAILED] Failed to create intent text: {e}")
            return "No explicit intent provided."

    async def generate_user_embeddings(
        self, user_data: UserData
    ) -> Dict[VectorType, np.ndarray]:
        """Generate all 3 embeddings for a user (personal, org, intent)"""
        try:
            logger.info("[***] Generating all three embeddings of a user")

            personal_text = truncate_text(self.create_personal_text(user_data))
            org_text = truncate_text(self.create_org_text(user_data))
            intent_text = truncate_text(self.create_intent_text(user_data))

            texts = [personal_text, org_text, intent_text]

            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small", input=texts
            )

            return {
                "personal": np.array(response.data[0].embedding),
                "org": np.array(response.data[1].embedding),
                "intent": np.array(response.data[2].embedding),
            }

        except Exception as e:
            logger.error(f"[FAILED] Multi user vector emebedding failed: {e}")
            raise e

    def calculate_multi_vector_similarity(
        self,
        user_embeddings: Dict[VectorType, np.ndarray],
        target_embedding: np.ndarray,
        weights: Dict[VectorType, float] = {"personal": 0.3, "org": 0.3, "intent": 0.4},
    ) -> Dict[str, float]:
        """Calculate weighted similrity of vector with multi user vector"""
        try:
            logger.info(
                "[***] Calculating similarity scores for target with multi user vector"
            )

            similarities = {}
            for vector_type, embedding in user_embeddings.items():
                similarities[vector_type] = cosine_similarity(
                    embedding, target_embedding
                )

            final_score = (
                weights["personal"] * similarities["personal"]
                + weights["org"] * similarities["org"]
                + weights["intent"] * similarities["intent"]
            )

            logger.info(
                f"[SUCCESS] THe taget vector has similarity score of {final_score}"
            )

            return {
                "final_score": final_score,
                **similarities,
            }

        except Exception as e:
            logger.error(
                f"[FAILED] Generating similarity score of target vector with multi user vector failed: {e}"
            )
            raise e

    def self_similarity_test(self, user_embeddings: Dict[VectorType, np.ndarray]):
        """Checking sensibility with self"""
        logger.info("[INFO] Self-similarity test:")
        personal_embedding = user_embeddings["personal"]
        similarity_result = self.calculate_multi_vector_similarity(
            user_embeddings, personal_embedding
        )
        logger.info(f"  Final score: {similarity_result['final_score']:.3f}")
        logger.info(f"  Personal: {similarity_result['personal']:.3f}")
        logger.info(f"  Organizational: {similarity_result['org']:.3f}")
        logger.info(f"  Intent: {similarity_result['intent']:.3f}")

    async def test_cross_user_similarity(self) -> None:
        """Test similarity between two different users"""
        try:
            users = (
                await self.db[settings.LOGIN_COLLECTION].find({}).limit(2).to_list(2)
            )

            if len(users) < 2:
                logger.warning("Need at least 2 users for cross-similarity test")
                return

            user1_id = str(users[0]["_id"])
            user2_id = str(users[1]["_id"])

            logger.info(
                f"[...] Testing similarity between user {user1_id[:8]}... and {user2_id[:8]}..."
            )

            user1_data = await self.get_user_data(user1_id)
            user2_data = await self.get_user_data(user2_id)

            user1_embeddings = await self.generate_user_embeddings(user1_data)
            user2_embeddings = await self.generate_user_embeddings(user2_data)

            logger.info("[INFO] Cross-user similarity tests:")

            sim1 = self.calculate_multi_vector_similarity(
                user1_embeddings, user2_embeddings["personal"]
            )
            logger.info(
                f"  User1 multi-vector vs User2 personal: {sim1['final_score']:.3f}"
            )

            sim2 = self.calculate_multi_vector_similarity(
                user1_embeddings, user2_embeddings["intent"]
            )
            logger.info(
                f"  User1 multi-vector vs User2 intent: {sim2['final_score']:.3f}"
            )

            logger.info(
                f"  Breakdown - Personal: {sim1['personal']:.3f}, Org: {sim1['org']:.3f}, Intent: {sim1['intent']:.3f}"
            )

        except Exception as e:
            logger.error(f"[FAILED] Cross-user similarity test failed: {e}")


async def main():
    try:
        logger.info("[...] Connecting to database")
        await connect_to_mongo()
        db = get_database()

        logger.info("[INFO] Initializing RecommendationService")
        service = RecommendationService(db)
        await service.check_database_collecitons()

        await service.test_cross_user_similarity()

    except Exception as e:
        logger.error(f"RecommendationService failed to launch: {e}")


if __name__ == "__main__":
    asyncio.run(main())
