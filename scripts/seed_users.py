"""
Async MongoDB Seeding script for testing AI Recommendation Engine
"""

import argparse
import asyncio
import os
import random
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import bcrypt
from bson import ObjectId
from faker import Faker
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.database import connect_to_mongo, get_database
from src.profiles.constants import (BUSINESS_MODEL, DESIGNATIONS_LABELS,
                                    INVESTMENT_INSTRUMENTS, LOOKING_FOR,
                                    OFFERINGS, PROFILE_SUBTYPE, PROFILE_VALUES,
                                    SECTOR_VALUES, SERVICE_TYPE_LABELS,
                                    STARTUP_REVENUE, STARTUP_STAGES,
                                    TICKET_SIZE, ProfileType)

fake = Faker(["en_IN", "de_DE", "en_US", "ja_JP"])


class UserDataSeeder:
    """
    Seeder for the following collections
    1. login_info (authentication + org id)
    2. user_profile (personal info)
    3. org_profile (company details)
    4. context_builder (networking preferences)
    """

    def __init__(self) -> None:
        self.db: Optional[AsyncDatabase] = None
        self.locales = ["en_IN", "de_DE", "en_US", "ja_JP"]

        self.login_info_collection: Optional[AsyncCollection] = None
        self.user_profile_collection: Optional[AsyncCollection] = None
        self.organisation_profile_collection: Optional[AsyncCollection] = None
        self.context_builder_collection: Optional[AsyncCollection] = None

        self.profile_type_weights: Dict[ProfileType, int] = {
            "startup": 25,
            "investors": 10,
            "aspirants_individuals": 25,
            "government": 5,
            "mentor_sme": 15,
            "incubation_acceleration": 5,
            "industry_corporate": 5,
            "ecosystem_service_provider": 5,
            "others": 5,
        }

        self.sectors = SECTOR_VALUES
        self.designations = DESIGNATIONS_LABELS

    def weighted_choice(self, choices: Dict[ProfileType, int]) -> str:
        """Select profile type based on ecosystem weights"""
        items = list(choices.keys())
        weights = list(choices.values())
        return random.choices(items, weights=weights)[0]

    async def initialize_collections(self):
        """Initialize database collections"""
        try:
            self.db = get_database()

            login_collection = getattr(settings, "LOGIN_COLLECTION", "login_info")
            profile_collection = getattr(
                settings, "USERS_PROFILE_COLLECTION", "user_profile"
            )
            organisation_collection = getattr(
                settings, "ORGANISATION_PROFILE_COLLECTION", "organisation_profile"
            )
            context_collection = getattr(
                settings, "CONTEXT_BUILDER_COLLECTION", "context_builder"
            )

            self.login_info_collection = self.db[login_collection]
            self.user_profile_collection = self.db[profile_collection]
            self.organisation_profile_collection = self.db[organisation_collection]
            self.context_builder_collection = self.db[context_collection]

            print("[OK] Connected to all 4 collections:")
            print(f"      - {login_collection}")
            print(f"      - {profile_collection}")
            print(f"      - {organisation_collection}")
            print(f"      - {context_collection}")

        except Exception as e:
            print(f"[ERROR] Failed to initialize collections: {e}")

    def generate_password_hash(self, password: str = "TempPass123!") -> str:
        """Generate secure bcrypt hash"""
        salt = bcrypt.gensalt(rounds=10)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def generate_indian_phone(self) -> str:
        """Generate realistic Indian mobile number"""
        first_digit = random.choice(["6", "7", "8", "9"])
        remaining_digits = "".join([str(random.randint(0, 9)) for _ in range(9)])
        return first_digit + remaining_digits

    def generate_bio(self, profile_type: str) -> str:
        """Generate realistic bio based on profile type"""
        sector = random.choice(self.sectors).replace("_", "&").title()
        years = random.randint(2, 15)

        bio_templates = {
            "startup": f"Building innovative {sector} solutions. Passionate about solving real-world problems through technology.",
            "investors": f"Early-stage investor focused on {sector} startups. {years}+ years in venture capital.",
            "aspirants_individuals": f"Aspiring professional or student passionate about {sector} and entrepreneurship. {years//2}+ years learning experience.",
            "government": f"Public servant committed to promoting {sector} innovation and entrepreneurship.",
            "mentor_sme": f"Experienced {sector} mentor with {years}+ years of industry expertise. Helping startups grow and succeed.",
            "incubation_acceleration": f"Driving startup growth in {sector} through acceleration and incubation programs. {years}+ years in ecosystem support.",
            "industry_corporate": f"Corporate innovation leader driving {sector} transformation in enterprise space.",
            "ecosystem_service_provider": f"Providing strategic support and services to {sector} startups and ecosystem players.",
            "others": f"Passionate professional in {sector} exploring new opportunities and making an impact. {years}+ years of experience.",
        }

        return bio_templates.get(
            profile_type, f"Professional working in {sector} ecosystem."
        )

    def generate_consistent_locations(self) -> dict[str, Any]:
        """generare realistic locations"""
        weights = [75, 5, 15, 5]
        locale = random.choices(self.locales, weights=weights)[0]

        locale_faker = Faker(locale)

        return {
            "country": locale_faker.country(),
            "state": locale_faker.city(),
            "city": locale_faker.city(),
            "address": locale_faker.address(),
        }

    def generate_organisation_name(self, profile_type: ProfileType) -> str:
        """Generate realistic org name by type"""
        if profile_type == "startup":
            prefixes = ["", "Next", "Smart", "Future", "Digital", "AI", "Tech"]
            suffixes = [
                "Labs",
                "Tech",
                "Solutions",
                "Ventures",
                "Systems",
                "Innovations",
            ]
            core = fake.company().split()[0]
            return f"{random.choice(prefixes)}{core} {random.choice(suffixes)}"

        elif profile_type == "investors":
            return random.choice(
                [
                    f"{fake.last_name()} Ventures",
                    f"{fake.last_name()} Capital",
                    f"Alpha {fake.company().split()[0]} Fund",
                    "Catalyst Ventures",
                ]
            )

        elif profile_type == "government":
            departments = [
                "Innovation",
                "Technology",
                "Digital",
                "Startup",
                "Enterprise",
                "Economic",
            ]
            levels = ["Ministry of", "Department of", "Directorate of", "Board of"]
            return f"{random.choice(levels)} {random.choice(departments)} Development"

        elif profile_type == "mentor_sme":
            return random.choice(
                [
                    f"{fake.last_name()} Consulting",
                    f"{fake.last_name()} Advisory",
                    f"{fake.company().split()[0]} Strategic Advisors",
                    "Growth Mentors Network",
                    "Strategic Business Advisors",
                    "Executive Mentorship Group",
                ]
            )

        elif profile_type == "incubation_acceleration":
            prefixes = ["Startup", "Innovation", "Tech", "Digital", "Future", "Growth"]
            suffixes = [
                "Incubator",
                "Accelerator",
                "Hub",
                "Lab",
                "Center",
                "Foundation",
            ]
            return f"{random.choice(prefixes)} {random.choice(suffixes)}"

        elif profile_type == "industry_corporate":
            prefixes = ["Startup", "Innovation", "Tech", "Digital", "Future", "Growth"]
            suffixes = [
                "Incubator",
                "Accelerator",
                "Hub",
                "Lab",
                "Center",
                "Foundation",
            ]
            return f"{random.choice(prefixes)} {random.choice(suffixes)}"

        elif profile_type == "ecosystem_service_provider":
            services = [
                "Legal",
                "Financial",
                "Consulting",
                "Advisory",
                "Strategic",
                "Business",
            ]
            types = [
                "Services",
                "Solutions",
                "Partners",
                "Associates",
                "Group",
                "Consultancy",
            ]
            return f"{random.choice(services)} {random.choice(types)} LLP"

        elif profile_type == "aspirants_individuals":
            return random.choice(
                [
                    f"{fake.city()} Institute of Technology",
                    f"{fake.last_name()} University",
                    f"Indian Institute of {random.choice(['Technology', 'Management', 'Science'])}",
                    f"{fake.company().split()[0]} College of Engineering",
                    "National Institute of Technology",
                    f"{fake.city()} Business School",
                ]
            )

        elif profile_type == "others":
            return random.choice(
                [
                    f"{fake.last_name()} & Associates",
                    f"{fake.company().split()[0]} Consulting",
                    "Independent Professional",
                    f"{fake.last_name()} Services",
                    "Freelance Network",
                ]
            )

        else:
            return fake.company()

    def generate_product_offerings(self, profile_type: ProfileType) -> List[str]:
        """Generate realistic product offerings based on profile type"""
        if profile_type in ["mentor_sme", "aspirants_individuals", "others"]:
            return []

        available_offerings = OFFERINGS.get(profile_type, [])

        selectable_offerings = [
            offering for offering in available_offerings if offering != "Others"
        ]

        if not selectable_offerings:
            return []

        num_offerings = random.randint(1, min(3, len(selectable_offerings)))
        selected_offerings = random.sample(selectable_offerings, num_offerings)

        if random.random() < 0.1 and "Others" in available_offerings:
            selected_offerings.append("Others")

        return selected_offerings

    # ========================================
    # COLLECTION 1: LOGIN_INFO (Authentication)
    # ========================================
    async def generate_login_info(
        self, organisation_profile_id: ObjectId
    ) -> Dict[str, Any]:
        """Generate login_info document with authentication details"""

        return {
            "_id": ObjectId(),
            "first_name": fake.first_name(),
            "last_name": fake.last_name() if random.random() > 0.2 else "",
            "profile_image": f"https://tngss-documents.s3.ap-south-1.amazonaws.com/events/{fake.uuid4()[:12]}.JPG",
            "role": "user",
            "email_id": fake.email(),
            "is_email_verified": random.choice([True, True, False]),
            "phone_number": self.generate_indian_phone(),
            "is_phone_number_verified": random.choice([False, False, False, True]),
            "password": self.generate_password_hash(),
            "auth_provider": "email",
            "organisation_profile_id": organisation_profile_id,
            "status": "active",
            "is_deleted": False,
            "auth_token": (
                f"auth_{fake.uuid4()[:16]}" if random.random() > 0.6 else None
            ),
            "createdAt": fake.date_time_between(start_date="-2y", end_date="now"),
            "updatedAt": datetime.now(timezone.utc),
            "__v": 0,
        }

    # ========================================
    # COLLECTION 2: USER_PROFILE (Personal details)
    # ========================================
    async def generate_user_profile(
        self, user_id: ObjectId, profile_type: ProfileType, organization_name: str
    ) -> Dict[str, Any]:
        """Generate user_profile document with personal information"""
        if profile_type == "aspirants_individuals":
            dob_date = fake.date_of_birth(minimum_age=18, maximum_age=28)
        else:
            dob_date = fake.date_of_birth(minimum_age=24, maximum_age=65)

        dob = datetime.combine(dob_date, datetime.min.time())

        location_data = self.generate_consistent_locations()

        return {
            "_id": ObjectId(),
            "user_id": user_id,
            "dob": dob,
            "gender": random.choice(["male", "female", "transgender"]),
            "designation": random.choice(self.designations[profile_type]),
            "education_qualification": "",
            "education_institute": (
                organization_name if profile_type == "aspirants_individuals" else ""
            ),
            "bio": self.generate_bio(profile_type),
            "address": location_data["address"],
            "city": location_data["city"],
            "state": location_data["state"],
            "country": location_data["country"],
            "linkedin_url": f"https://linkedin.com/in/{fake.user_name()}",
            "x_url": f"https://x.com/{fake.user_name()}",
            "website_url": fake.url() if random.random() > 0.6 else "",
            "focused_sector": random.choice(self.sectors),
            "focused_stage": "",
            "organization_name": (
                organization_name if profile_type != "aspirants_individuals" else ""
            ),
            "company_linkedin_url": "",
            "is_deleted": False,
            "auth_token": (
                f"auth_{fake.uuid4()[:16]}" if random.random() > 0.6 else None
            ),
            "createdAt": fake.date_time_between(start_date="-2y", end_date="now"),
            "updatedAt": datetime.now(timezone.utc),
            "__v": 0,
        }

    # ========================================
    # COLLECTION 3: ORGANISATION_PROFILE (Company details)
    # ========================================
    async def generate_organisation_profile(
        self, user_id: ObjectId, profile_type: ProfileType
    ) -> Dict[str, Any]:
        """Generate organisation_profile document with"""
        sector = random.choice(self.sectors)
        organisation_name = self.generate_organisation_name(profile_type=profile_type)
        location_data = self.generate_consistent_locations()

        base_data = {
            "_id": ObjectId(),
            "user_id": user_id,
            "profile_type": profile_type,
            "sub_type": random.choice(PROFILE_SUBTYPE[profile_type]),
            "organisation_logo": f"https://logo.clearbit.com/{fake.domain_name()}",
            "organisation_name": organisation_name,
            "city": location_data["city"],
            "country": location_data["country"],
            "about_organisation": f"Innovative {sector.replace('_', ' ')} organization focused on growth and excellence.",
            "website": f"https://{fake.domain_name()}",
            "organisation_founding_year": str(random.randint(2015, 2024)),
            "organisation_linkedin": f"https://linkedin.com/company/{organisation_name.lower().replace(' ', '-')}",
            "product_or_service_based": random.choice(
                SERVICE_TYPE_LABELS[profile_type]
            ),
            "product_offering_details": self.generate_product_offerings(profile_type),
            "incubated": (
                random.choice([True, False]) if profile_type == "startup" else False
            ),
            "incubator_name": fake.company() if profile_type == "startup" else "",
            "looking_for": random.sample(
                LOOKING_FOR[profile_type],
                random.randint(2, min(6, len(LOOKING_FOR[profile_type]))),
            ),
            "bussiness_model": (
                random.sample(
                    BUSINESS_MODEL, random.randint(1, min(3, len(BUSINESS_MODEL)))
                )
                if profile_type in ["startup", "investors"]
                else []
            ),
            "sector": sector,
            "is_deleted": False,
            "createdAt": fake.date_time_between(start_date="-1y", end_date="now"),
            "updatedAt": datetime.now(timezone.utc),
            "__v": 0,
        }

        if profile_type == "startup":
            base_data.update(
                {
                    "stage": random.choice(STARTUP_STAGES),
                    "team_size": random.randint(2, 150),
                    "fund_rise_till_date": random.choice(STARTUP_REVENUE),
                    "revenue": random.choice(STARTUP_REVENUE),
                    "pitch_deck": (
                        f"https://docsend.com/{fake.uuid4()[:8]}"
                        if random.random() > 0.4
                        else ""
                    ),
                }
            )

            return base_data

        if profile_type == "investor":
            base_data.update(
                {
                    "funding_stage": random.choice(STARTUP_STAGES),
                    "investment_instrument": random.choice(INVESTMENT_INSTRUMENTS),
                    "revenue_stage": random.choice(STARTUP_REVENUE),
                    "ticket_size": random.choice(TICKET_SIZE),
                }
            )

            return base_data

        return base_data

    # ========================================
    # COLLECTION 4: CONTEXT_BUILDER (Networking Preferences)
    # ========================================
    async def generate_context_builder(
        self, user_id: ObjectId, profile_type: ProfileType
    ) -> Dict[str, Any]:
        """Generate context_builder for networking preferences"""

        looking_to_connect = random.sample(PROFILE_VALUES, random.randint(1, 8))
        looking_to_meet = set()

        for profile in looking_to_connect:
            clean_designations = [
                d for d in DESIGNATIONS_LABELS[profile] if d != "Others"
            ]

            if clean_designations:
                sample_size = random.randint(1, min(6, len(clean_designations)))
                sampled = random.sample(clean_designations, sample_size)
                looking_to_meet.update(sampled)

        looking_to_meet = list(looking_to_meet)

        return {
            "_id": ObjectId(),
            "user_id": user_id,
            "looking_to_connect": looking_to_connect,
            "looking_to_meet": looking_to_meet,
            "sector": random.sample(self.sectors, random.randint(3, 29)),
            "is_deleted": False,
            "createdAt": fake.date_time_between(start_date="-1y", end_date="now"),
            "updatedAt": datetime.now(timezone.utc),
            "__v": 0,
        }

    # ========================================
    # Profile creation business logic
    # ========================================

    async def clear_all_collections(self):
        """Clear existing data from all 4 collections"""

        collections = [
            ("login_info", self.login_info_collection),
            ("user_profile", self.user_profile_collection),
            ("organisation_profile", self.organisation_profile_collection),
            ("context_builder", self.context_builder_collection),
        ]

        print("+------------------------------------------------+")
        print("| ~ Clearing existing data from all collections ~ |")
        print("+------------------------------------------------+")

        for name, collection in collections:
            if collection is not None:
                result = await collection.delete_many({})
                print(
                    f"[CLEARED] {result.deleted_count:5,} documents removed from '{name}'"
                )

        print("+------------------------------------------------+")
        print("|                 All collections cleared      |")
        print("+------------------------------------------------+")

    async def create_complete_user(self, profile_type: ProfileType) -> Dict[str, Any]:
        """Create complete user across ALL 4 collections with proper linking"""

        user_id = ObjectId()
        organisation_profile_id = ObjectId()

        organisation_data = await self.generate_organisation_profile(
            user_id, profile_type
        )
        organisation_data["_id"] = organisation_profile_id

        login_data = await self.generate_login_info(organisation_profile_id)
        login_data["_id"] = user_id

        user_profile_data = await self.generate_user_profile(
            user_id, profile_type, organisation_data["organisation_name"]
        )

        context_data = await self.generate_context_builder(user_id, profile_type)

        return {
            "login_info": login_data,
            "user_profile": user_profile_data,
            "organisation_profile": organisation_data,
            "context_builder": context_data,
            "profile_type": profile_type,
        }

    async def seed_all_collections(
        self, num_users: int = 100, clear_existing: bool = False
    ):
        """Main seeding function for all 4 collections"""

        print(f"+{'-'*50}+")
        print(f"| 🌱 Starting to seed {num_users} users across 4 collections... |")
        print(f"+{'-'*50}+")

        if clear_existing:
            await self.clear_all_collections()

        profile_distribution = []
        for i in range(num_users):
            profile_type = self.weighted_choice(self.profile_type_weights)
            profile_distribution.append(profile_type)

        # Prepare batch lists for all 4 collections
        login_info_batch = []
        user_profile_batch = []
        org_profile_batch = []
        context_builder_batch = []

        users_processed = 0

        # Generate all user data
        for i, profile_type in enumerate(profile_distribution):
            try:
                user_data = await self.create_complete_user(profile_type)

                login_info_batch.append(user_data["login_info"])
                user_profile_batch.append(user_data["user_profile"])
                org_profile_batch.append(user_data["organisation_profile"])
                context_builder_batch.append(user_data["context_builder"])

                users_processed += 1

                if users_processed % 100 == 0:
                    print(
                        f"[INFO] Prepared {users_processed}/{num_users} complete user profiles..."
                    )

            except Exception as e:
                print(f"[ERROR] Creating user {i+1}: {e}")
                continue

        try:
            if login_info_batch:
                print("[INFO] Performing bulk inserts across all 4 collections...")

                batch_size = 500
                total_inserted = {"login": 0, "profile": 0, "org": 0, "context": 0}

                for i in range(0, len(login_info_batch), batch_size):
                    login_batch = login_info_batch[i : i + batch_size]
                    profile_batch = user_profile_batch[i : i + batch_size]
                    org_batch = org_profile_batch[i : i + batch_size]
                    context_batch = context_builder_batch[i : i + batch_size]

                    collections_data = [
                        (self.login_info_collection, login_batch, "login"),
                        (self.user_profile_collection, profile_batch, "profile"),
                        (self.organisation_profile_collection, org_batch, "org"),
                        (self.context_builder_collection, context_batch, "context"),
                    ]

                    for collection, batch, name in collections_data:
                        if collection is not None and batch:
                            try:
                                result = await collection.insert_many(batch)
                                total_inserted[name] = len(result.inserted_ids)
                                print(
                                    f"[SUCCESS] Inserted {len(result.inserted_ids)} {name} records"
                                )
                            except Exception as e:
                                print(f"[✗] Failed to insert {name} records: {e}")

                print("\n+-------------------- SUCCESS --------------------+")
                print("   Bulk insert completed for all 4 collections")
                print(f"   - login_info documents      : {total_inserted['login']:,}")
                print(f"   - user_profile documents    : {total_inserted['profile']:,}")
                print(f"   - organisation_profile docs: {total_inserted['org']:,}")
                print(f"   - context_builder documents: {total_inserted['context']:,}")

                from collections import Counter

                distribution = Counter(profile_distribution)
                print("\n+----------------- PROFILE DISTRIBUTION -----------------+")
                for profile_type, count in sorted(
                    distribution.items(), key=lambda x: x[1], reverse=True
                ):
                    percentage = (count / len(profile_distribution)) * 100
                    print(
                        f"   - {profile_type:25} : {count:5,} users ({percentage:5.1f}%)"
                    )

                # Sample verification
                sample_login = login_info_batch[0]
                sample_profile = user_profile_batch[0]
                sample_org = org_profile_batch[0]
                sample_context = context_builder_batch[0]

                print("\n+----------------- SAMPLE USER -----------------+")
                print(
                    f"   Name       : {sample_login['first_name']} {sample_login['last_name']}"
                )
                print(f"   Email      : {sample_login['email_id']}")
                print(f"   Designation: {sample_profile['designation']}")
                print(f"   Education  : {sample_profile['education_qualification']}")
                print(f"   Organization: {sample_org['organisation_name']}")
                print(f"   Sector      : {sample_org.get('sector', 'N/A')}")
                print(
                    f"   Looking to connect: {', '.join(sample_context['looking_to_connect'])}"
                )

                print("\n+-----------------------------------------------------+")
                print(
                    "All collections seeded successfully. Ready for recommendation engine."
                )
                print("+-----------------------------------------------------+")

        except Exception as e:
            print(f"[ERROR] During bulk insert: {e}")
            raise


async def main():
    """Main CLI function"""

    parser = argparse.ArgumentParser(
        description="Seed all 4 collections of the user in MongoDB required to test the Recommendation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.seed_users.py --users 100
  python -m scripts.seed_users.py --users 1000 --clear
        """,
    )

    parser.add_argument(
        "--users", type=int, default=100, help="Number of users to create"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing data before seeding"
    )

    args = parser.parse_args()

    try:
        print("#" * 50)
        print("[*] 4-Collection Recommendation Engine Seeder")
        print("#" * 50)

        print("[*] Connecting to MongoDB...")
        await connect_to_mongo()

        seeder = UserDataSeeder()
        print("[*] Initializing collections...")
        await seeder.initialize_collections()

        await seeder.seed_all_collections(args.users, args.clear)

        print("[SUCCESS] Seeding completed successfully!")
        print(f"[+] Created {args.users:,} user profiles across 4 collections")
    except KeyboardInterrupt:
        print("\n[!] Seeding interrupted by user")
    except Exception as e:
        print(f"\n[!] Seeding failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
