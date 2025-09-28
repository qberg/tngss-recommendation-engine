@router.get("/api/profile-options/{profile_type}")
async def get_profile_options(profile_type: ProfileType):
    return {
        "profile_type": profile_type,
        "offerings": {
            "applicable": profile_type
            not in ["mentor_sme", "aspirants_individuals", "others"],
            "options": (
                OFFERINGS.get(profile_type, [])
                if profile_type not in ["mentor_sme", "aspirants_individuals", "others"]
                else []
            ),
        },
        "looking_for": {
            "applicable": True,  # All profiles have this
            "options": LOOKING_FOR.get(profile_type, []),
        },
    }
