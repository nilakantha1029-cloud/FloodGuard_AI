"""
Emergency contact reference numbers (India).

These are general national helplines, provided as a reference layer —
NOT verified per-district. Always direct users to confirm with local
authorities during an actual emergency; state disaster-control-room
numbers vary and change more often than national ones.
"""

NATIONAL_EMERGENCY_CONTACTS = [
    {"name": "National Emergency Number", "number": "112", "note": "All-in-one police/fire/medical emergency line"},
    {"name": "Disaster Management Helpline (NDMA)", "number": "1078", "note": "National Disaster Management Authority"},
    {"name": "Ambulance", "number": "108", "note": "Emergency medical response"},
    {"name": "Fire Services", "number": "101", "note": "Fire emergencies"},
    {"name": "Police", "number": "100", "note": "Police emergency line"},
    {"name": "Disaster Management (alt.)", "number": "1070", "note": "Used by several state control rooms"},
    {"name": "Women's Helpline", "number": "1091", "note": "Women's safety emergency line"},
]


def get_emergency_contacts(country="India"):
    """Extend this per-country as the platform scales beyond India."""
    if country == "India":
        return NATIONAL_EMERGENCY_CONTACTS
    return [{"name": "Local emergency services", "number": "N/A", "note": "Country-specific contacts not yet configured"}]
