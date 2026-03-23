"""
Data quality cleansers — updated with 7 additional fixes:
  1. Fax normalisation  — same E.164 pipeline as phone (was treating fax as float)
  2. License number     — re-attach state prefix when stripped (MA-12345 vs 12345)
  3. License state      — full name → 2-letter ISO code (Massachusetts → MA)
  4. First name         — nickname/abbreviation → canonical form (Mike → Michael)
  5. Specialty          — synonym & abbreviation normalisation (GI → Gastroenterology)
  6. Practice name      — protect medical acronyms from title-case mangling (UCSF, BMC)
  7. Gender             — unify to single character (Male → M, Female → F)

Libraries used (all optional — graceful fallback if not installed):
  phonenumbers    — E.164 phone/fax normalisation
  email-validator — email validation + normalisation
  usaddress       — US postal address parsing
  nameparser      — full name splitting
"""

from __future__ import annotations
import re
from typing import Optional

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import phonenumbers
    from phonenumbers import NumberParseException
    _HAS_PHONENUMBERS = True
except ImportError:
    _HAS_PHONENUMBERS = False

try:
    from email_validator import validate_email, EmailNotValidError
    _HAS_EMAIL_VALIDATOR = True
except ImportError:
    _HAS_EMAIL_VALIDATOR = False

try:
    import usaddress
    _HAS_USADDRESS = True
except ImportError:
    _HAS_USADDRESS = False

try:
    from nameparser import HumanName
    _HAS_NAMEPARSER = True
except ImportError:
    _HAS_NAMEPARSER = False


# ═════════════════════════════════════════════════════════════════════════════
# FIX 3 — US state name → 2-letter code lookup
# ═════════════════════════════════════════════════════════════════════════════
_STATE_NAME_TO_CODE: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
    # Common abbreviation aliases
    "calif": "CA", "colo": "CO", "conn": "CT", "fla": "FL",
    "ill": "IL", "ind": "IN", "kan": "KS", "ky": "KY",
    "la": "LA", "mich": "MI", "minn": "MN", "miss": "MS",
    "mo": "MO", "mont": "MT", "neb": "NE", "nev": "NV",
    "okla": "OK", "ore": "OR", "pa": "PA", "tenn": "TN",
    "tex": "TX", "vt": "VT", "va": "VA", "wash": "WA",
    "wis": "WI", "wyo": "WY",
}

# Valid 2-letter codes (for pass-through)
_VALID_STATE_CODES = set(_STATE_NAME_TO_CODE.values())


def normalise_state(value: str) -> str:
    """
    Normalise a US state value to its 2-letter code.
    Handles full names (Massachusetts→MA), common abbreviations, and
    malformed compound values (TN-IN → TN, takes first valid code).
    Pass-through for already-correct 2-letter codes.
    """
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()

    # Already a valid 2-letter code
    if raw.upper() in _VALID_STATE_CODES:
        return raw.upper()

    # Handle compound garbage like "TN-IN" — take first valid token
    tokens = re.split(r"[-/,\s]+", raw)
    for tok in tokens:
        if tok.upper() in _VALID_STATE_CODES:
            return tok.upper()

    # Full name lookup
    code = _STATE_NAME_TO_CODE.get(raw.lower().strip())
    if code:
        return code

    # Return original if we can't resolve
    return raw


# ═════════════════════════════════════════════════════════════════════════════
# FIX 4 — First name nickname / abbreviation → canonical form
# ═════════════════════════════════════════════════════════════════════════════
# Bidirectional: nickname → canonical  AND  initial → canonical (if unambiguous)
_NICKNAME_MAP: dict[str, str] = {
    # Common English nicknames
    "mike": "Michael", "mikey": "Michael", "mick": "Michael", "mickey": "Michael",
    "dave": "David",   "davy": "David",
    "jim": "James",    "jimmy": "James",   "jamie": "James",
    "bob": "Robert",   "bobby": "Robert",  "rob": "Robert",  "robbie": "Robert",
    "bill": "William", "billy": "William", "will": "William", "willy": "William",
    "joe": "Joseph",   "joey": "Joseph",
    "tom": "Thomas",   "tommy": "Thomas",
    "dick": "Richard", "rick": "Richard",  "ricky": "Richard",
    "dan": "Daniel",   "danny": "Daniel",
    "chris": "Christopher",
    "nick": "Nicholas",
    "matt": "Matthew",
    "alex": "Alexander",
    "andy": "Andrew",
    "tony": "Anthony",
    "ed": "Edward",    "eddie": "Edward",  "ned": "Edward",
    "fred": "Frederick",
    "charlie": "Charles", "chuck": "Charles",
    "harry": "Harold",
    "pat": "Patrick",  "patty": "Patricia",
    "sue": "Susan",    "susie": "Susan",
    "kate": "Katherine", "kathy": "Katherine", "cathy": "Catherine",
    "beth": "Elizabeth", "liz": "Elizabeth", "lisa": "Elizabeth",
    "jen": "Jennifer", "jenny": "Jennifer",
    "sam": "Samuel",   "sammy": "Samuel",
    "ben": "Benjamin",
    "steve": "Steven", "stephen": "Stephen",
    "greg": "Gregory",
    "ken": "Kenneth",
    "ron": "Ronald",
    "don": "Donald",
    "ray": "Raymond",
    "pete": "Peter",
    "larry": "Lawrence",
    "terry": "Terence",
    "jerry": "Gerald",
    "gary": "Gerald",
    "barry": "Barry",
    "sara": "Sarah",
    "kate": "Katherine",
    "sue": "Susan",
    "liz": "Elizabeth",
    # International / medical community common
    "raj": "Rajesh",
    "may": "Mei",        # Chinese name variant
    "annie": "Ann",
    "maggie": "Margaret",
    "meg": "Margaret",
    "wendy": "Gwendolyn",
    "nan": "Nancy",
    "bea": "Beatrice",
    "dot": "Dorothy",
}

# Initial-only patterns like "P." → can't resolve without context,
# but we flag them for review
_INITIAL_PATTERN = re.compile(r"^[A-Z]\.$")


def normalise_first_name(value: str) -> str:
    """
    Resolve common first name nicknames and abbreviations to canonical form.
    Returns the canonical name if found, otherwise returns title-cased original.
    Single initials (e.g. "P.") are returned unchanged — they require NPI anchor.
    """
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()

    # Single initial — can't resolve without context, leave as-is
    if _INITIAL_PATTERN.match(raw):
        return raw

    canonical = _NICKNAME_MAP.get(raw.lower())
    if canonical:
        return canonical

    # Not in map — return title-cased
    return raw.title()


# ═════════════════════════════════════════════════════════════════════════════
# FIX 5 — Specialty synonym / abbreviation → canonical NUCC-aligned term
# ═════════════════════════════════════════════════════════════════════════════
_SPECIALTY_MAP: dict[str, str] = {
    # Abbreviations
    "gi":                        "Gastroenterology",
    "ent":                       "Otolaryngology",
    "ob/gyn":                    "Obstetrics and Gynecology",
    "ob-gyn":                    "Obstetrics and Gynecology",
    "obgyn":                     "Obstetrics and Gynecology",
    "ortho":                     "Orthopedics",
    "neuro":                     "Neurology",
    "cards":                     "Cardiology",
    "onc":                       "Oncology",
    "peds":                      "Pediatrics",
    "psych":                     "Psychiatry",
    "pulm":                      "Pulmonology",
    "rheum":                     "Rheumatology",
    "uro":                       "Urology",
    "derm":                      "Dermatology",
    "endo":                      "Endocrinology",
    "nephro":                    "Nephrology",
    "hem":                       "Hematology",

    # British/variant spellings
    "orthopaedics":              "Orthopedics",
    "orthopaedic surgery":       "Orthopedic Surgery",
    "paediatrics":               "Pediatrics",
    "gynaecology":               "Gynecology",
    "haematology":               "Hematology",
    "anaesthesiology":           "Anesthesiology",
    "anaesthetics":              "Anesthesiology",

    # Synonyms / near-synonyms
    "family practice":           "Family Medicine",
    "general practice":          "Family Medicine",
    "primary care":              "Family Medicine",
    "internal medicine/primary care": "Internal Medicine",
    "infectious diseases":       "Infectious Disease",
    "infectious disease medicine": "Infectious Disease",
    "diabetology":               "Endocrinology",        # diabetology is a sub of endo
    "metabolism":                "Endocrinology",
    "hematology/oncology":       "Oncology",             # combined → parent specialty in golden
    "hem/onc":                   "Oncology",
    "cardiac electrophysiology": "Cardiology",           # sub-specialty → parent
    "interventional cardiology": "Cardiology",
    "electrophysiology":         "Cardiology",
    "general surgery":           "Surgery",
    "neurosurgery":              "Surgery",
    "colorectal surgery":        "Surgery",
    "thoracic surgery":          "Surgery",
    "vascular surgery":          "Surgery",
}


def normalise_specialty(value: str) -> str:
    """
    Normalise a specialty string to its canonical NUCC-aligned term.
    Handles abbreviations (GI), British spellings (Orthopaedics),
    and near-synonyms (Family Practice → Family Medicine).
    Returns title-cased original if no mapping found.
    """
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()
    canonical = _SPECIALTY_MAP.get(raw.lower())
    if canonical:
        return canonical
    # Check partial match for compound values
    raw_lower = raw.lower()
    for k, v in _SPECIALTY_MAP.items():
        if k in raw_lower and len(k) > 4:   # avoid short false matches
            return v
    return raw


# ═════════════════════════════════════════════════════════════════════════════
# FIX 6 — Protected acronyms for title case
# ═════════════════════════════════════════════════════════════════════════════
# Medical institution and organisation acronyms that must stay uppercase
_PROTECTED_ACRONYMS: set[str] = {
    "UCSF", "UCLA", "UCSD", "UCONN", "UMASS",
    "BMC", "MGH", "NYU", "NYP", "NYPH",
    "UPMC", "OHSU", "OSU", "LSU", "USC",
    "CHOP", "TCH", "CHOA",
    "VA", "DOD", "NIH", "CDC", "FDA", "CMS",
    "MD", "DO", "RN", "NP", "PA", "PhD", "MD-PhD",
    "PLLC", "LLC", "PC", "LLP",
    "USA", "US", "UK", "EU",
    "ER", "ICU", "OR", "NICU", "PICU",
}

_ACRONYM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _PROTECTED_ACRONYMS) + r")\b",
    re.IGNORECASE,
)


def title_case(value: str) -> str:
    """
    Title-case a string while protecting known medical acronyms (UCSF, BMC, etc.),
    handling apostrophes (O'Brien), and Mc/Mac prefixes (McDonald).
    """
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()

    # Standard title case
    result = raw.title()

    # Fix apostrophe artifacts
    result = re.sub(r"'S\b", "'s", result)
    # Restore dropped apostrophes in Irish O' surnames (O'Brien, O'Connor, etc.)
    _IRISH_O = {
        'obrien','oconnor','oneill','osullivan','omalley','oreilly','obyrne',
        'odonnell','ofarrell','ohara','omahony','oshea','oriordan','oloughlin',
        'odoherty','okane','odriscoll','oshaughnessy',
    }
    result = ' '.join(
        ("O'" + w[1:].capitalize()) if w.lower() in _IRISH_O else w
        for w in result.split()
    )

    # Fix Mc/Mac prefixes
    result = re.sub(r"\bMc([a-z])", lambda m: "Mc" + m.group(1).upper(), result)
    result = re.sub(r"\bMac([a-z])", lambda m: "Mac" + m.group(1).upper(), result)

    # Fix "Of The" prepositions → lowercase
    result = re.sub(r"\b(Of|The|And|For|In|At|By|To|From|With|A|An)\b",
                    lambda m: m.group(0).lower(), result)
    # But keep capitalised at start of string
    if result:
        result = result[0].upper() + result[1:]

    # Restore protected acronyms to uppercase
    def _restore(m: re.Match) -> str:
        word = m.group(0).upper()
        return word if word in _PROTECTED_ACRONYMS else m.group(0)

    result = _ACRONYM_PATTERN.sub(_restore, result)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# FIX 7 — Gender normalisation
# ═════════════════════════════════════════════════════════════════════════════
_GENDER_MAP: dict[str, str] = {
    "male": "M", "m": "M", "man": "M", "boy": "M",
    "female": "F", "f": "F", "woman": "F", "girl": "F",
    "unknown": "U", "not specified": "U", "other": "O",
    "non-binary": "O", "nonbinary": "O", "nb": "O",
}


def normalise_gender(value: str) -> str:
    """Normalise gender to single character: M / F / O / U."""
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    return _GENDER_MAP.get(str(value).strip().lower(), str(value).strip())


# ═════════════════════════════════════════════════════════════════════════════
# FIX 2 — License number state-prefix restoration
# ═════════════════════════════════════════════════════════════════════════════

def normalise_license_number(value: str, state_code: str = "") -> str:
    """
    Ensure a license number has its state prefix.
    If value is purely numeric and a 2-letter state_code is provided,
    prepend it as '{STATE}-{NUMBER}'.
    E.g. '12345' + 'MA' → 'MA-12345'
    """
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()

    # Already has a prefix (letters then dash/hyphen then digits)
    if re.match(r"^[A-Z]{2}-\d+", raw, re.IGNORECASE):
        return raw.upper()

    # Numeric only — prepend state code if available
    if re.match(r"^\d+$", raw) and state_code:
        code = normalise_state(state_code)
        if code and len(code) == 2:
            return f"{code}-{raw}"

    return raw


# ═════════════════════════════════════════════════════════════════════════════
# FIX 1 — Phone / Fax E.164 normalisation
# ═════════════════════════════════════════════════════════════════════════════

def normalise_phone(value: str, default_region: str = "US") -> str:
    """
    Normalise a phone or fax number to E.164 format (+12025551234).
    Handles values that were accidentally cast to float (16175550191.0 → +16175550191).
    """
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()

    # Fix float conversion artifact: '16175550191.0' → '16175550191'
    if re.match(r"^\d+\.0$", raw):
        raw = raw[:-2]   # strip the .0
        if not raw.startswith("+"):
            raw = "+" + raw

    if not _HAS_PHONENUMBERS:
        return re.sub(r"[^\d+]", "", raw)

    try:
        parsed = phonenumbers.parse(raw, default_region)
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(
                parsed, phonenumbers.PhoneNumberFormat.E164
            )
    except NumberParseException:
        pass
    return raw


def validate_phone(value: str, default_region: str = "US") -> bool:
    if not value or str(value).strip() in ("", "nan", "None"):
        return False
    if not _HAS_PHONENUMBERS:
        return bool(re.match(r"^\+?[\d\s\-().]{7,15}$", str(value).strip()))
    try:
        parsed = phonenumbers.parse(str(value).strip(), default_region)
        return phonenumbers.is_valid_number(parsed)
    except Exception:
        return False


# ── Email ─────────────────────────────────────────────────────────────────────

def normalise_email(value: str) -> str:
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()
    if not _HAS_EMAIL_VALIDATOR:
        return raw.lower().strip()
    try:
        validated = validate_email(raw, check_deliverability=False)
        return validated.normalized
    except EmailNotValidError:
        return ""


def validate_email_addr(value: str) -> bool:
    if not value or str(value).strip() in ("", "nan", "None"):
        return False
    raw = str(value).strip()
    if not _HAS_EMAIL_VALIDATOR:
        return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", raw))
    try:
        validate_email(raw, check_deliverability=False)
        return True
    except EmailNotValidError:
        return False


# ── Address ───────────────────────────────────────────────────────────────────
_ABBR = {
    r"\bSt\b":   "Street",    r"\bAve\b":   "Avenue",
    r"\bBlvd\b": "Boulevard", r"\bDr\b":    "Drive",
    r"\bRd\b":   "Road",      r"\bLn\b":    "Lane",
    r"\bCt\b":   "Court",     r"\bPl\b":    "Place",
    r"\bSq\b":   "Square",    r"\bFwy\b":   "Freeway",
    r"\bHwy\b":  "Highway",   r"\bPkwy\b":  "Parkway",
    r"\bN\b":    "North",     r"\bS\b":     "South",
    r"\bE\b":    "East",      r"\bW\b":     "West",
    r"\bApt\b":  "Apartment", r"\bSte\b":   "Suite",
    r"\bFlr\b":  "Floor",     r"\bBldg\b":  "Building",
}


def expand_address_abbreviations(value: str) -> str:
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    result = str(value).strip()
    for pattern, replacement in _ABBR.items():
        result = re.sub(pattern, replacement, result)
    return result


def parse_address(value: str) -> dict[str, str]:
    if not value or str(value).strip() in ("", "nan", "None"):
        return {}
    raw = str(value).strip()
    if not _HAS_USADDRESS:
        return {"full_address": raw}
    try:
        tagged, _ = usaddress.tag(raw)
        return dict(tagged)
    except Exception:
        return {"full_address": raw}


def standardise_address(value: str) -> str:
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    expanded = expand_address_abbreviations(str(value).strip())
    return title_case(expanded)


# ── Name ──────────────────────────────────────────────────────────────────────

def standardise_name(value: str) -> str:
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()
    if not _HAS_NAMEPARSER:
        return title_case(raw)
    name = HumanName(raw)
    return str(name).strip() or title_case(raw)


def parse_full_name(value: str) -> dict[str, str]:
    if not value or str(value).strip() in ("", "nan", "None"):
        return {}
    raw = str(value).strip()
    if not _HAS_NAMEPARSER:
        return {"full": raw}
    name = HumanName(raw)
    return {
        "title":  str(name.title).strip(),
        "first":  str(name.first).strip(),
        "middle": str(name.middle).strip(),
        "last":   str(name.last).strip(),
        "suffix": str(name.suffix).strip(),
    }


# ── Date ──────────────────────────────────────────────────────────────────────

def standardise_date(value: str, day_first: bool = False) -> str:
    """Normalise a date string to ISO 8601 (YYYY-MM-DD)."""
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()
    try:
        import pandas as pd
        parsed = pd.to_datetime(raw, dayfirst=day_first, errors="raise")
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return raw


# ═════════════════════════════════════════════════════════════════════════════
# Bulk apply — maps fix label → cleanser function
# Called by app.py Step 3 when user clicks "Apply fixes"
# ═════════════════════════════════════════════════════════════════════════════

def apply_fix_to_column(df, col: str, fix_type: str):
    """
    Apply a named fix to a DataFrame column. Returns a modified copy.

    Handles all fix types surfaced by _suggest_dq() in app.py plus the
    new domain-specific fixes (state, specialty, gender, license, acronyms).
    """
    import pandas as pd
    df = df.copy()
    if col not in df.columns:
        return df

    s = df[col].astype(str).str.strip()

    if "E.164" in fix_type or "phone" in fix_type.lower() or "fax" in fix_type.lower():
        df[col] = s.apply(normalise_phone)

    elif "Title Case" in fix_type or "casing" in fix_type.lower():
        # For first names: also run through nickname resolver
        col_lower = col.lower()
        if any(x in col_lower for x in ("first", "given")):
            df[col] = s.apply(normalise_first_name)
        elif any(x in col_lower for x in ("practice", "organization", "org", "facility", "hospital")):
            df[col] = s.apply(title_case)   # uses protected-acronym version
        else:
            df[col] = s.apply(title_case)

    elif "email" in fix_type.lower():
        df[col] = s.apply(normalise_email)

    elif "ISO 8601" in fix_type or "YYYY-MM-DD" in fix_type:
        df[col] = s.apply(standardise_date)

    elif "abbreviation" in fix_type.lower() or "St→Street" in fix_type:
        df[col] = s.apply(standardise_address)

    elif "state" in fix_type.lower() or "ISO code" in fix_type.lower():
        df[col] = s.apply(normalise_state)

    elif "specialty" in fix_type.lower() or "synonym" in fix_type.lower():
        df[col] = s.apply(normalise_specialty)

    elif "gender" in fix_type.lower():
        df[col] = s.apply(normalise_gender)

    elif "license" in fix_type.lower() and "prefix" in fix_type.lower():
        # License number fix requires the adjacent state column
        state_col = None
        for candidate in ("license_state", "state", "lic_state"):
            if candidate in df.columns:
                state_col = candidate
                break
        if state_col:
            df[col] = df.apply(
                lambda row: normalise_license_number(
                    str(row[col]), str(row[state_col])
                ), axis=1
            )
        else:
            df[col] = s.apply(lambda v: normalise_license_number(v))

    return df


def available_cleansers() -> dict[str, bool]:
    """Return which optional libraries are installed."""
    return {
        "phonenumbers (phone/fax E.164)":  _HAS_PHONENUMBERS,
        "email-validator (email norm.)":   _HAS_EMAIL_VALIDATOR,
        "usaddress (US addr parsing)":     _HAS_USADDRESS,
        "nameparser (name splitting)":     _HAS_NAMEPARSER,
    }
