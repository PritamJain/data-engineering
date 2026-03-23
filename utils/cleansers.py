"""
Data quality cleansers using free open-source libraries.

Libraries used:
  phonenumbers   — Google's libphonenumber port. Normalises to E.164.
  email-validator — Validates + normalises email addresses.
  usaddress      — Parses US postal addresses into structured components.
  nameparser     — Splits full names into first/middle/last/suffix.

All imports are guarded — if a library is not installed the function
falls back gracefully and returns the original value unchanged.
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


# ── Phone ─────────────────────────────────────────────────────────────────────

def normalise_phone(value: str, default_region: str = "US") -> str:
    """
    Normalise a phone number to E.164 format (+12025551234).
    Falls back to the original stripped value if parsing fails.
    """
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()
    if not _HAS_PHONENUMBERS:
        # Basic strip: keep digits and leading +
        digits = re.sub(r"[^\d+]", "", raw)
        return digits if digits else raw
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
    """Return True if the phone number is valid."""
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
    """
    Validate and normalise an email address (lowercase, strip whitespace).
    Returns empty string if invalid.
    """
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()
    if not _HAS_EMAIL_VALIDATOR:
        # Basic normalisation: lowercase + strip
        return raw.lower().strip()
    try:
        validated = validate_email(raw, check_deliverability=False)
        return validated.normalized
    except EmailNotValidError:
        return ""   # mark as invalid by returning empty


def validate_email_addr(value: str) -> bool:
    """Return True if the email address is syntactically valid."""
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
    r"\bSt\b":     "Street",    r"\bAve\b":    "Avenue",
    r"\bBlvd\b":   "Boulevard", r"\bDr\b":     "Drive",
    r"\bRd\b":     "Road",      r"\bLn\b":     "Lane",
    r"\bCt\b":     "Court",     r"\bPl\b":     "Place",
    r"\bSq\b":     "Square",    r"\bFwy\b":    "Freeway",
    r"\bHwy\b":    "Highway",   r"\bPkwy\b":   "Parkway",
    r"\bN\b":      "North",     r"\bS\b":      "South",
    r"\bE\b":      "East",      r"\bW\b":      "West",
    r"\bNE\b":     "Northeast", r"\bNW\b":     "Northwest",
    r"\bSE\b":     "Southeast", r"\bSW\b":     "Southwest",
    r"\bApt\b":    "Apartment", r"\bSte\b":    "Suite",
    r"\bFlr\b":    "Floor",     r"\bBldg\b":   "Building",
    r"\bPO Box\b": "PO Box",
}


def expand_address_abbreviations(value: str) -> str:
    """Expand common US address abbreviations to full words."""
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    result = str(value).strip()
    for pattern, replacement in _ABBR.items():
        result = re.sub(pattern, replacement, result)
    return result


def parse_address(value: str) -> dict[str, str]:
    """
    Parse a US address string into components using usaddress.
    Returns a dict with keys like AddressNumber, StreetName, PlaceName, StateName, ZipCode.
    Falls back to {"full_address": value} if usaddress is not installed or parsing fails.
    """
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
    """Expand abbreviations and title-case the result."""
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    expanded = expand_address_abbreviations(str(value).strip())
    return expanded.title()


# ── Name ──────────────────────────────────────────────────────────────────────

def standardise_name(value: str) -> str:
    """Title-case a name, handling Mc/Mac prefixes."""
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()
    if not _HAS_NAMEPARSER:
        return raw.title()
    name = HumanName(raw)
    return str(name).strip() or raw.title()


def parse_full_name(value: str) -> dict[str, str]:
    """
    Split a full name into {'first': ..., 'middle': ..., 'last': ..., 'suffix': ...}.
    Requires nameparser. Falls back to {'full': value}.
    """
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

_DATE_PATTERNS = [
    (re.compile(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})$"), "dmy"),   # DD/MM/YYYY or MM/DD/YYYY
    (re.compile(r"^(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})$"), "ymd"),   # YYYY-MM-DD
    (re.compile(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})$"),  "dmy2"), # DD/MM/YY
]


def standardise_date(value: str, day_first: bool = False) -> str:
    """
    Normalise a date string to ISO 8601 (YYYY-MM-DD).
    day_first=True assumes DD/MM/YYYY; False assumes MM/DD/YYYY for ambiguous formats.
    Returns original value if parsing fails.
    """
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()
    try:
        import pandas as pd
        parsed = pd.to_datetime(raw, dayfirst=day_first, errors="raise")
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return raw


# ── Casing ────────────────────────────────────────────────────────────────────

def title_case(value: str) -> str:
    """Title-case a string, handling apostrophes and hyphens."""
    if not value or str(value).strip() in ("", "nan", "None"):
        return ""
    raw = str(value).strip()
    # Use nameparser for names if available, else basic title case
    result = raw.title()
    # Fix common title-case artifacts: O'Brien, McDonald
    result = re.sub(r"'S\b", "'s", result)
    result = re.sub(r"\bMc([a-z])", lambda m: "Mc" + m.group(1).upper(), result)
    result = re.sub(r"\bMac([a-z])", lambda m: "Mac" + m.group(1).upper(), result)
    return result


# ── Bulk apply to DataFrame column ───────────────────────────────────────────

def apply_fix_to_column(df, col: str, fix_type: str):
    """
    Apply a named fix to a DataFrame column in-place (returns modified copy).

    fix_type values match the Fix strings generated by _suggest_dq() in app.py:
      "Normalise to E.164 format"
      "Standardise casing (Title Case)"
      "Flag invalid email addresses"    ← blanks invalids, keeps valids normalised
      "Standardise to ISO 8601 (YYYY-MM-DD)"
      "Expand abbreviations (St→Street, Ave→Avenue)"
    """
    df = df.copy()
    if col not in df.columns:
        return df

    s = df[col].astype(str)

    if "E.164" in fix_type:
        df[col] = s.apply(normalise_phone)

    elif "Title Case" in fix_type or "casing" in fix_type.lower():
        df[col] = s.apply(title_case)

    elif "email" in fix_type.lower():
        df[col] = s.apply(normalise_email)

    elif "ISO 8601" in fix_type or "YYYY-MM-DD" in fix_type:
        df[col] = s.apply(standardise_date)

    elif "abbreviation" in fix_type.lower() or "St→Street" in fix_type:
        df[col] = s.apply(standardise_address)

    return df


def available_cleansers() -> dict[str, bool]:
    """Return which optional cleansing libraries are installed."""
    return {
        "phonenumbers (phone E.164)":    _HAS_PHONENUMBERS,
        "email-validator (email norm.)": _HAS_EMAIL_VALIDATOR,
        "usaddress (US addr parsing)":   _HAS_USADDRESS,
        "nameparser (name splitting)":   _HAS_NAMEPARSER,
    }
