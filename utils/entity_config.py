"""
Entity configuration — per-entity-type knowledge injected into LLM prompts.

Adding a new entity type requires only adding an entry to ENTITY_CONFIGS below.
The LLM prompts in llm.py pull from this at runtime based on the entity_type
selected by the user in the sidebar.

Supported entity types (auto-detected from entity_type string, case-insensitive):
  HCP / Provider / Physician / Doctor / Clinician
  Customer / Consumer / Individual / Person / Contact
  Account / Organization / Company / Business / B2B
  Patient / Member / Beneficiary
  Location / Site / Facility / Clinic / Hospital
"""

from __future__ import annotations


# ── Per-entity inference guide ────────────────────────────────────────────────
# These are injected into the semantic analysis prompt as:
# "ENTITY-SPECIFIC FIELD RECOGNITION GUIDE"

_HCP_GUIDE = """
HEALTHCARE PROVIDER (HCP) field recognition — apply these when entity_type is HCP/Provider:
  • 10-digit all-digit column                    → NPI (strong_identifier)
  • "letter(s) + 7 digits" pattern (AA1234563)   → DEA number (scoped_identifier)
  • Alphanumeric with state prefix (MA-12345)    → State medical license (scoped_identifier)
  • Values like "208D00000X", "207R00000X"        → NUCC taxonomy code (taxonomy_code)
  • Values like "MD", "DO", "RN", "NP", "PhD"    → Professional credential (credential)
  • Cardiology, GI, Family Medicine etc.         → Medical specialty (taxonomy_code)
  • Hospital, clinic, practice name              → organization_name
  • NPI Type, NPI Status, Enumeration Date       → system_field
  Strong identifiers for HCP: NPI, DEA, UPIN, MRN
  Scoped identifiers for HCP: State License (per state), Medicaid ID (per state), DEA
  Negative rule eligibility: Gender (if coded M/F), HCPType (if controlled vocab only)
""".strip()

_CUSTOMER_GUIDE = """
CUSTOMER / CONSUMER field recognition — apply when entity_type is Customer/Consumer/Individual:
  • 9-digit all-digit column                     → SSN (strong_identifier — handle sensitively)
  • "XX-XXXXXXX" format (12-3456789)             → Federal Tax ID / EIN (strong_identifier)
  • Loyalty card number, membership number       → scoped_identifier (per program)
  • Customer ID, Account Number, Client ID       → scoped_identifier (per system/brand)
  • Email address                                → contact_info (supporting only — can change)
  • First, Last, Middle name                     → person_name
  • Company / Employer name                      → organization_name
  • Billing / Shipping address fields            → address_component
  • DOB, Age, Gender, Marital Status             → demographic (supporting only)
  • Income tier, credit score, segment code      → taxonomy_code (supporting/filter only)
  • Opt-in flags, consent flags                  → system_field (not matchable)
  • Order ID, Transaction ID, Session ID         → system_field
  Strong identifiers for Customer: SSN, EIN, Passport Number, National ID
  Scoped identifiers: Loyalty Number (per program), Customer ID (per system), Membership ID
  Negative rule eligibility: Gender (if coded M/F only), Country (if reliably populated)
  NEVER use email, phone, or address as primary — these change and are shared
""".strip()

_ACCOUNT_GUIDE = """
ACCOUNT / ORGANIZATION / B2B field recognition — apply when entity_type is Account/Company/Org:
  • 9-digit (XX-XXXXXXX) format                  → EIN / Federal Tax ID (strong_identifier)
  • DUNS number (9-digit)                        → DUNS (strong_identifier — globally unique)
  • Stock ticker, LEI code                       → strong_identifier
  • Company name, Trade name, DBA name           → organization_name (fuzzy match)
  • Website domain (acme.com)                    → scoped_identifier (unique per domain)
  • Account Number, CRM ID, Salesforce ID        → scoped_identifier (per system)
  • Industry code (SIC, NAICS)                   → taxonomy_code (filter only)
  • Revenue tier, employee count tier, segment   → taxonomy_code (filter only)
  • Billing / HQ address fields                  → address_component
  • Phone, email                                 → contact_info (supporting only)
  • Parent company, subsidiary flag              → system_field
  Strong identifiers for Account: EIN, DUNS, LEI, Stock Ticker
  Scoped identifiers: Account Number (per CRM), Domain (per organization)
  For org name matching: use fuzzy/DoubleMetaphone — "Inc", "LLC", "Corp" suffixes vary
  Negative rule eligibility: Country (if reliably populated and coded ISO-2)
""".strip()

_PATIENT_GUIDE = """
PATIENT / MEMBER / BENEFICIARY field recognition — apply when entity_type is Patient/Member:
  • Medical Record Number (MRN)                  → scoped_identifier (per facility)
  • Insurance Member ID                          → scoped_identifier (per payer)
  • Medicare / Medicaid ID                       → scoped_identifier (per program)
  • SSN                                          → strong_identifier (use carefully — PII)
  • First, Last, Middle name                     → person_name
  • DOB                                          → demographic (critical corroborator)
  • Gender                                       → demographic
  • Address fields                               → address_component
  • Plan ID, Group Number                        → scoped_identifier (per payer)
  • Diagnosis codes (ICD-10)                     → taxonomy_code (not matchable)
  • Encounter ID, Claim ID, Visit ID             → system_field
  Strong identifiers: SSN, Medicare ID, Medicaid ID
  Scoped identifiers: MRN (per facility), Member ID (per payer), Policy Number (per insurer)
  Negative rule eligibility: Gender (M/F only), DOB should NEVER be a negative rule field
""".strip()

_LOCATION_GUIDE = """
LOCATION / FACILITY / SITE field recognition — apply when entity_type is Location/Facility/Site:
  • NPI (for healthcare facilities)              → strong_identifier (type=2 org NPI)
  • Facility License Number                      → scoped_identifier (per state)
  • FIPS code, Census tract                      → scoped_identifier (geographic scope)
  • Facility name, Hospital name, Clinic name    → organization_name (fuzzy match)
  • Physical address fields                      → address_component (PRIMARY match signal)
  • Phone, fax                                   → contact_info
  • Facility type, bed count tier                → taxonomy_code (filter only)
  • Parent system, network affiliation           → system_field
  Strong identifiers: NPI (facility type 2), DEA (if dispensing)
  Scoped identifiers: State license, Medicare Provider Number, CMS Certification Number
  For location matching: composite address rule (street + city + state) is HIGH value
  Negative rule eligibility: Facility Type (if controlled vocab)
""".strip()

_GENERIC_GUIDE = """
GENERIC entity field recognition — read sample_values carefully to determine semantic type:
  • Globally unique numeric/alphanumeric ID      → strong_identifier
  • ID unique only within a scope/system         → scoped_identifier
  • Person name parts                            → person_name
  • Organisation/company name                   → organization_name
  • Address parts                               → address_component
  • Email, phone                                → contact_info
  • Date of birth, age, gender                  → demographic
  • Category codes, classification codes        → taxonomy_code
  • System timestamps, row IDs, audit fields    → system_field
  • Free text notes                             → free_text
""".strip()


# ── Per-entity negative rule guidance ────────────────────────────────────────

_NEGATIVE_RULE_GUIDANCE = {
    "hcp": """
  APPROVED for negativeRule (HCP):
    ✅ Gender       — only if coded M/F, not free text
    ✅ HCPType      — only if controlled vocabulary
  BANNED for negativeRule (HCP):
    ❌ Specialty / TaxonomyCode — different systems use different codes for same specialty
    ❌ DOB — format variations across systems block legitimate merges
    ❌ DEA — same HCP has multiple DEA numbers
""".strip(),

    "customer": """
  APPROVED for negativeRule (Customer):
    ✅ Gender       — only if coded M/F consistently
    ✅ Country      — only if ISO-2 coded and reliably populated
  BANNED for negativeRule (Customer):
    ❌ DOB — format variations, often missing
    ❌ Email / Phone — change frequently, shared by household
    ❌ Income tier / Segment — unreliable across source systems
""".strip(),

    "account": """
  APPROVED for negativeRule (Account/Org):
    ✅ Country      — if ISO-2 coded and p_missing < 0.2
    ✅ EntityType   — if values like "LLC","Corp","Partnership" are consistent
  BANNED for negativeRule (Account/Org):
    ❌ Industry code — same company has different SIC/NAICS across systems
    ❌ Revenue tier  — changes year to year
""".strip(),

    "patient": """
  APPROVED for negativeRule (Patient):
    ✅ Gender       — only if coded M/F consistently AND p_missing < 0.2
  BANNED for negativeRule (Patient):
    ❌ DOB — critical for matching but NOT for blocking (too many format errors)
    ❌ Insurance plan — patient changes plans frequently
    ❌ Diagnosis codes — not identity fields
""".strip(),

    "location": """
  APPROVED for negativeRule (Location):
    ✅ FacilityType — if controlled vocabulary (Hospital, Clinic, etc.)
    ✅ Country       — if ISO-2 coded
  BANNED for negativeRule (Location):
    ❌ Phone / Fax — change, shared across departments
    ❌ Bed count — changes year to year
""".strip(),
}


# ── Entity type detection ─────────────────────────────────────────────────────

def detect_entity_class(entity_type: str) -> str:
    """
    Map a free-text entity type to one of our known classes.
    Returns: 'hcp' | 'customer' | 'account' | 'patient' | 'location' | 'generic'
    """
    et = entity_type.lower().strip()

    hcp_terms      = ("hcp","provider","physician","doctor","clinician","practitioner",
                       "nurse","therapist","dentist","pharmacist","prescriber")
    customer_terms = ("customer","consumer","individual","person","contact","lead",
                       "prospect","subscriber","member","loyalty","retail","b2c")
    account_terms  = ("account","organization","organisation","company","business",
                       "enterprise","b2b","vendor","supplier","partner","employer",
                       "corporate","firm","entity","brand")
    patient_terms  = ("patient","beneficiary","enrollee","insured","claimant",
                       "recipient")
    location_terms = ("location","site","facility","clinic","hospital","practice",
                       "office","branch","store","venue","address")

    for term in hcp_terms:
        if term in et: return "hcp"
    for term in patient_terms:
        if term in et: return "patient"
    for term in location_terms:
        if term in et: return "location"
    for term in account_terms:
        if term in et: return "account"
    for term in customer_terms:
        if term in et: return "customer"
    return "generic"


def get_inference_guide(entity_type: str) -> str:
    """Return the field recognition guide for the given entity type."""
    cls = detect_entity_class(entity_type)
    return {
        "hcp":      _HCP_GUIDE,
        "customer": _CUSTOMER_GUIDE,
        "account":  _ACCOUNT_GUIDE,
        "patient":  _PATIENT_GUIDE,
        "location": _LOCATION_GUIDE,
        "generic":  _GENERIC_GUIDE,
    }.get(cls, _GENERIC_GUIDE)


def get_negative_rule_guidance(entity_type: str) -> str:
    """Return the negative rule eligibility guidance for the given entity type."""
    cls = detect_entity_class(entity_type)
    return _NEGATIVE_RULE_GUIDANCE.get(cls, _NEGATIVE_RULE_GUIDANCE.get("customer", ""))


def get_entity_label(entity_type: str) -> str:
    """Return a human-readable label for display."""
    cls = detect_entity_class(entity_type)
    return {
        "hcp":      "Healthcare Provider (HCP)",
        "customer": "Customer / Consumer",
        "account":  "Account / Organization",
        "patient":  "Patient / Member",
        "location": "Location / Facility",
        "generic":  "Generic Entity",
    }.get(cls, entity_type)


def get_dq_field_hints(entity_type: str) -> dict[str, list[str]]:
    """
    Return column name patterns → fix types for _suggest_dq() in app.py.
    This makes the DQ detection entity-aware without hardcoding column names.
    """
    cls = detect_entity_class(entity_type)

    # Base hints applicable to all entity types
    base = {
        "phone":    ["phone","mobile","tel","cell","fax"],
        "fax":      ["fax"],
        "email":    ["email","mail","e-mail"],
        "date":     ["dob","birth","date","enumeration","created","updated"],
        "address":  ["address","street","addr","line1","line2"],
        "state":    ["state","province","region","license_state","lic_state"],
        "gender":   ["gender","sex"],
        "first":    ["first","given","firstname","fname"],
        "last":     ["last","surname","lastname","lname"],
        "middle":   ["middle","middlename","mname"],
        "org_name": ["practice","company","organization","organisation",
                     "employer","facility","hospital","clinic","business","firm"],
    }

    # Entity-specific extras
    extras = {
        "hcp": {
            "specialty": ["specialty","speciality","spec","discipline"],
            "license":   ["license","licence","lic_num","licnum","reg_num"],
        },
        "customer": {
            "loyalty":   ["loyalty","membership","member_id","account_no","account_number",
                          "customer_id","client_id"],
        },
        "account": {
            "industry":  ["industry","sector","sic","naics","vertical"],
            "company":   ["company","dba","trade_name","legal_name","brand"],
        },
        "patient": {
            "mrn":       ["mrn","medical_record","patient_id","member_id"],
            "insurance": ["plan","policy","group_number","member_number"],
        },
        "location": {
            "facility":  ["facility","site","location","branch","store"],
        },
    }

    result = dict(base)
    result.update(extras.get(cls, {}))
    return result
