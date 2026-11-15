"""Original company normalization and heuristic subgroups."""
import re

SUFFIX = r"\b(inc|incorporated|llc|ltd|corp|corporation|co|company|the|plc|group|holdings|lp|llp)\b"


def norm_co(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(SUFFIX, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or "unknown"


# ---------------------------------------------------------------- subgroups
BAY = ["san francisco", "oakland", "san jose", "berkeley", "palo alto", "mountain view",
       "sunnyvale", "santa clara", "fremont", "redwood city", "san mateo", "menlo park",
       "cupertino", "hayward", "walnut creek", "pleasanton", "milpitas", "emeryville",
       "richmond", "concord", "livermore", "daly city", "san rafael", "foster city",
       "burlingame", "san bruno", "santa rosa", "napa", "vallejo", "san ramon", "dublin"]
LA = ["los angeles", "long beach", "irvine", "santa monica", "pasadena", "burbank",
      "glendale", "anaheim", "torrance", "santa ana", "culver city", "el segundo",
      "ontario", "riverside", "costa mesa", "newport beach", "orange", "fullerton"]
SD = ["san diego", "chula vista", "carlsbad", "oceanside", "la jolla", "escondido"]


def region(loc):
    l = str(loc).lower()
    if l in ("", "nan", "none"):
        return "Unknown"
    if any(c in l for c in BAY):
        return "Bay Area"
    if any(c in l for c in LA):
        return "LA area"
    if any(c in l for c in SD):
        return "San Diego area"
    if "sacramento" in l:
        return "Sacramento"
    return "Other CA"


ROLES = [
    ("Customer service", r"customer service|call center|customer support|representative"),
    ("Healthcare", r"nurse|\brn\b|medical|clinical|physician|therapist|pharmac|caregiver|\bcna\b|patient"),
    ("Tech/Engineering", r"engineer|developer|software|\bdata\b|analyst|scientist|programmer|devops|\bit\b"),
    ("Food service", r"server|cook|barista|crew member|restaurant|bartender|dishwasher|kitchen|food"),
    ("Warehouse/Logistics", r"warehouse|driver|delivery|package|picker|forklift|logistics|courier|fulfillment"),
    ("Retail/Sales", r"sales|retail|cashier|store|merchandis|associate"),
    ("Management", r"manager|director|supervisor|\blead\b|head of|\bvp\b|president"),
    ("Education", r"teacher|tutor|professor|instructor|educat"),
]


def role_family(t):
    t = str(t).lower()
    if t in ("", "nan", "none", "anonymous employee", "anonymous"):
        return "Unknown"
    for name, pat in ROLES:
        if re.search(pat, t):
            return name
    return "Other"


def status_group(s):
    s = str(s).lower()
    if "former" in s:
        return "Former"
    if "current" in s:
        return "Current"
    return "Unknown"
