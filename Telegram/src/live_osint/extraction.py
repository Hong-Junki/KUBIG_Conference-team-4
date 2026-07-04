from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone


CONFLICT_KEYWORDS = {
    "airstrike": 0.95,
    "air strike": 0.95,
    "shelling": 0.90,
    "missile": 0.90,
    "rocket": 0.85,
    "drone": 0.75,
    "explosion": 0.75,
    "clash": 0.80,
    "armed clash": 0.95,
    "battle": 0.80,
    "attack": 0.75,
    "ambush": 0.80,
    "casualties": 0.65,
    "killed": 0.70,
    "wounded": 0.60,
    "troop": 0.45,
    "evacuation": 0.40,
    "protest": 0.35,
    "riot": 0.60,
    "curfew": 0.35,
}

NEGATION_PATTERNS = [
    r"\bno\s+(armed\s+)?clashes?\b",
    r"\bno\s+casualties\b",
    r"\bno\s+attack\b",
    r"\bpeaceful\b",
    r"\bunconfirmed\b",
]

COUNTRY_ALIASES = {
    "AFG": ["afghanistan", "kabul"],
    "ARM": ["armenia", "yerevan"],
    "AZE": ["azerbaijan", "baku"],
    "BFA": ["burkina faso", "ouagadougou"],
    "CAF": ["central african republic", "bangui"],
    "COD": ["drc", "congo", "kinshasa"],
    "ETH": ["ethiopia", "addis ababa"],
    "HTI": ["haiti", "port-au-prince"],
    "IND": ["india", "new delhi", "delhi"],
    "IRN": ["iran", "tehran"],
    "IRQ": ["iraq", "baghdad"],
    "ISR": ["israel", "tel aviv", "jerusalem"],
    "KEN": ["kenya", "nairobi"],
    "KWT": ["kuwait", "kuwait city"],
    "LBN": ["lebanon", "beirut", "sidon", "tyre"],
    "LBY": ["libya", "tripoli"],
    "LTU": ["lithuania", "vilnius"],
    "MLI": ["mali", "bamako"],
    "MMR": ["myanmar", "burma", "yangon", "naypyidaw"],
    "NER": ["niger", "niamey"],
    "NGA": ["nigeria", "abuja", "lagos"],
    "PAK": ["pakistan", "islamabad", "karachi"],
    "PSE": ["gaza", "palestine", "palestinian", "west bank", "jenin", "rafah", "gaza city"],
    "ROU": ["romania", "bucharest", "galati"],
    "RUS": [
        "russia",
        "moscow",
        "belgorod",
        "kursk",
        "tambov",
        "tambov oblast",
        "michurinsk",
        "st. petersburg",
        "saint petersburg",
        "syzran",
        "samara oblast",
    ],
    "SAU": ["saudi arabia", "riyadh"],
    "SDN": ["sudan", "khartoum", "darfur"],
    "SOM": ["somalia", "mogadishu"],
    "SSD": ["south sudan", "juba"],
    "SYR": ["syria", "damascus", "idlib", "aleppo"],
    "TCD": ["chad", "ndjamena", "n'djamena"],
    "TUR": ["turkey", "turkiye", "ankara"],
    "UKR": [
        "ukraine",
        "kyiv",
        "kiev",
        "kharkiv",
        "kramatorsk",
        "sumy",
        "donetsk",
        "dnipro",
        "zaporizhzhia",
        "zaporizhia",
        "odesa",
        "odessa",
        "kupiansk",
        "kherson",
        "mykolaiv",
        "chernihiv",
        "poltava",
        "cherkasy",
        "crimea",
    ],
    "USA": ["united states", "u.s.", "us navy", "american forces", "washington"],
    "YEM": ["yemen", "sanaa", "aden", "hodeidah"],
}

CITY_COORDS = {
    "kabul": (34.5553, 69.2075),
    "kharkiv": (49.9935, 36.2304),
    "kyiv": (50.4501, 30.5234),
    "kramatorsk": (48.7381, 37.5844),
    "sumy": (50.9077, 34.7981),
    "dnipro": (48.4647, 35.0462),
    "donetsk": (48.0159, 37.8028),
    "zaporizhzhia": (47.8388, 35.1396),
    "gaza": (31.5017, 34.4668),
    "gaza city": (31.5017, 34.4668),
    "rafah": (31.2969, 34.2445),
    "khartoum": (15.5007, 32.5599),
    "darfur": (13.0, 24.0),
    "sidon": (33.5571, 35.3715),
    "tyre": (33.2704, 35.2038),
    "beirut": (33.8938, 35.5018),
    "jenin": (32.4594, 35.3009),
    "west bank": (31.9466, 35.3027),
    "new delhi": (28.6139, 77.209),
    "delhi": (28.6139, 77.209),
    "nairobi": (-1.2921, 36.8219),
    "kuwait city": (29.3759, 47.9774),
    "vilnius": (54.6872, 25.2797),
    "bucharest": (44.4268, 26.1025),
    "galati": (45.4353, 28.008),
    "tambov": (52.7212, 41.4523),
    "tambov oblast": (52.6417, 41.4216),
    "michurinsk": (52.8978, 40.4907),
    "st. petersburg": (59.9311, 30.3609),
    "saint petersburg": (59.9311, 30.3609),
    "syzran": (53.1558, 48.4745),
    "samara oblast": (53.4184, 50.4726),
    "odesa": (46.4825, 30.7233),
    "odessa": (46.4825, 30.7233),
    "kupiansk": (49.7106, 37.6158),
    "kherson": (46.6354, 32.6169),
    "mykolaiv": (46.975, 31.9946),
    "chernihiv": (51.4982, 31.2893),
    "poltava": (49.5883, 34.5514),
    "cherkasy": (49.4444, 32.0598),
    "crimea": (45.3, 34.4),
    "idlib": (35.9306, 36.6339),
    "aleppo": (36.2021, 37.1343),
    "mogadishu": (2.0469, 45.3182),
}

COUNTRY_CENTROIDS = {
    "AFG": (33.9391, 67.71),
    "ARM": (40.0691, 45.0382),
    "AZE": (40.1431, 47.5769),
    "BFA": (12.2383, -1.5616),
    "CAF": (6.6111, 20.9394),
    "COD": (-4.0383, 21.7587),
    "ETH": (9.145, 40.4897),
    "HTI": (18.9712, -72.2852),
    "IND": (20.5937, 78.9629),
    "IRN": (32.4279, 53.688),
    "IRQ": (33.2232, 43.6793),
    "ISR": (31.0461, 34.8516),
    "KEN": (-0.0236, 37.9062),
    "KWT": (29.3117, 47.4818),
    "LBN": (33.8547, 35.8623),
    "LBY": (26.3351, 17.2283),
    "LTU": (55.1694, 23.8813),
    "MLI": (17.5707, -3.9962),
    "MMR": (21.9162, 95.956),
    "NER": (17.6078, 8.0817),
    "NGA": (9.082, 8.6753),
    "PAK": (30.3753, 69.3451),
    "PSE": (31.9522, 35.2332),
    "ROU": (45.9432, 24.9668),
    "RUS": (61.524, 105.3188),
    "SAU": (23.8859, 45.0792),
    "SDN": (12.8628, 30.2176),
    "SOM": (5.1521, 46.1996),
    "SSD": (6.877, 31.307),
    "SYR": (34.8021, 38.9968),
    "TCD": (15.4542, 18.7322),
    "TUR": (38.9637, 35.2433),
    "UKR": (48.3794, 31.1656),
    "USA": (37.0902, -95.7129),
    "YEM": (15.5527, 48.5164),
}


@dataclass
class ExtractedEvent:
    event_id: str
    is_conflict_related: bool
    country: str | None
    location_name: str | None
    latitude: float | None
    longitude: float | None
    location_precision: str
    event_type: str
    severity: float
    confidence: float
    summary: str
    matched_keywords: list[str]


def stable_id(*parts: object) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def parse_datetime(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif value:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def infer_country(text: str) -> tuple[str | None, str | None, float | None, float | None, str]:
    lowered = text.lower()
    matches = []
    for iso3, aliases in COUNTRY_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                matches.append((alias in CITY_COORDS, len(alias), iso3, alias))

    if not matches:
        return None, None, None, None, "missing"

    _, _, best_country, best_location = sorted(matches, reverse=True)[0]

    lat = lon = None
    precision = "country"
    if best_location in CITY_COORDS:
        lat, lon = CITY_COORDS[best_location]
        precision = "city"
    elif best_country in COUNTRY_CENTROIDS:
        lat, lon = COUNTRY_CENTROIDS[best_country]
    return best_country, best_location, lat, lon, precision


def infer_event_type(matched: list[str]) -> str:
    if any(k in matched for k in ["airstrike", "air strike", "missile", "rocket", "drone"]):
        return "strike"
    if any(k in matched for k in ["shelling", "explosion"]):
        return "shelling_explosion"
    if any(k in matched for k in ["clash", "armed clash", "battle", "ambush"]):
        return "armed_clash"
    if any(k in matched for k in ["protest", "riot", "curfew"]):
        return "civil_unrest"
    if any(k in matched for k in ["troop", "evacuation"]):
        return "military_movement"
    return "conflict_signal"


def extract_event(raw_message: dict) -> ExtractedEvent:
    text = str(raw_message.get("text") or "")
    lowered = text.lower()
    matched = [k for k in CONFLICT_KEYWORDS if re.search(rf"\b{re.escape(k)}\b", lowered)]
    negated = any(re.search(pattern, lowered) for pattern in NEGATION_PATTERNS)
    country, location, lat, lon, location_precision = infer_country(text)

    keyword_score = max((CONFLICT_KEYWORDS[k] for k in matched), default=0.0)
    severity = keyword_score
    if "casualties" in matched or "killed" in matched:
        severity += 0.08
    if negated:
        severity *= 0.45
    severity = max(0.0, min(1.0, severity))

    confidence = 0.25
    if matched:
        confidence += 0.35
    if country:
        confidence += 0.20
    if raw_message.get("url"):
        confidence += 0.10
    reliability = raw_message.get("source_reliability")
    if isinstance(reliability, int | float):
        confidence += (float(reliability) - 0.5) * 0.20
    if negated:
        confidence -= 0.15
    confidence = max(0.0, min(1.0, confidence))

    is_conflict_related = bool(matched) and severity >= 0.30
    event_type = infer_event_type(matched)
    summary = summarize_text(text)
    event_id = stable_id(raw_message.get("channel"), raw_message.get("message_id"), raw_message.get("date"), text)

    return ExtractedEvent(
        event_id=event_id,
        is_conflict_related=is_conflict_related,
        country=country,
        location_name=location,
        latitude=lat,
        longitude=lon,
        location_precision=location_precision,
        event_type=event_type,
        severity=round(severity, 4),
        confidence=round(confidence, 4),
        summary=summary,
        matched_keywords=matched,
    )


def summarize_text(text: str, max_chars: int = 220) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    clean = re.sub(r"[\uFFFD]", "", clean)
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "..."
