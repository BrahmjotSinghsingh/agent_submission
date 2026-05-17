import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ─── Lookup Tables ────────────────────────────────────────────────────────────

KEY_TO_CODE: Dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

LANGUAGE_ALIASES: Dict[str, List[str]] = {
    "spanish": ["Spanish", "Latin American Spanish"],
    "english": [
        "English (USA)", "English International", "English (UK)",
        "English (Australia)", "English (India)", "English (Canada)",
        "English (South Africa)",
    ],
    "english us": ["English (USA)"],
    "english uk": ["English (UK)", "English International"],
    "french": ["French", "French (Canada)"],
    "german": ["German"],
    "dutch": ["Dutch"],
    "italian": ["Italian"],
    "portuguese": ["Portuguese", "Portuguese (Brazil)"],
    "hindi": ["Hindi"],
    "chinese": ["Chinese Simplified", "Chinese Traditional"],
    "japanese": ["Japanese"],
    "korean": ["Korean"],
}

SKILL_ALIASES: Dict[str, List[str]] = {
    ".net": [".NET", "ASP.NET", "C#"],
    "java": ["Core Java", "Spring", "Java"],
    "spring": ["Spring", "Java", "microservices"],
    "javascript": ["JavaScript", "Angular", "React", "Node"],
    "angular": ["Angular", "frontend", "front end"],
    "sql": ["SQL", "database", "relational database"],
    "aws": ["Amazon Web Services", "AWS", "cloud"],
    "docker": ["Docker", "container", "cloud-native"],
    "kubernetes": ["Kubernetes", "container orchestration"],
    "python": ["Python", "data science", "machine learning"],
    "excel": ["Excel", "spreadsheet", "Microsoft Excel"],
    "word": ["Word", "Microsoft Word"],
    "powerpoint": ["PowerPoint", "Microsoft PowerPoint"],
    "customer service": ["customer service", "contact center", "call simulation"],
    "contact center": ["contact center", "call center", "inbound calls", "chat support"],
    "sales": ["sales", "selling", "sales transformation"],
    "finance": ["financial accounting", "finance", "numerical reasoning", "statistics"],
    "healthcare": ["healthcare", "medical", "HIPAA", "patient records"],
    "safety": ["safety", "dependability", "compliance", "plant operator"],
    "leadership": ["leadership", "executive", "director", "manager"],
    "graduate": ["graduate", "entry level", "management trainee"],
    "cognitive": ["cognitive", "reasoning", "numerical", "deductive", "inductive"],
    "personality": ["personality", "behavior", "OPQ"],
    "situational judgement": ["situational judgment", "situational judgement", "scenarios", "SJT"],
}

TECH_TOKENS = [
    "python", "java", "spring", "angular", "react", "node", "sql", "aws", "docker",
    "kubernetes", "hadoop", "spark", "kafka", "hive", "hbase", "selenium", "c++", "c#",
    "html", "css", "javascript", "excel", "word", "powerpoint", "linux", "networking",
]


# ─── Utility Functions ────────────────────────────────────────────────────────

def as_clean_list(value: Any) -> List[str]:
    if isinstance(value, list):
        items = value
    elif value is None or (isinstance(value, float) and pd.isna(value)):
        items = []
    else:
        items = [x.strip() for x in str(value).split(",")]

    out, seen = [], set()
    for item in items:
        s = str(item).strip()
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def parse_duration_minutes(
    duration: Any, duration_raw: Any = ""
) -> Tuple[Optional[int], str]:
    text = f"{duration or ''} {duration_raw or ''}".strip().lower()
    if not text:
        return None, "missing"
    if "untimed" in text:
        return None, "untimed"
    if "variable" in text:
        return None, "variable"
    if "tbc" in text or "n/a" in text or text.strip() in {"-", "na"}:
        return None, "not_available"
    numbers = re.findall(r"\d+", text)
    if numbers:
        return int(numbers[0]), "numeric"
    return None, "unknown"


def test_type_from_keys(keys: List[str]) -> str:
    codes = [KEY_TO_CODE[k] for k in keys if k in KEY_TO_CODE]
    return ",".join(dict.fromkeys(codes)) if codes else "—"


def infer_tags(item: Dict[str, Any]) -> List[str]:
    blob = " ".join([
        str(item.get("name", "")),
        str(item.get("description", "")),
        " ".join(as_clean_list(item.get("keys"))),
        " ".join(as_clean_list(item.get("job_levels"))),
    ]).lower()

    tags = set()
    for canonical, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in blob:
                tags.add(canonical)
                break
    for token in TECH_TOKENS:
        if token in blob:
            tags.add(token)
    return sorted(tags)


# ─── Document Builders ────────────────────────────────────────────────────────

def join_or_dash(values: List[str]) -> str:
    return ", ".join(values) if values else "—"


def product_text(item: Dict[str, Any]) -> str:
    tags = join_or_dash(item.get("skill_tags", []))
    keys = join_or_dash(item.get("keys", []))
    job_levels = join_or_dash(item.get("job_levels", []))
    languages = join_or_dash(item.get("languages", []))
    duration = item.get("duration") or "—"
    dm = item.get("duration_minutes")
    duration_note = f"{dm} minutes" if dm is not None else item.get("duration_type", "—")

    return f"""
Product: {item.get("name", "—")}
Entity ID: {item.get("entity_id", "—")}
URL: {item.get("link", "—")}
Test Type Codes: {item.get("test_type", "—")}
Assessment Categories: {keys}
Measures / Description: {item.get("description", "—")}
Suitable Job Levels: {job_levels}
Languages: {languages}
Duration: {duration}
Duration Normalized: {duration_note}
Remote: {item.get("remote", "—")}
Adaptive: {item.get("adaptive", "—")}
Search Tags / Aliases: {tags}
""".strip()


def chroma_safe_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "entity_id": str(item.get("entity_id", "")),
        "name": str(item.get("name", "")),
        "link": str(item.get("link", "")),
        "keys": "|".join(item.get("keys", [])),
        "test_type": str(item.get("test_type", "—")),
        "job_levels": "|".join(item.get("job_levels", [])),
        "languages": "|".join(item.get("languages", [])),
        "duration": str(item.get("duration", "")),
        "duration_minutes": int(item["duration_minutes"]) if item.get("duration_minutes") is not None else -1,
        "duration_type": str(item.get("duration_type", "")),
        "adaptive": bool(item.get("is_adaptive", False)),
        "remote": bool(item.get("is_remote", False)),
        "skill_tags": "|".join(item.get("skill_tags", [])),
        "doc_type": "product",
    }


# ─── Main Loader ──────────────────────────────────────────────────────────────

def load_catalog() -> List[Dict[str, Any]]:
    """Download (or load from cache) and preprocess the SHL product catalog."""
    settings = get_settings()
    cache = Path(settings.CATALOG_CACHE_PATH)

    if not cache.exists():
        logger.info("Downloading SHL catalog from %s", settings.SHL_CATALOG_URL)
        urllib.request.urlretrieve(settings.SHL_CATALOG_URL, cache)
        logger.info("Catalog saved to %s", cache)
    else:
        logger.info("Loading catalog from cache: %s", cache)

    raw_text = cache.read_text(encoding="utf-8")
    try:
        shl_data = json.loads(raw_text)
    except json.JSONDecodeError:
        shl_data = json.loads(raw_text, strict=False)

    logger.info("Loaded %d raw SHL records", len(shl_data))
    return shl_data


def build_records(shl_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrich raw SHL records with computed fields."""
    records = []
    for item in shl_data:
        x = dict(item)
        x["job_levels"] = as_clean_list(x.get("job_levels"))
        x["languages"] = as_clean_list(x.get("languages"))
        x["keys"] = as_clean_list(x.get("keys"))
        dm, dt = parse_duration_minutes(x.get("duration"), x.get("duration_raw"))
        x["duration_minutes"] = dm
        x["duration_type"] = dt
        x["test_type"] = test_type_from_keys(x["keys"])
        x["skill_tags"] = infer_tags(x)
        x["is_adaptive"] = str(x.get("adaptive", "")).strip().lower() == "yes"
        x["is_remote"] = str(x.get("remote", "")).strip().lower() == "yes"
        records.append(x)
    logger.info("Built %d enriched records", len(records))
    return records


def build_documents(records: List[Dict[str, Any]]) -> List[Document]:
    """Convert enriched records to LangChain Documents."""
    docs = [
        Document(
            page_content=product_text(item),
            metadata=chroma_safe_metadata(item),
        )
        for item in records
    ]
    logger.info("Created %d LangChain documents", len(docs))
    return docs
