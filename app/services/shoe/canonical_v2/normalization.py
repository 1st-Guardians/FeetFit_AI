from __future__ import annotations

import re
import unicodedata

from .models import NormalizedIdentity, VariantProfile


BRAND_ALIASES = {
    "나이키": "NIKE", "NIKE": "NIKE",
    "아디다스": "ADIDAS", "ADIDAS": "ADIDAS",
    "뉴발란스": "NEW BALANCE", "NEWBALANCE": "NEW BALANCE", "NEW BALANCE": "NEW BALANCE", "NB": "NEW BALANCE",
    "아식스": "ASICS", "ASICS": "ASICS",
    "호카": "HOKA", "HOKA ONE ONE": "HOKA", "HOKA": "HOKA",
    "써코니": "SAUCONY", "SAUCONY": "SAUCONY",
    "푸마": "PUMA", "PUMA": "PUMA",
    "리복": "REEBOK", "REEBOK": "REEBOK",
    "미즈노": "MIZUNO", "MIZUNO": "MIZUNO",
    "살로몬": "SALOMON", "SALOMON": "SALOMON",
    "반스": "VANS", "VANS": "VANS",
    "브룩스": "BROOKS", "BROOKS": "BROOKS",
    "머렐": "MERRELL", "MERRELL": "MERRELL",
}

SALES_PHRASES = (
    "29CM단독", "29CM 단독", "무신사 단독", "단독", "무료배송", "본사직영",
    "공식", "정품", "특가", "할인", "쿠폰", "오늘출발", "예약배송",
)

# Only identity-bearing Korean words are translated. This is deliberately not
# a general transliterator; unknown names remain candidates for review.
MODEL_ALIASES = {
    "페가수스": "PEGASUS", "보메로": "VOMERO", "슈퍼노바": "SUPERNOVA",
    "젤 카야노": "GEL KAYANO", "젤 님버스": "GEL NIMBUS", "젤 1130": "GEL 1130",
    "클럽 C": "CLUB C", "올드스쿨": "OLD SKOOL", "웨이브": "WAVE",
}

_SKU_TOKEN = re.compile(r"(?<![A-Z0-9])(?=[A-Z0-9_-]{5,18}(?![A-Z0-9]))(?=[A-Z0-9_-]*[A-Z])(?=[A-Z0-9_-]*\d)[A-Z0-9_-]+", re.I)
_SPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^0-9A-Z가-힣]+")
COLOR_TERMS = ("BLACK", "WHITE", "BLUE", "RED", "GREEN", "GREY", "GRAY", "NAVY", "BEIGE", "BROWN", "PINK", "PURPLE", "ORANGE", "YELLOW", "SILVER", "GOLD", "블랙", "화이트", "블루", "레드", "그린", "그레이", "네이비", "베이지", "브라운", "핑크", "퍼플", "실버", "골드")


def normalize_brand(value: str | None) -> str:
    key = _SPACE.sub(" ", unicodedata.normalize("NFKC", value or "").strip()).upper()
    compact = key.replace(" ", "")
    return BRAND_ALIASES.get(key, BRAND_ALIASES.get(compact, key))


def normalize_sku(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", unicodedata.normalize("NFKC", value or "").upper())


def extract_title_skus(value: str | None) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", value or "").upper()
    result = {normalize_sku(token) for token in _SKU_TOKEN.findall(text)}
    # Exclude common non-SKU marketing and variant tokens.
    blocked = {"GORETEX", "29CM", "2COLORS", "3COLORS", "UNISEX"}
    return tuple(sorted(x for x in result if len(x) >= 5 and x not in blocked))


def _canonical_text(value: str | None, *, strip_color: bool = True) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    for source, target in MODEL_ALIASES.items():
        text = re.sub(re.escape(source), target, text, flags=re.I)
    for phrase in SALES_PHRASES:
        text = re.sub(r"\[?\s*" + re.escape(phrase) + r"\s*\]?", " ", text, flags=re.I)
    text = _SPACE.sub(" ", _NON_WORD.sub(" ", text.upper())).strip()
    if strip_color:
        for color in COLOR_TERMS:
            text = re.sub(rf"\b{re.escape(color)}\b", " ", text, flags=re.I)
    return _SPACE.sub(" ", text).strip()


def _detect(pattern: str, text: str, value: str) -> str | None:
    return value if re.search(pattern, text, flags=re.I) else None


def extract_variant(name: str | None) -> VariantProfile:
    text = _canonical_text(name, strip_color=False)
    generation_matches = re.findall(r"(?<![A-Z0-9])(\d{1,2})(?![A-Z0-9])", text)
    generation = generation_matches[-1] if generation_matches else None
    gender = None
    if re.search(r"\b(WOMEN|WOMENS|WNS|WMNS|W)\b|여성|우먼", text):
        gender = "WOMEN"
    elif re.search(r"\b(MEN|MENS|M)\b|남성|맨즈", text):
        gender = "MEN"
    elif re.search(r"\bUNISEX\b|공용", text):
        gender = "UNISEX"
    width = None
    if re.search(r"\b(4E|EXTRA WIDE|X WIDE|XWIDE)\b", text):
        width = "EXTRA_WIDE"
    elif re.search(r"\b(2E|WIDE)\b|와이드", text):
        width = "WIDE"
    waterproof = _detect(r"\b(GTX|GORE TEX|GORETEX|WATERPROOF|WP)\b|고어텍스", text, "WATERPROOF")
    terrain = "TRAIL" if re.search(r"\bTRAIL\b|트레일", text) else ("ROAD" if re.search(r"\bROAD\b|로드", text) else None)
    age = "KIDS" if re.search(r"\b(KIDS|KID|GS|PS|TD)\b|키즈|주니어", text) else None
    collaboration = "COLLAB" if re.search(r"\b(X|COLLAB|COLLABORATION)\b|협업", text) else None
    construction_terms = ("PRO", "PLUS", "ELITE", "SHIELD", "KNIT", "LEATHER", "LOW", "MID", "HIGH", "LITE", "DECON", "VINTAGE", "PREMIUM", "XLG")
    construction = tuple(term for term in construction_terms if re.search(rf"\b{term}\b", text))
    color = next((term for term in COLOR_TERMS if re.search(rf"\b{re.escape(term)}\b", text, flags=re.I)), None)
    return VariantProfile(generation, gender, width, waterproof, terrain, age, collaboration, construction, color)


def normalize_identity(
    brand: str | None,
    model_name: str | None,
    skus: tuple[str, ...] | list[str] = (),
    sku_provenance: str = "NONE",
) -> NormalizedIdentity:
    normalized_brand = normalize_brand(brand)
    text = _canonical_text(model_name)
    for alias in sorted(BRAND_ALIASES, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(alias)}\b", " ", text, flags=re.I)
    title_skus = extract_title_skus(model_name)
    for sku in title_skus:
        text = re.sub(re.escape(sku), " ", text, flags=re.I)
    model = _SPACE.sub(" ", text).strip()
    normalized_skus = {normalize_sku(sku) for sku in skus if normalize_sku(sku)}
    if not normalized_skus and title_skus:
        normalized_skus.update(title_skus)
        sku_provenance = "TITLE_EXTRACTED"
    tokens = tuple(sorted(set(model.split())))
    return NormalizedIdentity(normalized_brand, model, tokens, tuple(sorted(normalized_skus)), sku_provenance, extract_variant(model_name))
