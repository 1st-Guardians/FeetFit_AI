from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.core.config import settings
from app.schemas.shoe_server import ServerUserFootState

REASON_TYPES = ("FOREFOOT", "HEEL", "INSOLE")

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|(?<=[다요])\s+|\n+")
_INVISIBLE_TEXT_PATTERN = re.compile(r"[\u200b-\u200d\u2060\ufeff]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_review_text(text: str) -> str:
    """Return a conservative fingerprint for duplicate review text.

    Punctuation and words are intentionally preserved so two genuinely different
    reviews are not merged. Unicode presentation differences, invisible
    characters, and whitespace-only differences are ignored.
    """

    normalized = unicodedata.normalize("NFKC", text)
    normalized = _INVISIBLE_TEXT_PATTERN.sub("", normalized)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip().casefold()


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_PATTERN.split(text)
    return [part.strip() for part in parts if part.strip()]


_KEYWORDS_BY_REASON: dict[str, list[str]] = {
    "FOREFOOT": [
        "발볼", "앞볼", "앞꿈치", "앞코", "엄지", "발가락", "폭이", "볼이",
        "사이즈", "정사이즈", "반업",
    ],
    "HEEL": [
        "뒤꿈치", "뒷굽", "힐", "까짐", "까졌", "미끄러", "헐렁", "뒤축", "발목",
        "쿠션", "푹신", "폭신", "딱딱", "충격", "반발", "바닥", "밑창", "미드솔",
        "장시간", "오래", "피로", "안정감", "착화감",
    ],
    "INSOLE": [
        "깔창",
        "통풍",
        "통기성",
        "열감",
        "습기",
        "땀",
        "답답",
        "쾌적",
        "인솔",
        "신발 내부",
    ],
}

_NON_FIT_TOPIC_KEYWORDS = (
    "바지",
    "밑단",
    "코디",
    "패션",
    "예뻐",
    "예쁘",
    "이뻐",
    "디자인",
    "실루엣",
    "쉐입",
    "날렵",
    "둥글",
    "로우",
    "전체 핏",
    "착화 핏",
    "전체적",
    "딱 맞",
    "색상",
    "가슴",
    "허리",
)
_EVIDENCE_BOUNDARY_WORDS = tuple(
    dict.fromkeys(
        keyword
        for keywords in _KEYWORDS_BY_REASON.values()
        for keyword in keywords
    )
) + _NON_FIT_TOPIC_KEYWORDS
_EVIDENCE_CLAUSE_SPLIT_PATTERN = re.compile(
    r"\n+|[.!?;,]+|\s+(?=(?:"
    + "|".join(re.escape(word) for word in _EVIDENCE_BOUNDARY_WORDS)
    + r"))"
)
_POINT_CLAUSE_SPLIT_PATTERN = re.compile(
    # Do not split merely because a token ends in 다/요.  Adverbs such as
    # `생각보다 편하고` otherwise become the two fragments `생각보다` and
    # `편하고`, destroying the actual wear evidence.
    r"(?<=[.!?])\s+|\n+|[;]+"
)

_POINT_SUMMARY_KEYWORDS = tuple(
    dict.fromkeys(
        keyword
        for keywords in _KEYWORDS_BY_REASON.values()
        for keyword in keywords
    )
) + (
    "사이즈",
    "정사이즈",
    "반업",
    "1업",
    "사이즈 업",
    "사이즈 다운",
    "mm",
    "cm",
    "미리",
    "밀리미터",
    "센티",
    "센티미터",
    "착화감",
    "편하",
    "편해",
    "편안",
    "무게",
    "가볍",
    "가벼",
    "무겁",
    "무거",
    "굽",
    "소재",
    "부드럽",
    "색감",
    "색상",
    "디자인",
    "실루엣",
    "쉐입",
    "날렵",
    "둥글",
    "로우",
    "전체 핏",
    "착화 핏",
    "전체적",
    "딱 맞",
    "예뻐",
    "예쁘",
    "이뻐",
    "데일리",
    "코디",
    "여기저기",
    "어디에",
    "어디든",
    "다양한 스타일",
    "잘 어울",
    "어울리",
    "청바지",
    "데님",
    "슬랙스",
    "스커트",
    "치마",
    "바지",
)
_POINT_SUMMARY_BLOCKED_KEYWORDS = (
    "가슴",
    "허리",
    "어깨",
    "치료",
    "진단",
    "질환",
    "병원",
    "시스템 지시",
    "이전 지시",
    "프롬프트",
    "명령을",
    "json을",
)

POINT_EVIDENCE_CATEGORIES = (
    "SIZE_FIT",
    "WIDTH_FIT",
    "HEEL_FEEL",
    "CUSHION_FEEL",
    "LONG_WEAR",
    "WEIGHT_FEEL",
    "DESIGN",
    "STYLING",
    "OTHER",
)

_POINT_SUBJECT_MARKERS: dict[str, tuple[str, ...]] = {
    "SIZE": (
        "한 사이즈 업",
        "반 사이즈 업",
        "반업",
        "1업",
        "정사이즈",
        "사이즈",
        "전체 핏",
        "착화 핏",
        "전체적",
    ),
    "WIDTH": ("발볼", "앞볼", "발등", "발볼러"),
    "TOEBOX": ("앞코", "발가락"),
    "HEEL": ("뒤꿈치", "뒷꿈치", "뒤축", "발목"),
    "CUSHION": ("쿠션", "깔창"),
    "SOLE": ("바닥", "밑창", "아웃솔", "깔창"),
    "LONG_WEAR": ("하루종일", "장시간", "종일", "오래"),
    "WEIGHT": ("착화 시", "신발 무게", "제품 무게", "무게", "무게감"),
    "WEAR": ("착화감",),
    "BREATHABILITY": (
        "통풍",
        "통기",
        "열감",
        "습기",
        "땀",
        "쾌적",
    ),
    "APPEARANCE": (
        "디자인",
        "색감",
        "색상",
        "외관",
        "실루엣",
        "쉐입",
        "날렵",
        "로우 프로파일",
        "낮은 실루엣",
    ),
    "STYLING": (
        "청바지",
        "데님",
        "슬랙스",
        "스커트",
        "치마",
        "통큰 바지",
        "와이드 팬츠",
        "데일리",
        "일상용",
        "코디",
        "매치",
        "여기저기",
        "어디에나",
        "어디에",
        "어디든",
        "다양한 스타일",
        "다양한 코디",
        "잘 어울",
        "어울리",
    ),
}


@dataclass(frozen=True, order=True)
class PointSignal:
    category: str
    subject: str
    value: str
    stance: str

    def __post_init__(self) -> None:
        if self.category not in POINT_EVIDENCE_CATEGORIES:
            raise ValueError(f"Unknown point evidence category: {self.category}")


_SIZE_OPTION_TOKEN_PATTERN = re.compile(
    r"(?P<FULL_UP>(?<!\d)1\s*업|한\s*사이즈\s*업)"
    r"|(?P<HALF_UP>반\s*(?:사이즈\s*)?업)"
    r"|(?P<TRUE_SIZE>정\s*\d*\s*사이즈)"
    r"|(?P<DOWN>사이즈\s*다운|사이즈다운)"
    # `10 업`, `업해서`, `업하세요` are common MUSINSA wording.  Keep
    # numeric increments generic: 10 mm must never be silently rewritten as
    # either a half-size or a full-size recommendation.
    r"|(?P<GENERIC_UP>"
    r"(?:사이즈\s*업|사이즈업)"
    r"|(?<![가-힣A-Za-z0-9반])(?:\d+\s*(?:mm|미리)?\s*(?:정도\s*)?)?"
    r"업(?:해서|하니|하면|해야|하세(?:요|욤)?|하세요|했(?:더니)?|해|을|이|으로)?"
    r")"
)
_SIZE_OPTION_GROUP_CONNECTOR_PATTERN = re.compile(
    r"\s*(?:나|이나|또는|혹은|/|,|및|와|과)\s*"
)
_SIZE_NEGATIVE_TERMS = (
    "까",
    "아프",
    "불편",
    "좁",
    "작",
    "끼",
    "눌",
    "답답",
    "힘들",
    "크",
)
_SIZE_POSITIVE_TERMS = (
    "편하",
    "편해",
    "편합",
    "편했",
    "편안",
    "적당",
    "잘 맞",
    "딱 맞",
    "좋",
    "여유",
)
_SIZE_RECOMMENDATION_PATTERN = re.compile(
    r"(?:해야|추천|고려|하세|하세요|가세|가세요|사시|신으)"
)
_SIZE_RECOMMENDATION_REJECTION_PATTERN = re.compile(
    r"(?:추천|고려|선택).{0,8}(?:별로|싫|좋지\s*않|권하지\s*않|안\s*하|하지\s*않)"
)
_WIDE_FOOT_CONTEXT_PATTERNS = (
    re.compile(r"(?:제|내)\s*발볼.{0,8}넓"),
    re.compile(r"(?:저는|제가|나는|평소)\s*발볼.{0,8}넓"),
    re.compile(
        r"발볼(?:이|은|가)?\s*(?:좀|많이|매우)?\s*넓(?:은)?\s*"
        r"(?:사람|분)(?:이|이라|인데|이고|입니다|이에요|이라서)?"
    ),
    re.compile(r"발볼(?:이|은|가)?\s*(?:있|크)(?:는|은|면|다면|신)?"),
    re.compile(r"발등(?:이|은|가)?\s*(?:있|높)(?:는|은|면|다면)?"),
    re.compile(r"발볼(?:이|은|가)?\s*넓(?:으면|다면)"),
    re.compile(r"발등(?:이|은|가)?\s*높(?:으면|다면)"),
)
_EXPLICIT_SHOE_WIDTH_PATTERN = re.compile(
    r"(?:이\s*신발|신발|제품|모델).{0,12}발볼"
)


def _term_occurrence_is_negated(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 8) : start]
    after = text[end : end + 24]
    return bool(
        re.search(r"(?:안|못|별로|전혀)\s*$", before)
        or re.match(r"(?:하|리)?지(?:는|도)?\s*(?:않|못)", after)
        or re.match(r"(?:감)?(?:이|은|는|가)?\s*(?:없|부족)", after)
        or re.match(
            r"(?:은|는|운|한)?\s*편(?:은|이)?\s*(?:아니|아닌|아닙)", after
        )
        or re.match(
            r"(?:은|는|운|한)?\s*(?:것|건)?\s*(?:아니|아닌|아닙)", after
        )
        or re.match(r"(?:이|은|는)?\s*(?:아니|아닌|아닙)", after)
        or re.match(
            r".{0,28}(?:느낌|생각|편|핏|선택|대상|수준)(?:이|은|는|도)?\s*"
            r"(?:아니|아닌|아닙|없|들지\s*(?:않|못))",
            after,
        )
        or re.match(
            r".{0,24}(?:느끼|생각하)(?:지|지는|지는)?\s*(?:않|못)",
            after,
        )
        or re.match(r".{0,24}(?:라고|다고)?\s*할\s*수\s*(?:없|없는)", after)
        or re.match(r".{0,24}(?:라고|다고|으로)?\s*보(?:기|기는)\s*어렵", after)
        or re.match(r".{0,18}(?:이라기|라기)\s*보다", after)
    )


def _has_unnegated_term(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        for match in re.finditer(re.escape(term), text):
            if not _term_occurrence_is_negated(text, match.start(), match.end()):
                return True
    return False


def _has_negated_term(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        for match in re.finditer(re.escape(term), text):
            if _term_occurrence_is_negated(text, match.start(), match.end()):
                return True
    return False


def _point_subject_contexts(text: str, subject: str) -> tuple[str, ...]:
    """Return every local predicate span for one product attribute.

    Reviews often place opposing attributes in one sentence (for example
    `발볼은 좁지만 착화감은 좋아요`).  Using the whole sentence's sentiment
    for every subject creates false consensus, so each subject stops at the
    next different typed subject marker.
    """

    occurrences: set[tuple[int, str]] = set()
    for group, markers in _POINT_SUBJECT_MARKERS.items():
        for marker in markers:
            for match in re.finditer(marker, text):
                # `바닥` describes the outsole, while the same characters in
                # `발바닥` describe a wear outcome.  Treating the latter as a
                # new SOLE subject truncates long-wear evidence mid-word.
                if marker == "바닥" and match.start() > 0 and text[match.start() - 1] == "발":
                    continue
                occurrences.add((match.start(), group))
    ordered_occurrences = sorted(occurrences)
    target_starts = sorted(
        {position for position, group in ordered_occurrences if group == subject}
    )
    if not target_starts:
        return ()
    spans: list[str] = []
    for target_start in target_starts:
        has_previous_other_subject = any(
            position < target_start and group != subject
            for position, group in ordered_occurrences
        )
        start = target_start if has_previous_other_subject else 0
        end = next(
            (
                position
                for position, group in ordered_occurrences
                if position > target_start and group != subject
            ),
            len(text),
        )
        span = text[start:end]
        if span and span not in spans:
            spans.append(span)
    return tuple(spans)


def _point_subject_context(text: str, subject: str) -> str:
    # Joining all local spans lets contradictory repeated mentions emit both
    # directions, which the aggregation layer then marks MIXED instead of
    # manufacturing a false consensus from only the first occurrence.
    return " ".join(_point_subject_contexts(text, subject))


def _size_option_contexts(text: str) -> list[tuple[str, str]]:
    """Keep each size option tied to its own predicate.

    Coordinated options such as ``정사이즈나 반업은 좁다`` share the predicate
    after the final option, while contrasting clauses such as
    ``반업은 편하고 정사이즈는 좁다`` remain separate.
    """

    matches = list(_SIZE_OPTION_TOKEN_PATTERN.finditer(text))
    contexts: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        last_grouped = index
        while last_grouped + 1 < len(matches):
            gap = text[
                matches[last_grouped].end() : matches[last_grouped + 1].start()
            ]
            if not _SIZE_OPTION_GROUP_CONNECTOR_PATTERN.fullmatch(gap):
                break
            last_grouped += 1
        context_end = (
            matches[last_grouped + 1].start()
            if last_grouped + 1 < len(matches)
            else len(text)
        )
        # A predicate belonging to another typed product attribute must not be
        # borrowed as evidence that the selected size worked.  For example,
        # ``반업했는데 디자인은 좋아요`` says nothing about half-up fit.
        typed_subject_starts = (
            marker_match.start()
            for group, markers in _POINT_SUBJECT_MARKERS.items()
            if group != "SIZE"
            for marker in markers
            for marker_match in re.finditer(re.escape(marker), text)
            if match.end() <= marker_match.start() < context_end
        )
        context_end = min(typed_subject_starts, default=context_end)
        contexts.append((match.lastgroup or "", text[match.start() : context_end]))
    return contexts


def _size_option_stance(option: str, context: str) -> str:
    if _SIZE_RECOMMENDATION_REJECTION_PATTERN.search(context):
        return "NEUTRAL"
    if _has_unnegated_term(context, _SIZE_NEGATIVE_TERMS):
        return "NEGATIVE"
    if _has_unnegated_term(context, _SIZE_POSITIVE_TERMS):
        return "POSITIVE"
    # Merely reporting a purchase/selection (``반업했는데``) is not proof that
    # the option fit well.  Only an explicit recommendation can establish a
    # positive size-option signal without a local fit predicate.
    if _SIZE_RECOMMENDATION_PATTERN.search(context) and (
        option != "TRUE_SIZE" or "추천" in context or "가세" in context
    ):
        return "POSITIVE"
    return "NEUTRAL"


def _describes_wearer_wide_foot(text: str) -> bool:
    if _EXPLICIT_SHOE_WIDTH_PATTERN.search(text):
        return False
    return any(pattern.search(text) for pattern in _WIDE_FOOT_CONTEXT_PATTERNS)

_POSITIVE_KEYWORDS = [
    "편하", "편해", "편안", "여유", "넉넉", "좋았", "좋고", "좋아요", "좋습니다",
    "시원하", "시원", "부드럽", "안정적", "쾌적", "조아", "조하", "굳", "굿",
]
_NEGATIVE_KEYWORDS = [
    "좁아", "좁다", "아팠", "아파요", "아파", "까졌", "까짐", "딱딱하", "딱딱",
    "답답하", "답답", "불편", "미끄러", "열감", "습기", "피로", "부담",
]


def bucket_review_by_reason(review_text: str) -> list[str]:
    matched = []
    for reason_type, keywords in _KEYWORDS_BY_REASON.items():
        if any(keyword in review_text for keyword in keywords):
            matched.append(reason_type)
    return matched


def extract_reason_evidence(reason_type: str, review_text: str) -> str | None:
    """Keep only clauses that explicitly support one fit reason.

    The semantic shortlist operates on individual sentences, but only review IDs
    are persisted by Feetfit_Server. When a later summary request returns the full
    review body, this function prevents unrelated fashion, appearance, or body
    anecdotes from being exposed to Ollama as fit evidence.
    """

    if reason_type not in _KEYWORDS_BY_REASON:
        raise ValueError(f"Unknown reason type: {reason_type}")

    normalized_source = unicodedata.normalize("NFKC", review_text)
    normalized_source = _INVISIBLE_TEXT_PATTERN.sub("", normalized_source)
    clauses = _EVIDENCE_CLAUSE_SPLIT_PATTERN.split(normalized_source)
    selected: list[str] = []
    seen: set[str] = set()
    for raw_clause in clauses:
        clause = _WHITESPACE_PATTERN.sub(" ", raw_clause).strip()
        if not clause or reason_type not in bucket_review_by_reason(clause):
            continue
        if any(keyword in clause for keyword in _NON_FIT_TOPIC_KEYWORDS):
            continue
        fingerprint = normalize_review_text(clause)
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(clause)
        if len(selected) >= 3:
            break
    return " ".join(selected) or None


def extract_point_summary_evidence(review_text: str) -> list[str]:
    """Extract safe fit/product/style clauses for the overall shoe description.

    Unlike area summaries, pointSummary may describe sizing, cushioning, weight,
    appearance, and outfit compatibility. The returned clauses are still
    verbatim evidence: emotional body metaphors, medical language, and prompt
    injection-like instructions are excluded before Ollama sees them.
    """

    normalized_source = unicodedata.normalize("NFKC", review_text)
    normalized_source = _INVISIBLE_TEXT_PATTERN.sub("", normalized_source)
    clauses = _POINT_CLAUSE_SPLIT_PATTERN.split(normalized_source)
    selected: list[str] = []
    seen: set[str] = set()
    for raw_clause in clauses:
        clause = _WHITESPACE_PATTERN.sub(" ", raw_clause).strip()
        normalized_clause = normalize_review_text(clause)
        if not normalized_clause:
            continue
        if any(
            keyword in normalized_clause
            for keyword in _POINT_SUMMARY_BLOCKED_KEYWORDS
        ):
            continue
        if not any(keyword in normalized_clause for keyword in _POINT_SUMMARY_KEYWORDS):
            continue
        if normalized_clause in seen:
            continue
        seen.add(normalized_clause)
        selected.append(clause)
    return selected


def classify_point_evidence(clause: str) -> tuple[PointSignal, ...]:
    """Map one safe review clause to traceable, direction-aware product facts.

    Classification is deliberately multi-label.  A sentence can support size,
    width, heel and design facts, but later aggregation still keeps every fact
    tied to this exact review instead of combining a global bag of words.
    """

    text = normalize_review_text(clause)
    signals: set[PointSignal] = set()

    def add(category: str, subject: str, value: str, stance: str) -> None:
        signals.add(PointSignal(category, subject, value, stance))

    width_text = _point_subject_context(text, "WIDTH")
    toebox_text = _point_subject_context(text, "TOEBOX")
    heel_text = _point_subject_context(text, "HEEL")
    cushion_text = _point_subject_context(text, "CUSHION")
    sole_text = _point_subject_context(text, "SOLE")
    long_wear_text = _point_subject_context(text, "LONG_WEAR")
    wear_text = _point_subject_context(text, "WEAR")
    breathability_text = _point_subject_context(text, "BREATHABILITY")
    appearance_text = _point_subject_context(text, "APPEARANCE")
    styling_text = _point_subject_context(text, "STYLING")
    size_fit_text = _point_subject_context(text, "SIZE")

    size_value_by_option = {
        "FULL_UP": "FULL_UP",
        "HALF_UP": "HALF_UP",
        "TRUE_SIZE": "TRUE_SIZE",
        "GENERIC_UP": "GENERIC_UP",
        "DOWN": "DOWN",
    }
    generic_up_has_size_context = any(
        marker in text
        for marker in (
            "사이즈",
            "발볼",
            "앞볼",
            "앞코",
            "발등",
            "발가락",
            "발볼러",
            "칼발",
        )
    ) or bool(
        re.search(
            r"\d+(?:\.\d+)?\s*(?:mm|cm|미리|밀리미터|센티|센티미터)"
            r"\s*(?:정도\s*)?(?:사이즈\s*)?업",
            text,
        )
    )
    for option, context in _size_option_contexts(text):
        if option == "GENERIC_UP" and not generic_up_has_size_context:
            # Bare `업` is ambiguous in Korean (업무/스타일 업/etc.).  Only
            # accept the shorthand when the same clause is explicitly about
            # shoe sizing or foot shape.
            continue
        add(
            "SIZE_FIT",
            "SIZE_OPTION",
            size_value_by_option[option],
            _size_option_stance(option, context),
        )
    explicit_fit_matches = _has_unnegated_term(
        text,
        (
            "핏이 잘 맞",
            "핏은 잘 맞",
            "핏도 잘 맞",
        ),
    )
    if (
        size_fit_text
        and _has_unnegated_term(size_fit_text, ("딱 맞", "밀착"))
    ) or explicit_fit_matches:
        add("SIZE_FIT", "OVERALL_FIT", "EXACT", "POSITIVE")

    width_subject = bool(width_text)
    toebox_subject = bool(toebox_text)
    width_tight_space = _has_unnegated_term(
        width_text, ("좁", "타이트", "끼", "눌", "작")
    )
    width_roomy_space = _has_unnegated_term(
        width_text, ("여유", "넉넉")
    )
    toebox_tight_space = _has_unnegated_term(
        toebox_text, ("좁", "타이트", "끼", "눌", "작")
    )
    toebox_roomy_space = _has_unnegated_term(
        toebox_text, ("여유", "넉넉")
    )
    if width_subject:
        wide_foot_context = _describes_wearer_wide_foot(width_text)
        if width_tight_space:
            add("WIDTH_FIT", "WIDTH_SPACE", "TIGHT", "NEGATIVE")
        if (
            width_roomy_space
            or (_has_unnegated_term(width_text, ("넓",)) and not wide_foot_context)
        ):
            add("WIDTH_FIT", "WIDTH_SPACE", "ROOMY", "POSITIVE")
        if wide_foot_context or any(
            marker in width_text
            for marker in (
                "발볼러",
                "발볼 발등",
                "발볼과 발등",
                "발볼이나 발등",
                "발등있는",
            )
        ):
            add("WIDTH_FIT", "FOOT_CONDITION", "WIDE_OR_HIGH_INSTEP", "NEUTRAL")
    if toebox_subject:
        if toebox_tight_space:
            add("WIDTH_FIT", "TOEBOX_SPACE", "TIGHT", "NEGATIVE")
        if toebox_roomy_space or _has_unnegated_term(toebox_text, ("넓",)):
            add("WIDTH_FIT", "TOEBOX_SPACE", "ROOMY", "POSITIVE")

    heel_subject = bool(heel_text)
    if heel_subject:
        stable_negated = bool(
            _has_negated_term(heel_text, ("안정", "잡아", "받쳐", "지지"))
            or re.search(r"(?:잡아|받쳐|지지).{0,6}지\s*(?:않|못)", heel_text)
        )
        if not stable_negated and _has_unnegated_term(
            heel_text, ("안정", "잡아", "받쳐", "지지")
        ):
            add("HEEL_FEEL", "HEEL_HOLD", "STABLE", "POSITIVE")
        if _has_unnegated_term(heel_text, ("헐렁", "들뜨", "벗겨", "미끄러")):
            add("HEEL_FEEL", "HEEL_HOLD", "LOOSE", "NEGATIVE")
        rubbing_negated = bool(
            re.search(r"(?:쓸리|마찰).{0,4}지\s*(?:않|못)", heel_text)
            or re.search(r"마찰(?:이|은)?\s*(?:없|적)", heel_text)
        )
        if not rubbing_negated and _has_unnegated_term(
            heel_text, ("까", "쓸", "마찰")
        ):
            add("HEEL_FEEL", "HEEL_COMFORT", "RUBBING", "NEGATIVE")
        elif _has_unnegated_term(heel_text, ("불편", "아프", "통증")):
            add("HEEL_FEEL", "HEEL_COMFORT", "DISCOMFORT", "NEGATIVE")
        heel_negative = _has_unnegated_term(
            heel_text, ("까", "쓸", "마찰", "불편", "아프")
        )
        if not heel_negative and _has_unnegated_term(
            heel_text, ("편하", "편해", "편합", "폭신", "부드럽")
        ):
            add("HEEL_FEEL", "HEEL_COMFORT", "COMFORTABLE", "POSITIVE")

    cushion_subject = bool(cushion_text)
    if cushion_subject:
        coordinated_neutral_cushion = bool(
            re.search(
                r"(?:푹신|폭신|부드럽).{0,16}(?:딱딱|단단).{0,12}"
                r"(?:않|아니|아닙)",
                cushion_text,
            )
        )
        soft_negated = bool(
            _has_negated_term(cushion_text, ("푹신", "폭신", "부드럽"))
            or any(marker in cushion_text for marker in ("안 푹신", "안 폭신", "안 부드럽"))
            or coordinated_neutral_cushion
        )
        if not soft_negated and _has_unnegated_term(
            cushion_text, ("푹신", "폭신", "훅신", "부드럽")
        ):
            add("CUSHION_FEEL", "CUSHION_SOFTNESS", "SOFT", "POSITIVE")
        if _has_unnegated_term(
            cushion_text,
            ("딱딱", "단단", "쿠션감 없", "쿠션이 없", "바닥감"),
        ):
            add("CUSHION_FEEL", "CUSHION_SOFTNESS", "FIRM", "NEGATIVE")
        if "충격" in cushion_text:
            cushion_negative = _has_unnegated_term(
                cushion_text, ("아프", "불편", "부담", "크")
            )
            add("CUSHION_FEEL", "SHOCK_FEEL", "HIGH" if cushion_negative else "LOW", "NEGATIVE" if cushion_negative else "POSITIVE")
        cushion_quality_negative = bool(
            re.search(
                r"쿠션(?:감)?(?:이|은|도)?\s*"
                r"(?:별로|부족|아쉽|불편|나쁘|크지\s*않)",
                cushion_text,
            )
        )
        cushion_quality_positive = bool(
            not cushion_quality_negative
            and re.search(
                r"쿠션(?:감)?(?:이|은|도)?\s*"
                r"(?:아주|정말|너무|꽤|대체로)?\s*"
                r"(?:좋|만족|편안|편하)",
                cushion_text,
            )
        )
        if cushion_quality_positive:
            add("CUSHION_FEEL", "CUSHION_QUALITY", "POSITIVE", "POSITIVE")
        elif cushion_quality_negative:
            add("CUSHION_FEEL", "CUSHION_QUALITY", "NEGATIVE", "NEGATIVE")

    insole_thickness_text = cushion_text if "깔창" in cushion_text else ""
    if (
        sole_text and _has_unnegated_term(sole_text, ("얇",))
    ) or _has_unnegated_term(insole_thickness_text, ("얇",)):
        add("CUSHION_FEEL", "SOLE_THICKNESS", "THIN", "NEGATIVE")
    if (
        sole_text and _has_unnegated_term(sole_text, ("두껍",))
    ) or _has_unnegated_term(insole_thickness_text, ("두껍",)):
        add("CUSHION_FEEL", "SOLE_THICKNESS", "THICK", "POSITIVE")

    duration_scope = long_wear_text or text
    duration_is_negated = bool(
        re.search(
            r"(?:오래|장시간|종일|하루\s*종일).{0,8}"
            r"(?:신|착용|걷|걸(?:으면|을\s*때|어도)|서\s*있|활동)"
            r"(?:지)?\s*(?:않|못)",
            duration_scope,
        )
        or re.search(
            r"(?:신|착용|걷|서\s*있|활동)(?:지)?\s*(?:않|못).{0,8}"
            r"(?:오래|장시간|종일|하루\s*종일)",
            duration_scope,
        )
    )
    duration = not duration_is_negated and bool(
        re.search(
            r"(?:오래|장시간|종일|하루\s*종일).{0,10}"
            r"(?:신|착용|걷|걸(?:으면|을\s*때|어도)|서\s*있|활동)",
            duration_scope,
        )
        or re.search(
            r"(?:신|착용|걷|걸(?:으면|을\s*때|어도)|서\s*있|활동).{0,10}"
            r"(?:오래|장시간|종일|하루\s*종일)",
            duration_scope,
        )
    )
    if duration and _has_unnegated_term(
        duration_scope, ("아프", "아파", "피로", "불편", "부담", "힘들")
    ):
        add("LONG_WEAR", "LONG_WEAR_COMFORT", "DISCOMFORT", "NEGATIVE")
    elif duration and _has_unnegated_term(
        duration_scope, ("편하", "편해", "편합", "편했", "편안", "괜찮")
    ):
        add("LONG_WEAR", "LONG_WEAR_COMFORT", "COMFORTABLE", "POSITIVE")

    if wear_text:
        # `불편해요` contains the positive-looking substring `편해`.  An
        # explicit negative predicate must therefore win before the generic
        # positive check.
        wear_contexts = _point_subject_contexts(text, "WEAR")
        wear_negative = any(
            _has_unnegated_term(context, ("아프", "불편", "답답", "힘들"))
            for context in wear_contexts
        )
        wear_positive = any(
            not _has_unnegated_term(context, ("아프", "불편", "답답", "힘들"))
            and _has_unnegated_term(
                context, ("편하", "편해", "편합", "편했", "편안", "좋", "만족")
            )
            for context in wear_contexts
        )
        if wear_negative:
            add("OTHER", "WEAR_COMFORT", "DISCOMFORT", "NEGATIVE")
        if wear_positive:
            add("OTHER", "WEAR_COMFORT", "COMFORTABLE", "POSITIVE")

    # Buyers much more often write "신어보니 편하다" or "발이 편하다"
    # than the noun "착화감".  Keep the predicate tied to an explicit wearing
    # subject so styling phrases such as "편하게 코디" cannot become comfort
    # evidence. Consensus is still required by the aggregation layer.
    if not wear_text:
        explicit_subject_comfort = re.search(
                r"(?:신발(?:이|은|도)?|발(?:이|은|도)?|"
                r"신어\s*보(?:니|면)|신(?:고|으면|어도|었더니|었는데)|"
                r"착용(?:하면|해\s*보니|했더니)).{0,18}"
                r"(?:편하|편해|편안|불편|아프|답답|피로|부담)",
                text,
            )
        standalone_comfort = re.search(
            r"(?:^|\s)편(?:해요|합니다|하네요|안해요|안합니다)(?:\s|[.!?]|$)",
            text,
        )
        standalone_has_local_typed_subject = bool(
            standalone_comfort
            and re.search(
                r"(?:발볼|앞볼|앞코|발가락|사이즈|뒤꿈치|뒷꿈치|뒤축|"
                r"발목|쿠션|깔창|바닥|밑창|통풍|통기|열감|습기)",
                text[max(0, standalone_comfort.start() - 18) : standalone_comfort.start()],
            )
        )
        explicit_wear_context = bool(
            explicit_subject_comfort
            or (standalone_comfort and not standalone_has_local_typed_subject)
        )
        if explicit_wear_context:
            generic_wear_negative = _has_unnegated_term(
                text, ("아프", "불편", "답답", "힘들", "피로", "부담")
            )
            generic_wear_positive = (
                not generic_wear_negative
                and _has_unnegated_term(
                    text, ("편하", "편해", "편합", "편했", "편안")
                )
            )
            if generic_wear_negative:
                add("OTHER", "WEAR_COMFORT", "DISCOMFORT", "NEGATIVE")
            elif generic_wear_positive:
                add("OTHER", "WEAR_COMFORT", "COMFORTABLE", "POSITIVE")

    if breathability_text:
        breathability_negative = bool(
            _has_unnegated_term(
                breathability_text,
                ("답답", "덥", "열감", "습기", "축축", "땀차", "땀이 차"),
            )
            or re.search(
                r"(?:통풍|통기)(?:이|가|성(?:이|은)?)?.{0,8}"
                r"(?:안\s*되|별로|부족|나쁘)",
                breathability_text,
            )
        )
        breathability_positive = bool(
            not breathability_negative
            and (
                _has_unnegated_term(
                    breathability_text,
                    ("시원", "쾌적", "통풍", "통기"),
                )
                and not re.search(
                    r"(?:통풍|통기)(?:이|가|성(?:이|은)?)?.{0,8}"
                    r"(?:안\s*되|별로|부족|나쁘)",
                    breathability_text,
                )
            )
        )
        if breathability_negative:
            add("OTHER", "BREATHABILITY", "LOW", "NEGATIVE")
        elif breathability_positive:
            add("OTHER", "BREATHABILITY", "HIGH", "POSITIVE")

    heavy_is_negated = _has_negated_term(text, ("무게감", "무겁", "무거")) or bool(
        re.search(r"무게감(?:이|은|도)?\s*(?:없|적)", text)
    )
    # ``쿠션에서 무게감이 느껴진다`` is not an assertion about the shoe's
    # physical weight.  Keep that ambiguous cushion wording out of WEIGHT so it
    # can never be rewritten as either ``무겁다`` or ``가볍다`` downstream.
    def has_physical_weight_predicate(terms: tuple[str, ...]) -> bool:
        nonphysical_subjects = (
            "쿠션",
            "쿠션감",
            "깔창",
            "바닥",
            "디자인",
            "색감",
            "색상",
            "실루엣",
            "외관",
        )
        physical_subjects = ("신발", "제품", "착화", "신었", "들었", "무게")
        for term in terms:
            for match in re.finditer(re.escape(term), text):
                if _term_occurrence_is_negated(text, match.start(), match.end()):
                    continue
                prefix = text[max(0, match.start() - 28) : match.start()]
                suffix = text[match.end() : match.end() + 24]
                if re.match(
                    r"(?:운|운\s*느낌의|게\s*느껴지는)?\s*"
                    r"(?:쿠션(?:감)?|깔창|바닥|디자인|색감|색상|실루엣|외관)",
                    suffix,
                ):
                    continue
                if term == "무게감":
                    if re.search(
                        r"(?:쿠션(?:감)?|깔창|바닥|디자인|색감|색상|실루엣|외관)"
                        r"(?:에서|에|의|이\s*주는|이\s*느껴지는)\s*(?:다소\s*)?$",
                        prefix,
                    ):
                        continue
                    return True
                last_nonphysical = max(
                    (prefix.rfind(subject) for subject in nonphysical_subjects),
                    default=-1,
                )
                last_physical = max(
                    (prefix.rfind(subject) for subject in physical_subjects),
                    default=-1,
                )
                if last_nonphysical > last_physical:
                    continue
                return True
        return False
    if (
        not heavy_is_negated
        and has_physical_weight_predicate(("무게감", "무겁", "무거"))
    ):
        add("WEIGHT_FEEL", "WEIGHT", "HEAVY", "NEGATIVE")
    if has_physical_weight_predicate(("가볍", "가벼")):
        add("WEIGHT_FEEL", "WEIGHT", "LIGHT", "POSITIVE")

    appearance = bool(appearance_text) or any(
        marker in text for marker in ("예쁘", "예뻐", "예쁩", "이뻐")
    )
    if appearance:
        appearance_contexts = _point_subject_contexts(text, "APPEARANCE") or (
            appearance_text or text,
        )
        appearance_negative = any(
            _has_negated_term(
                context,
                ("예쁘", "예뻐", "예쁩", "이뻐", "좋", "깔끔", "만족"),
            )
            or any(
                marker in context
                for marker in (
                    "디자인이 별로",
                    "디자인은 별로",
                    "색감이 별로",
                    "색감은 별로",
                    "색상은 별로",
                    "외관은 별로",
                    "마음에 안",
                    "마음에 들지 않",
                )
            )
            for context in appearance_contexts
        )
        appearance_positive = any(
            _has_unnegated_term(
                context,
                (
                    "예쁘",
                    "예뻐",
                    "예쁩",
                    "이뻐",
                    "좋",
                    "깔끔",
                    "만족",
                    "마음에 들",
                ),
            )
            for context in appearance_contexts
        )
        if appearance_positive:
            add("DESIGN", "APPEARANCE", "POSITIVE", "POSITIVE")
        if appearance_negative:
            add("DESIGN", "APPEARANCE", "NEGATIVE", "NEGATIVE")
        if not appearance_positive and not appearance_negative:
            add("DESIGN", "APPEARANCE", "MENTIONED", "NEUTRAL")
    if _has_unnegated_term(text, ("둥글",)):
        add("DESIGN", "SILHOUETTE", "ROUNDED", "NEUTRAL")
    if _has_unnegated_term(text, ("날렵", "슬림한 실루엣")):
        add("DESIGN", "SILHOUETTE", "SLIM", "NEUTRAL")
    profile_negated = bool(
        _has_negated_term(text, ("로우 프로파일", "로우", "낮은 굽", "굽은 낮"))
        or re.search(r"로우.{0,18}(?:아니|아닌|아닙|않|못)", text)
    )
    if not profile_negated and _has_unnegated_term(
        text, ("로우 프로파일", "로우", "낮은 굽", "굽은 낮")
    ):
        add("DESIGN", "PROFILE", "LOW", "NEUTRAL")

    styling_values = {
        "JEANS": ("청바지", "데님"),
        "SLACKS": ("슬랙스",),
        "SKIRT": ("스커트", "치마"),
        "WIDE_PANTS": ("통큰 바지", "와이드 팬츠"),
        "DAILY": ("데일리", "일상용"),
    }
    negative_pairing = bool(
        _has_negated_term(styling_text, ("어울", "좋", "쉽"))
        or re.search(r"어울리?지\s*(?:않|못)", styling_text)
        or re.search(r"(?:코디|매치|활용|신기).{0,8}(?:어렵|좋지 않|쉽지 않)", styling_text)
    )
    positive_pairing = not negative_pairing and _has_unnegated_term(
        styling_text, ("잘 어울", "잘어울", "어울")
    )
    explicit_styling_positive = bool(
        positive_pairing
        or (
            re.search(r"(?:코디|매치|활용).{0,8}(?:좋|쉽)", styling_text)
            and _has_unnegated_term(styling_text, ("좋", "쉽"))
        )
        or (
            re.search(r"(?:신기|신어|착용하기).{0,8}좋", styling_text)
            and _has_unnegated_term(styling_text, ("좋",))
        )
    )
    mentioned_styling_values: set[str] = set()
    for value, markers in styling_values.items():
        if any(marker in styling_text for marker in markers):
            mentioned_styling_values.add(value)
            outfit_stance = (
                "NEGATIVE"
                if negative_pairing
                else ("POSITIVE" if explicit_styling_positive else "NEUTRAL")
            )
            add("STYLING", "OUTFIT", value, outfit_stance)
    generic_styling_context = any(
        marker in styling_text
        for marker in (
            "여기저기",
            "어디에나",
            "어디든",
            "다양한 코디",
            "다양한 스타일",
            "여러 코디",
            "코디하기",
            "스타일내기",
        )
    )
    if (
        explicit_styling_positive
        and (generic_styling_context or (positive_pairing and not mentioned_styling_values))
        and not negative_pairing
    ):
        add("STYLING", "OUTFIT", "GENERAL", "POSITIVE")

    if not signals:
        if "소재" in text:
            add("OTHER", "MATERIAL_DETAIL", "MENTIONED", "NEUTRAL")
        if "굽" in text:
            add("OTHER", "HEEL_HEIGHT_DETAIL", "MENTIONED", "NEUTRAL")
        if "착화감" in text:
            add("OTHER", "PRODUCT_DETAIL", "MENTIONED", "NEUTRAL")
    return tuple(sorted(signals))


def polarity_score(review_text: str) -> int:
    score = 0
    score += sum(1 for keyword in _POSITIVE_KEYWORDS if keyword in review_text)
    score -= sum(1 for keyword in _NEGATIVE_KEYWORDS if keyword in review_text)
    return score


@dataclass(frozen=True)
class FootNeed:
    wide_forefoot_needed: bool
    forefoot_note: str
    heel_instability_risk: bool
    heel_note: str
    breathability_needed: bool
    insole_note: str


def _safe(value: float | None, default: float = 0.0) -> float:
    return default if value is None else value


def derive_foot_need(state: ServerUserFootState) -> FootNeed:
    daily = state.daily_foot_analysis
    tinea = state.tina_pedis_analysis
    hallux = state.hallux_valgus_analysis

    left_width = _safe(daily.left_foot_width_mm) if daily else 0.0
    right_width = _safe(daily.right_foot_width_mm) if daily else 0.0
    left_length = _safe(daily.measured_left_foot_size_mm) if daily else 0.0
    right_length = _safe(daily.measured_right_foot_size_mm) if daily else 0.0
    width_length_ratio = 0.0
    if left_length and right_length:
        width_length_ratio = ((left_width / left_length) + (right_width / right_length)) / 2

    left_angle = _safe(hallux.left_toe_angle_degree) if hallux else 0.0
    right_angle = _safe(hallux.right_toe_angle_degree) if hallux else 0.0
    max_hallux_angle = max(left_angle, right_angle)

    wide_forefoot_needed = (
        width_length_ratio >= settings.shoe_forefoot_width_ratio_neutral
        or max_hallux_angle >= settings.shoe_hallux_angle_neutral_degree
    )
    forefoot_note_parts = []
    if width_length_ratio >= settings.shoe_forefoot_width_ratio_neutral:
        forefoot_note_parts.append("발볼이 넓은 편")
    if max_hallux_angle >= settings.shoe_hallux_angle_neutral_degree:
        forefoot_note_parts.append(f"무지외반 각도가 {max_hallux_angle:.1f}도로 앞코 압박에 주의가 필요")
    forefoot_note = ", ".join(forefoot_note_parts) or "발볼과 앞코 압박이 크지 않은 평이한 발 상태"

    pressure_imbalances: list[float] = []
    if (
        daily is not None
        and daily.left_pressure_percent is not None
        and daily.right_pressure_percent is not None
    ):
        pressure_imbalances.append(
            abs(daily.left_pressure_percent - daily.right_pressure_percent)
        )
    pressure_imbalances.extend(
        abs(row.left_pressure_ratio - row.right_pressure_ratio)
        for row in state.static_pressure_analyses
        if row.left_pressure_ratio is not None and row.right_pressure_ratio is not None
    )
    pressure_imbalance = max(pressure_imbalances, default=0.0)
    balance_values = [
        value
        for value in (
            daily.balance_score if daily else None,
            *(row.balance_score for row in state.static_pressure_analyses),
        )
        if value is not None
    ]
    balance_score = min(balance_values, default=100.0)
    rearfoot_pressure = max(
        (
            row.rearfoot_pressure_ratio
            for row in state.static_pressure_analyses
            if row.rearfoot_pressure_ratio is not None
        ),
        default=0.0,
    )
    heel_instability_risk = (
        pressure_imbalance >= settings.shoe_pressure_imbalance_neutral_percent
        or balance_score < settings.shoe_balance_neutral_score
        or rearfoot_pressure >= settings.shoe_rearfoot_pressure_neutral_percent
    )
    heel_note_parts = []
    if pressure_imbalance >= settings.shoe_pressure_imbalance_neutral_percent:
        heel_note_parts.append(f"좌우 압력 차이가 {pressure_imbalance:.1f}%p")
    if balance_score < settings.shoe_balance_neutral_score:
        heel_note_parts.append(f"자세 균형 점수가 {balance_score:.0f}점으로 낮은 편")
    if rearfoot_pressure >= settings.shoe_rearfoot_pressure_neutral_percent:
        heel_note_parts.append(f"뒤꿈치 영역 하중 비율이 {rearfoot_pressure:.1f}%로 높은 편")
    heel_note = ", ".join(heel_note_parts) or "좌우 압력과 자세 균형이 안정적인 편"

    humidity = _safe(daily.avg_humidity_percent) if daily else 0.0
    fungal_score = _safe(tinea.fungal_suspicion_safety_score, 100.0) if tinea else 100.0
    skin_reaction_score = _safe(tinea.skin_reaction_safety_score, 100.0) if tinea else 100.0

    breathability_needed = (
        humidity >= settings.shoe_humidity_neutral_percent
        or fungal_score < settings.shoe_fungal_safety_neutral_score
        or skin_reaction_score < settings.shoe_skin_safety_neutral_score
    )
    insole_note_parts = []
    if humidity >= settings.shoe_humidity_neutral_percent:
        insole_note_parts.append(f"평균 습도가 {humidity:.0f}%로 높은 편")
    if fungal_score < settings.shoe_fungal_safety_neutral_score:
        insole_note_parts.append("무좀 의심 신호가 있어 통기성 관리가 필요")
    if skin_reaction_score < settings.shoe_skin_safety_neutral_score:
        insole_note_parts.append("피부 반응 지표상 내부 환경 관리가 필요")
    insole_note = ", ".join(insole_note_parts) or "발 환경(습도)이 양호한 편"

    return FootNeed(
        wide_forefoot_needed=wide_forefoot_needed,
        forefoot_note=forefoot_note,
        heel_instability_risk=heel_instability_risk,
        heel_note=heel_note,
        breathability_needed=breathability_needed,
        insole_note=insole_note,
    )


def build_need_query(reason_type: str, need: FootNeed) -> str:
    if reason_type == "FOREFOOT":
        requirement = (
            "발볼과 앞코에 여유가 있고 압박이 적은 신발이 필요하다"
            if need.wide_forefoot_needed
            else "발볼이 넉넉하지 않아도 되는 발 상태다"
        )
        return (
            f"{need.forefoot_note}. {requirement}. 무신사 실제 착화 리뷰에서 발볼/앞코가 "
            "넓고 편안하거나, 반대로 좁아 아프고 발가락이 눌리는 경험과 정사이즈·"
            "사이즈 업·다운 조언을 모두 찾는다."
        )
    if reason_type == "HEEL":
        requirement = (
            "뒤꿈치 영역 하중과 좌우 균형을 고려해 충격 부담을 덜어주는 신발이 필요하다"
            if need.heel_instability_risk
            else "뒤꿈치 영역 하중과 좌우 균형의 추가 부담이 크지 않은 상태다"
        )
        return (
            f"{need.heel_note}. {requirement}. 무신사 실제 착화 리뷰에서 쿠션감, 바닥 충격, "
            "장시간 착용 시 발 부담, 안정감과 착화감에 관한 긍정·부정 경험을 모두 찾는다. "
            "사용자의 뒤꿈치 움직임이나 고정감으로 해석하지 않는다."
        )
    if reason_type == "INSOLE":
        requirement = (
            "통기성이 좋고 내부 습기와 열감을 관리하기 쉬운 신발이 필요하다"
            if need.breathability_needed
            else "깔창 통기성에 큰 제약은 없는 발 상태다"
        )
        return (
            f"{need.insole_note}. {requirement}. 무신사 실제 착화 리뷰에서 통풍, 열감, 습기, "
            "깔창과 신발 내부의 쾌적함에 관한 긍정·부정 경험을 모두 찾는다."
        )
    raise ValueError(f"Unknown reason type: {reason_type}")


def default_title(reason_type: str, risk_level: str) -> str:
    # Keep titles deterministic and area-level. A low aggregate score can be
    # caused by any of the available RunRepeat components (for example shock
    # absorption rather than breathability), so a component-specific title
    # would be capable of contradicting the quantitative evidence.
    titles = {
        ("FOREFOOT", "LOW"): "발볼 적합도 좋음",
        ("FOREFOOT", "MEDIUM"): "발볼 적합도 보통",
        ("FOREFOOT", "HIGH"): "발볼 적합도 주의",
        ("HEEL", "LOW"): "뒤꿈치 적합도 좋음",
        ("HEEL", "MEDIUM"): "뒤꿈치 적합도 보통",
        ("HEEL", "HIGH"): "뒤꿈치 적합도 주의",
        ("INSOLE", "LOW"): "깔창 적합도 좋음",
        ("INSOLE", "MEDIUM"): "깔창 적합도 보통",
        ("INSOLE", "HIGH"): "깔창 적합도 주의",
    }
    return titles[(reason_type, risk_level)]
