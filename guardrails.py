import re
from typing import Dict, Any, Tuple
AADHAAR_PATTERN = re.compile('\\b\\d{4}\\s?\\d{4}\\s?\\d{4}\\b')
ABHA_PATTERN = re.compile('\\b\\d{2}-\\d{4}-\\d{4}-\\d{4}\\b')
PAN_PATTERN = re.compile('\\b[A-Z]{5}[0-9]{4}[A-Z]\\b')
PHONE_PATTERN = re.compile('\\b(?:\\+91[\\-\\s]?)?[6-9]\\d{9}\\b')
PROMPT_INJECTION_KEYWORDS = ['ignore all previous instructions', 'ignore previous instructions', 'ignore all rules', 'reveal your system prompt', 'reveal system prompt', 'print system prompt', 'print your instructions', 'bypass security', 'developer mode enable', 'jailbreak']

def mask_pii(text: str) -> Tuple[str, Dict[str, int]]:
    redacted_counts = {'aadhaar': 0, 'abha_id': 0, 'pan': 0, 'phone': 0}
    abha_matches = ABHA_PATTERN.findall(text)
    if abha_matches:
        redacted_counts['abha_id'] += len(abha_matches)
        text = ABHA_PATTERN.sub('[ABHA_ID_REDACTED]', text)
    pan_matches = PAN_PATTERN.findall(text)
    if pan_matches:
        redacted_counts['pan'] += len(pan_matches)
        text = PAN_PATTERN.sub('[PAN_REDACTED]', text)
    aadhaar_matches = AADHAAR_PATTERN.findall(text)
    if aadhaar_matches:
        redacted_counts['aadhaar'] += len(aadhaar_matches)
        text = AADHAAR_PATTERN.sub('[AADHAAR_REDACTED]', text)
    phone_matches = PHONE_PATTERN.findall(text)
    if phone_matches:
        redacted_counts['phone'] += len(phone_matches)
        text = PHONE_PATTERN.sub('[PHONE_REDACTED]', text)
    return (text, redacted_counts)

def detect_prompt_injection(text: str) -> bool:
    lower_text = text.lower()
    for kw in PROMPT_INJECTION_KEYWORDS:
        if kw in lower_text:
            return True
    return False

def validate_output_groundedness(response_text: str, retrieved_context: str) -> float:
    if not retrieved_context:
        return 1.0
    context_words = set(re.findall('\\w+', retrieved_context.lower()))
    resp_words = set(re.findall('\\w+', response_text.lower()))
    if not resp_words:
        return 1.0
    common = resp_words.intersection(context_words)
    score = len(common) / len(resp_words)
    return min(1.0, round(score * 1.5, 2))