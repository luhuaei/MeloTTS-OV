import copy
import re
from importlib import import_module

from . import cleaned_text_to_sequence

language_module_path_map = {
    "ZH": "melo.text.chinese",
    "JP": "melo.text.japanese",
    "EN": "melo.text.english",
    "ZH_MIX_EN": "melo.text.chinese_mix",
    "KR": "melo.text.korean",
    "FR": "melo.text.french",
    "SP": "melo.text.spanish",
    "ES": "melo.text.spanish",
}


def _get_language_module(language):
    module_path = language_module_path_map[language]
    return import_module(module_path)


def _g2p_with_fallback(language_module, text, language):
    try:
        return language_module.g2p(text)
    except AssertionError:
        if language != "ZH":
            raise
        # In constrained runtime environments we may disable ZH_MIX_EN path;
        # fallback to Chinese-only content for robust inference.
        text_zh_only = re.sub(r"[A-Za-z]+(?:['’][A-Za-z]+)?", "", text)
        if hasattr(language_module, "text_normalize"):
            text_zh_only = language_module.text_normalize(text_zh_only)
        text_zh_only = re.sub(r"\s+", "", text_zh_only).strip()
        return language_module.g2p(text_zh_only)


def clean_text(text, language):
    language_module = _get_language_module(language)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = _g2p_with_fallback(language_module, norm_text, language)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language, device=None):
    language_module = _get_language_module(language)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = _g2p_with_fallback(language_module, norm_text, language)
    
    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = language_module.get_bert_feature(norm_text, word2ph, device=device)
    
    return norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
