#!/usr/bin/env python3
"""
Synthetic Comparative-Reasoning Dataset Generator for Hindi and Nepali (Phase 3)

Every sample is produced procedurally from templates, so every label is exact.

Sample format (one JSON object per line):
    text      : "प्रश्न: <introduction + premises + question>\\nतर्क: <reasoning>\\nउत्तर: <answer>"
    question  : the "प्रश्न: ..." line - the only part the model sees at test time
    reasoning : the "तर्क: ..." line the model has to produce itself
    answer    : single-word gold answer (an entity name, yes/no word, or "बराबर")
    candidates: every answer the question admits (its entities, or yes/no, plus "बराबर" where
                equality is possible), used for chance levels and likelihood-ranking baselines
    task      : transitive | numeric_extremum | numeric_pair | verbal_comparison

Task families:
  1. transitive        - 3-4 people with a hidden total order, given only as shuffled pairwise
                         premises (each phrased as "more", "less" or an adjective). Asks for the
                         largest, the smallest, or a yes/no question about a pair that is never
                         stated directly and has to be chained (A > B, B > C => A ? C).
  2. numeric_extremum  - 3 people or objects with explicit values; which is largest / smallest.
  3. numeric_pair      - 2 people or objects with explicit values (equal 1/3 of the time):
                         which is greater / smaller (or "बराबर"), or a yes/no greater / smaller /
                         equal question.
  4. verbal_comparison - one premise in any of several phrasings; a yes/no question whose entity
                         order and comparison direction are randomised independently.

Every question opens by introducing its entities ("अमित, रोहित और सुमित तीन दोस्त हैं।") in a
random order. Besides reading naturally, this gives the model each name as a standalone word:
in Nepali most later mentions are fused with a postposition (रामको, श्यामभन्दा), which the BPE
tokenizer merges into different tokens than the bare name. The reasoning line never puts
punctuation directly after a name for the same reason.

The reasoning is written as a chain of small local steps so a ~25M model can follow it:
  - each premise is first transcribed symbolically in the order its names are mentioned
    ("B की लंबाई A से कम है" -> "B < A"), and only then normalised to "A > B";
  - numeric values are listed value-first ("58 = स्केल"), so the step that names the winner
    ("सबसे अधिक मान 58 = स्केल") is a forward copy of what follows "58 =" in the list;
  - two numbers are compared in the order they were stated before the conclusion is
    restated in the order the question asks about;
  - a yes/no question is transcribed as a claim ("प्रश्न का दावा X < Y है") right after the
    concluded relation, so the answer only depends on whether the two operators match;
  - a "which is more / less" question gets its conclusion larger-first ("L > S"), so the answer
    is the first name for "more" and the second for "less".

Shortcut control: entity order, premise order, premise phrasing direction, question direction and
yes/no labels are all randomised, so the answer cannot be read off a position or a keyword.
Leakage control: person names and object names are split into disjoint train / val / test pools
(held-out entities), and every sample text is unique across all three splits.
"""

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

Entity = Tuple[str, Optional[str]]  # (surface name, "m" / "f" for people, None for objects)


# =====================================================================
# ENTITY BANKS (split into disjoint train / val / test pools at runtime)
# =====================================================================

HINDI_ENTITIES = {
    "male": ["अमित", "रोहित", "सुमित", "करण", "राहुल", "विकास", "आदित्य", "सौरव", "आकाश", "गौरव",
             "मयंक", "प्रशांत", "वरुण", "नितिन", "हर्ष", "समीर", "आर्यन", "विवेक", "अजय", "विजय",
             "संजय", "मनोज", "सुनील", "अनिल", "राजेश", "सुरेश", "महेश", "रमेश", "दिनेश", "अभिषेक",
             "अंकित", "अरुण", "अशोक", "भरत", "चंदन", "धर्मेंद्र", "गोपाल", "हरीश", "जितेंद्र", "कमल",
             "कुणाल", "लोकेश", "मोहन", "नरेश", "पंकज", "प्रदीप", "राकेश", "सचिन", "तरुण", "उमेश",
             "विनोद", "यश", "शिवम", "रवि", "सत्यम", "अनुराग", "दीपक", "मुकेश", "ललित", "निखिल",
             "आलोक", "अमन", "अंकुर", "अनुज", "अर्जुन", "आशीष", "अतुल", "बलराम", "भूपेंद्र", "बृजेश",
             "चेतन", "देवेंद्र", "धीरज", "दिलीप", "गणेश", "गिरीश", "गोविंद", "हेमंत", "हिमांशु", "जगदीश",
             "जयंत", "जीवन", "कपिल", "कार्तिक", "किशोर", "कृष्ण", "कुलदीप", "लक्ष्मण", "मदन", "महावीर",
             "मनीष", "मिथुन", "नकुल", "नमन", "नवीन", "नीरज", "ओम", "पवन", "पीयूष", "प्रकाश",
             "प्रमोद", "प्रवीण", "प्रेम", "पुनीत", "राघव", "राजीव", "रजनीश", "रणवीर", "ऋषभ", "रोहन",
             "संदीप", "संतोष", "सौरभ", "शंकर", "शेखर", "श्याम", "सिद्धार्थ", "सोहन", "सुधीर", "सूरज",
             "तनुज", "तेजस", "उदय", "उत्कर्ष", "वैभव", "विनय", "विपिन", "विशाल", "योगेश", "युवराज",
             "जतिन", "ध्रुव", "गगन", "हरि", "ईशान", "कैलाश", "केशव", "माधव", "मुरली", "नरेंद्र",
             "निर्मल", "पारस", "रघु", "रमन", "सागर", "समर", "सुशील", "तपन", "वीरेंद्र"],
    "female": ["सीमा", "पूजा", "नेहा", "प्रिया", "अंजलि", "कविता", "रेखा", "स्नेहा", "रितु", "मोनिका",
               "शालिनी", "दिव्या", "पायल", "तन्वी", "भावना", "स्वाति", "मेघा", "ईशा", "श्रुति", "आरती",
               "अनीता", "बबीता", "चित्रा", "दीपिका", "गीता", "हेमा", "इंदु", "जया", "कोमल", "लता",
               "माधुरी", "नीलम", "पल्लवी", "राधा", "सरिता", "सुनीता", "तृप्ति", "उषा", "वंदना", "ज्योति",
               "किरण", "ममता", "निशा", "प्रीति", "रश्मि", "संगीता", "सोनिया", "सुधा", "विनीता", "अर्चना",
               "बीना", "कंचन", "मीना", "नम्रता", "पुष्पा", "रूपा", "शिल्पा", "सपना", "सुमन", "वर्षा",
               "आशा", "अदिति", "अमृता", "अनामिका", "अनुष्का", "अपर्णा", "भारती", "चंचल", "चारु", "डिंपल",
               "दुर्गा", "एकता", "गरिमा", "गायत्री", "हर्षा", "हिना", "इंदिरा", "जानकी", "जूही", "कल्पना",
               "कामिनी", "करिश्मा", "कीर्ति", "कुसुम", "लक्ष्मी", "मालती", "मानसी", "मंजू", "मधु", "मीरा",
               "मोहिनी", "नंदिनी", "नयना", "निधि", "निर्मला", "नूतन", "पद्मा", "पार्वती", "पूनम", "प्रतिभा",
               "प्रज्ञा", "राखी", "रंजना", "रत्ना", "रीना", "रोशनी", "रुचि", "साधना", "सलोनी", "संध्या",
               "सारिका", "सरला", "शकुंतला", "शारदा", "शीला", "शोभा", "श्वेता", "सीता", "सोनाली", "सुलोचना",
               "सुषमा", "स्मिता", "सुरभि", "तारा", "तनु", "उर्मिला", "वैशाली", "वसुधा", "विद्या", "यामिनी",
               "अलका", "अंबिका", "बिंदु", "चंदा", "देविका", "गौरी", "हेमलता", "इला", "जमुना", "कांता",
               "लीला", "मृदुला", "नीता", "सावित्री", "तुलसी", "उमा", "वीणा", "यशोदा"],
    "objects": ["किताब", "कलम", "कुर्सी", "मेज़", "मोबाइल", "लैपटॉप", "घड़ी", "बैग", "जूता", "साइकिल",
                "टेलीविज़न", "पंखा", "कैमरा", "चश्मा", "बोतल", "छाता", "रेडियो", "अलमारी", "गिटार", "थैला",
                "कंबल", "तकिया", "गद्दा", "बाल्टी", "लोटा", "थाली", "कटोरी", "चम्मच", "गिलास", "प्याला",
                "केतली", "तवा", "कड़ाही", "झाड़ू", "आईना", "कंघी", "साबुन", "तौलिया", "रज़ाई", "चादर",
                "टोपी", "कमीज़", "घड़ा", "टोकरी", "ताला", "चाबी", "हारमोनियम", "सितार", "पेंसिल", "रबर",
                "स्केल", "कॉपी", "डायरी", "लिफ़ाफ़ा", "तस्वीर", "फूलदान", "दीया", "मोमबत्ती", "लालटेन", "टॉर्च",
                "बैटरी", "चार्जर", "हेडफ़ोन", "कीबोर्ड", "माउस", "प्रिंटर", "स्टूल", "चटाई", "दरी", "परदा",
                "गुलदस्ता", "खिलौना", "गेंद", "बल्ला", "पतंग", "ढोलक", "बांसुरी", "तबला", "थर्मस", "टिफ़िन",
                "कुकर", "मिक्सी", "इस्त्री", "हथौड़ा", "पेचकस", "कैंची", "सुई"],
}

NEPALI_ENTITIES = {
    "male": ["राम", "श्याम", "हरि", "गोपाल", "किशन", "अर्जुन", "विक्रम", "प्रदीप", "राजेश", "दिनेश",
             "महेश", "सुरेश", "नवीन", "सञ्जय", "दीपक", "मनिष", "बिकास", "सुदीप", "रमेश", "गणेश",
             "कृष्ण", "नारायण", "प्रकाश", "बिनोद", "सन्तोष", "रोशन", "सुमन", "अनिल", "सुनिल", "बिजय",
             "रबिन", "सागर", "आशिष", "निराजन", "प्रविन", "उमेश", "कमल", "युवराज", "भीम", "लक्ष्मण",
             "पुष्कर", "शेखर", "तिलक", "बलराम", "चन्द्र", "दिपेन्द्र", "गोविन्द", "हिमाल", "ईश्वर", "जनक",
             "केशव", "मदन", "नरेन्द्र", "ओम", "पवन", "राजु", "शिव", "टंक", "उद्धव", "यम",
             "अजय", "अमर", "अनुप", "अरुण", "अशोक", "बाबुराम", "बद्री", "भरत", "भुपेन्द्र", "बिपिन",
             "बिरेन्द्र", "बिष्णु", "चेतन", "दामोदर", "देवेन्द्र", "धर्मराज", "दिलिप", "गगन", "गिरिराज", "गोपी",
             "हेमन्त", "हिक्मत", "इन्द्र", "जीवन", "ज्ञानेन्द्र", "कपिल", "कर्ण", "खगेन्द्र", "कुमार", "लोकेन्द्र",
             "मधुसूदन", "महेन्द्र", "मनोज", "मिलन", "मुकेश", "नरेश", "निरज", "निर्मल", "पदम", "पङ्कज",
             "पारस", "पुरुषोत्तम", "रघु", "राजन", "राजेन्द्र", "रामचन्द्र", "रविन्द्र", "ऋषि", "रुपेश", "सचिन",
             "सन्दीप", "सरोज", "शंकर", "शरद", "शिशिर", "सोमनाथ", "सुबास", "सुधिर", "सुजन", "सुरज",
             "सुशील", "तेज", "त्रिभुवन", "उत्तम", "उदय", "उपेन्द्र", "विनय", "विवेक", "योगेन्द्र", "बसन्त",
             "चिरञ्जीवी", "देवराज", "हर्क", "जगत", "मणि", "नन्द", "शम्भु", "टेक", "अमृत", "बिमल",
             "गौतम", "हरिहर", "कुशल", "लव", "मोहन", "प्रेम", "रोहित", "समीर"],
    "female": ["गीता", "सीता", "रीता", "सरिता", "अनिता", "सुनिता", "बिन्दु", "कमला", "पार्वती", "सपना",
               "पुष्पा", "माया", "शोभा", "सृजना", "मञ्जु", "सुषमा", "रञ्जना", "संगीता", "कल्पना", "रश्मी",
               "दुर्गा", "लक्ष्मी", "अञ्जना", "बिमला", "चन्द्रा", "देवी", "गंगा", "हिमा", "इन्दिरा", "जमुना",
               "कञ्चन", "ललिता", "मीना", "निर्मला", "प्रतिमा", "राधिका", "सावित्री", "तारा", "उषा", "यमुना",
               "अर्चना", "भगवती", "दीपा", "गौरी", "जानकी", "करुणा", "मनिषा", "नम्रता", "पूजा", "रोजा",
               "सबिता", "सरस्वती", "सुस्मिता", "तुलसी", "उर्मिला", "बसन्ती", "आशा", "एलिना", "कविता", "निशा",
               "अम्बिका", "अनु", "अपेक्षा", "अस्मिता", "बबिता", "भावना", "बिना", "चमेली", "दिव्या", "धनमाया",
               "इशा", "ज्योति", "कुसुम", "लीला", "मधु", "मालती", "नीता", "निरु", "पवित्रा", "प्रमिला",
               "प्रेमा", "प्रिया", "पुनम", "रचना", "राधा", "रमिला", "रेखा", "रोशनी", "रुपा", "साधना",
               "सन्ध्या", "शान्ति", "शारदा", "शर्मिला", "शिला", "सुधा", "सुजाता", "सुलोचना", "सुनैना", "सुप्रिया",
               "तृष्णा", "उमा", "विद्या", "यशोदा", "बिनिता", "चाँदनी", "दीक्षा", "गरिमा", "हेमा", "इन्दु",
               "जुना", "रेनु", "सीमा", "सुमित्रा", "अनिसा", "आरती", "गोमा", "हीरा", "पद्मा", "पुतली",
               "रत्ना", "सम्झना", "अमिता", "चन्द्रकला", "देवकी", "गायत्री", "कोपिला", "पम्फा", "रिमा", "सृष्टि",
               "स्मृति"],
    "objects": ["किताब", "कलम", "कापी", "मेच", "टेबुल", "झोला", "घडी", "मोबाइल", "छाता", "जुत्ता",
                "कम्प्युटर", "पङ्खा", "रेडियो", "ऐना", "बोतल", "बाकस", "कप", "प्लेट", "ओछ्यान", "क्यामेरा",
                "कुर्सी", "थाल", "कचौरा", "चम्चा", "गिलास", "केटली", "कराई", "बाल्टी", "गाग्रो", "लोटा",
                "सिरानी", "डसना", "कम्बल", "तन्ना", "सिरक", "टोपी", "कमिज", "सुरुवाल", "चप्पल", "साबुन",
                "रुमाल", "ताल्चा", "साँचो", "डोको", "नाम्लो", "थर्मस", "कैँची", "मादल", "पेन्सिल", "रबर",
                "स्केल", "डायरी", "खाम", "तस्बिर", "फूलदान", "दियो", "मैनबत्ती", "लालटिन", "टर्च", "ब्याट्री",
                "चार्जर", "हेडफोन", "किबोर्ड", "माउस", "प्रिन्टर", "मुढा", "गुन्द्री", "पर्दा", "खेलौना", "बल",
                "ब्याट", "चङ्गा", "ढोलक", "बाँसुरी", "तबला", "टिफिन", "कुकर", "मिक्सी", "इस्त्री", "हथौडा",
                "पेचकस", "सियो", "डालो", "नाङ्लो", "ठेकी", "करुवा", "अम्खोरा", "सलाई"],
}

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

TASKS = ["transitive", "numeric_extremum", "numeric_pair", "verbal_comparison"]
TASK_WEIGHTS = [0.40, 0.20, 0.20, 0.20]


@dataclass(frozen=True)
class Attribute:
    """A comparable quantity: noun, possessive marker, unit, value range and (for people) adjectives."""
    noun: str
    poss: str
    unit: str
    lo: int
    hi: int
    more_adj: Optional[str] = None
    less_adj: Optional[str] = None


# =====================================================================
# LANGUAGE-SPECIFIC SURFACE REALISATION
# =====================================================================

class HindiPhrasing:
    """Hindi templates. The possessive (का/की) agrees with the attribute noun; adjectives
    (लंबा/लंबी ...) agree with the gender of the person they describe."""

    lang = "hindi"
    yes, no, equal = "हाँ", "नहीं", "बराबर"
    therefore, order_word, per_statement, copula, conj = "अतः", "क्रम", "कथन के अनुसार", "है", "और"
    meaning, most_word, claim_word = "अर्थात्", "सबसे", "प्रश्न का दावा"
    numbers = {2: "दो", 3: "तीन", 4: "चार"}
    people_attrs = [
        Attribute("लंबाई", "की", "सेमी", 120, 199, "लंबा", "छोटा"),
        Attribute("आयु", "की", "वर्ष", 5, 90, "बड़ा", "छोटा"),
        Attribute("वजन", "का", "किग्रा", 20, 110, "भारी", "हल्का"),
        Attribute("गति", "की", "किमी/घंटा", 5, 40, "तेज़", "धीमा"),
    ]
    object_attrs = [
        Attribute("मूल्य", "का", "रुपये", 10, 999),
        Attribute("वजन", "का", "किग्रा", 1, 60),
    ]

    @staticmethod
    def adj(word: str, gender: Optional[str]) -> str:
        """Inflects an आ-ending adjective to its feminine ई form (लंबा -> लंबी); others are invariant."""
        return word[:-1] + "ी" if gender == "f" and word.endswith("ा") else word

    @staticmethod
    def _more(more: bool) -> str:
        return "अधिक" if more else "कम"

    @staticmethod
    def _who(attr: Attribute, objects: bool) -> str:
        return f"किस वस्तु {attr.poss}" if objects else "किस" + attr.poss

    def introduce(self, ents: List[Entity], objects: bool) -> str:
        """Opening sentence naming every entity, e.g. "अमित, रोहित और सुमित तीन दोस्त हैं।"."""
        names = [e[0] for e in ents]
        listing = f"{', '.join(names[:-1])} और {names[-1]}"
        return f"{listing} {self.numbers[len(ents)]} {'वस्तुएँ' if objects else 'दोस्त'} हैं।"

    def relation_templates(self, hi: Entity, lo: Entity, attr: Attribute) -> List[Tuple[str, str]]:
        """All grammatical phrasings of 'hi has more <attr> than lo', each paired with its symbolic
        transcription in mention order ("b < a" when b is named first)."""
        (a, ga), (b, gb) = hi, lo
        n, p = attr.noun, attr.poss
        out = [(f"{a} {p} {n} {b} से अधिक है।", f"{a} > {b}"),
               (f"{b} {p} {n} {a} से कम है।", f"{b} < {a}"),
               (f"{b} की तुलना में {a} {p} {n} अधिक है।", f"{b} < {a}")]
        if attr.more_adj:
            out += [(f"{a} {b} से {self.adj(attr.more_adj, ga)} है।", f"{a} > {b}"),
                    (f"{b} {a} से {self.adj(attr.less_adj, gb)} है।", f"{b} < {a}")]
        return out

    def value(self, e: Entity, attr: Attribute, v: int) -> str:
        return f"{e[0]} {attr.poss} {attr.noun} {v} {attr.unit} है।"

    def q_extreme(self, attr: Attribute, most: bool, objects: bool) -> str:
        return f"इनमें से {self._who(attr, objects)} {attr.noun} सबसे {self._more(most)} है?"

    def q_pair(self, x: Entity, y: Entity, attr: Attribute, more: bool) -> str:
        return f"क्या {x[0]} {attr.poss} {attr.noun} {y[0]} से {self._more(more)} है?"

    def q_which(self, x: Entity, y: Entity, attr: Attribute, more: bool, objects: bool) -> str:
        return f"{x[0]} और {y[0]} में से {self._who(attr, objects)} {attr.noun} {self._more(more)} है?"

    def q_equal(self, x: Entity, y: Entity, attr: Attribute) -> str:
        return f"क्या {x[0]} और {y[0]} {attr.poss} {attr.noun} बराबर है?"

    def q_adj(self, x: Entity, y: Entity, attr: Attribute, more: bool) -> Optional[str]:
        word = attr.more_adj if more else attr.less_adj
        return f"क्या {x[0]} {y[0]} से {self.adj(word, x[1])} है?"

    def r_extreme(self, attr: Attribute, most: bool, e: Entity) -> str:
        return f"{self.therefore} सबसे {self._more(most)} {attr.noun} {e[0]} {attr.poss} है।"



class NepaliPhrasing:
    """Nepali templates. Adjective phrasings (अग्लो, जेठो ...) are only used with masculine subjects,
    because feminine agreement also changes the verb; noun phrasings (उचाइ बढी छ) work for everyone."""

    lang = "nepali"
    yes, no, equal = "हो", "होइन", "बराबर"
    therefore, order_word, per_statement, copula, conj = "त्यसैले", "क्रम", "भनाइ अनुसार", "हो", "र"
    meaning, most_word, claim_word = "अर्थात्", "सबैभन्दा", "प्रश्नको दाबी"
    numbers = {2: "दुई", 3: "तीन", 4: "चार"}
    people_attrs = [
        Attribute("उचाइ", "को", "सेमी", 120, 199, "अग्लो", "होचो"),
        Attribute("उमेर", "को", "वर्ष", 5, 90, "जेठो", "कान्छो"),
        Attribute("तौल", "को", "किग्रा", 20, 110, "गह्रौँ", "हलुका"),
        Attribute("गति", "को", "किमी/घण्टा", 5, 40, "छिटो", "ढिलो"),
    ]
    object_attrs = [
        Attribute("मूल्य", "को", "रुपैयाँ", 10, 999),
        Attribute("तौल", "को", "किग्रा", 1, 60),
    ]

    @staticmethod
    def _more(more: bool) -> str:
        return "बढी" if more else "कम"

    def introduce(self, ents: List[Entity], objects: bool) -> str:
        """Opening sentence naming every entity, e.g. "राम, श्याम र हरि तीन जना साथी हुन्।"."""
        names = [e[0] for e in ents]
        listing = f"{', '.join(names[:-1])} र {names[-1]}"
        return f"{listing} {self.numbers[len(ents)]} {'वटा सामान' if objects else 'जना साथी'} हुन्।"

    def relation_templates(self, hi: Entity, lo: Entity, attr: Attribute) -> List[Tuple[str, str]]:
        """All grammatical phrasings of 'hi has more <attr> than lo', each paired with its symbolic
        transcription in mention order ("b < a" when b is named first)."""
        (a, ga), (b, gb) = hi, lo
        n = attr.noun
        out = [(f"{a}को {n} {b}को भन्दा बढी छ।", f"{a} > {b}"),
               (f"{b}को {n} {a}को भन्दा कम छ।", f"{b} < {a}"),
               (f"{b}को तुलनामा {a}को {n} बढी छ।", f"{b} < {a}")]
        if attr.more_adj and ga == "m":
            out.append((f"{a} {b}भन्दा {attr.more_adj} छ।", f"{a} > {b}"))
        if attr.less_adj and gb == "m":
            out.append((f"{b} {a}भन्दा {attr.less_adj} छ।", f"{b} < {a}"))
        return out

    def value(self, e: Entity, attr: Attribute, v: int) -> str:
        return f"{e[0]}को {attr.noun} {v} {attr.unit} छ।"

    def q_extreme(self, attr: Attribute, most: bool, objects: bool) -> str:
        who = "कुन वस्तुको" if objects else "कसको"
        return f"यी मध्ये {who} {attr.noun} सबैभन्दा {self._more(most)} छ?"

    def q_pair(self, x: Entity, y: Entity, attr: Attribute, more: bool) -> str:
        return f"के {x[0]}को {attr.noun} {y[0]}को भन्दा {self._more(more)} छ?"

    def q_which(self, x: Entity, y: Entity, attr: Attribute, more: bool, objects: bool) -> str:
        who = "कुनको" if objects else "कसको"
        return f"{x[0]} र {y[0]} मध्ये {who} {attr.noun} {self._more(more)} छ?"

    def q_equal(self, x: Entity, y: Entity, attr: Attribute) -> str:
        return f"के {x[0]} र {y[0]}को {attr.noun} बराबर छ?"

    def q_adj(self, x: Entity, y: Entity, attr: Attribute, more: bool) -> Optional[str]:
        if x[1] != "m":
            return None
        return f"के {x[0]} {y[0]}भन्दा {attr.more_adj if more else attr.less_adj} छ?"

    def r_extreme(self, attr: Attribute, most: bool, e: Entity) -> str:
        return f"{self.therefore} सबैभन्दा {self._more(most)} {attr.noun} भएको व्यक्ति {e[0]} हो।"



PHRASINGS = {"hindi": HindiPhrasing(), "nepali": NepaliPhrasing()}
ENTITY_BANKS = {"hindi": HINDI_ENTITIES, "nepali": NEPALI_ENTITIES}


# =====================================================================
# TASK GENERATORS (language-agnostic logic; wording comes from the phrasing object)
# =====================================================================

def make_sample(L, question: str, reasoning: str, answer: str, candidates: List[str], task: str) -> Dict:
    """Assembles the training text and the separate question / reasoning / answer fields."""
    assert answer in candidates
    q_line, r_line = f"प्रश्न: {question}", f"तर्क: {reasoning}"
    return {"text": f"{q_line}\n{r_line}\nउत्तर: {answer}", "question": q_line, "reasoning": r_line,
            "answer": answer, "candidates": candidates, "task": task, "lang": L.lang}


def entity_candidates(ents: List[Entity]) -> List[str]:
    """Candidate answers for an entity question, in a fixed (alphabetical) order unrelated to the answer."""
    return sorted(e[0] for e in ents)


def introduction(L, ents: List[Entity], objects: bool, rng: random.Random) -> str:
    """Introduces the entities in a random order, independent of the answer."""
    return L.introduce(rng.sample(ents, len(ents)), objects)


def relation(L, x: Entity, y: Entity, x_higher: bool) -> str:
    """Symbolic statement "x > y है" / "x < y हो" with no punctuation fused onto the names."""
    return f"{x[0]} {'>' if x_higher else '<'} {y[0]} {L.copula}।"


def claim(L, x: Entity, y: Entity, op: str) -> str:
    """Symbolic transcription of a yes/no question, e.g. "प्रश्न का दावा X < Y है।"."""
    return f"{L.claim_word} {x[0]} {op} {y[0]} {L.copula}।"


def gen_transitive(L, pool: Dict[str, List[Entity]], rng: random.Random) -> Dict[str, str]:
    """Hidden order over 3-4 people, revealed only through shuffled adjacent-pair premises."""
    k = 4 if rng.random() < 0.3 else 3
    order = rng.sample(pool["people"], k)  # order[0] has the most of the attribute
    attr = rng.choice(L.people_attrs)
    pairs = [(order[i], order[i + 1]) for i in range(k - 1)]
    rng.shuffle(pairs)
    chosen = [rng.choice(L.relation_templates(hi, lo, attr)) for hi, lo in pairs]
    premises = " ".join(sentence for sentence, _ in chosen)
    transcribed = f" {L.conj} ".join(symbolic for _, symbolic in chosen)
    normalised = f" {L.conj} ".join(f"{hi[0]} > {lo[0]}" for hi, lo in pairs)
    chain = " > ".join(e[0] for e in order)

    q_type = rng.choice(["max", "min", "pair"])
    if q_type in ("max", "min"):
        most = q_type == "max"
        target = order[0] if most else order[-1]
        question, conclusion, answer = L.q_extreme(attr, most, objects=False), L.r_extreme(attr, most, target), target[0]
        candidates = entity_candidates(order)
    else:
        # Only pairs that are never stated directly, so the relation has to be chained.
        i, j = rng.choice([(a, b) for a in range(k) for b in range(a + 2, k)])
        x, y = (order[i], order[j]) if rng.random() < 0.5 else (order[j], order[i])
        more, x_higher = rng.random() < 0.5, order.index(x) < order.index(y)
        question = L.q_pair(x, y, attr, more)
        conclusion = f"{L.therefore} {relation(L, x, y, x_higher)} {claim(L, x, y, '>' if more else '<')}"
        answer = L.yes if x_higher == more else L.no
        candidates = [L.yes, L.no]

    reasoning = (f"{transcribed} {L.copula}। {L.meaning} {normalised} {L.copula}। "
                 f"{L.order_word}: {chain} {L.copula}। {conclusion}")
    intro = introduction(L, order, False, rng)
    return make_sample(L, f"{intro} {premises} {question}", reasoning, answer, candidates, "transitive")


def gen_numeric_extremum(L, pool: Dict[str, List[Entity]], rng: random.Random) -> Dict[str, str]:
    """Three entities with explicit distinct values; ask which has the most / least."""
    objects = rng.random() < 0.5
    ents = rng.sample(pool["objects" if objects else "people"], 3)
    attr = rng.choice(L.object_attrs if objects else L.people_attrs)
    vals = rng.sample(range(attr.lo, attr.hi + 1), 3)
    most = rng.random() < 0.5
    best = (max if most else min)(range(3), key=lambda i: vals[i])

    statements = " ".join(L.value(e, attr, v) for e, v in zip(ents, vals))
    listing = f" {L.conj} ".join(f"{v} = {e[0]}" for e, v in zip(ents, vals))
    pick = f"{L.most_word} {L._more(most)} मान {vals[best]} = {ents[best][0]} {L.copula}।"
    reasoning = f"{listing} {L.copula}। {pick}"
    intro = introduction(L, ents, objects, rng)
    return make_sample(L, f"{intro} {statements} {L.q_extreme(attr, most, objects)}", reasoning,
                       ents[best][0], entity_candidates(ents), "numeric_extremum")


def gen_numeric_pair(L, pool: Dict[str, List[Entity]], rng: random.Random) -> Dict[str, str]:
    """Two entities with explicit values (possibly equal); greater / smaller / equal questions."""
    objects = rng.random() < 0.5
    a, b = rng.sample(pool["objects" if objects else "people"], 2)
    attr = rng.choice(L.object_attrs if objects else L.people_attrs)
    q_type = rng.choice(["which_more", "which_less", "yn_more", "yn_less", "yn_equal"])
    va = rng.randint(attr.lo, attr.hi)
    if rng.random() < (0.5 if q_type == "yn_equal" else 1 / 3):
        vb = va
    else:
        vb = rng.choice([v for v in range(attr.lo, attr.hi + 1) if v != va])

    statements = f"{L.value(a, attr, va)} {L.value(b, attr, vb)}"
    # The question may name the two entities in either order.
    (x, vx), (y, vy) = ((a, va), (b, vb)) if rng.random() < 0.5 else ((b, vb), (a, va))
    sign = lambda u, w: ">" if u > w else "<" if u < w else "="
    compared = f"{va} = {a[0]} {L.conj} {vb} = {b[0]} {L.copula}। {va} {sign(va, vb)} {vb} {L.copula}, {L.therefore}"

    if q_type.startswith("which"):
        more = q_type == "which_more"
        question = L.q_which(x, y, attr, more, objects)
        (big, _), (small, _) = sorted([(x, vx), (y, vy)], key=lambda p: -p[1])
        if vx == vy:
            reasoning, answer = f"{compared} {x[0]} = {y[0]} {L.copula}।", L.equal
        else:
            reasoning, answer = f"{compared} {big[0]} > {small[0]} {L.copula}।", (big if more else small)[0]
        candidates = entity_candidates([x, y]) + [L.equal]
    else:
        concluded = f"{compared} {x[0]} {sign(vx, vy)} {y[0]} {L.copula}।"
        if q_type == "yn_equal":
            question, claimed, answer = L.q_equal(x, y, attr), "=", (L.yes if vx == vy else L.no)
        else:
            more = q_type == "yn_more"
            question, claimed = L.q_pair(x, y, attr, more), (">" if more else "<")
            answer = L.yes if (vx > vy if more else vx < vy) else L.no
        reasoning = f"{concluded} {claim(L, x, y, claimed)}"
        candidates = [L.yes, L.no]
    intro = introduction(L, [a, b], objects, rng)
    return make_sample(L, f"{intro} {statements} {question}", reasoning, answer, candidates, "numeric_pair")


def gen_verbal_comparison(L, pool: Dict[str, List[Entity]], rng: random.Random) -> Dict[str, str]:
    """One premise in a random phrasing; yes/no question with random entity order and direction."""
    hi, lo = rng.sample(pool["people"], 2)
    attr = rng.choice(L.people_attrs)
    premise, symbolic = rng.choice(L.relation_templates(hi, lo, attr))
    x, y = (hi, lo) if rng.random() < 0.5 else (lo, hi)
    more, x_higher = rng.random() < 0.5, x is hi
    question = L.q_adj(x, y, attr, more) if rng.random() < 0.5 else None
    question = question or L.q_pair(x, y, attr, more)
    reasoning = (f"{L.per_statement} {symbolic} {L.copula}, {L.therefore} {relation(L, x, y, x_higher)} "
                 f"{claim(L, x, y, '>' if more else '<')}")
    answer = L.yes if x_higher == more else L.no
    intro = introduction(L, [hi, lo], False, rng)
    return make_sample(L, f"{intro} {premise} {question}", reasoning, answer, [L.yes, L.no], "verbal_comparison")


GENERATORS = {
    "transitive": gen_transitive,
    "numeric_extremum": gen_numeric_extremum,
    "numeric_pair": gen_numeric_pair,
    "verbal_comparison": gen_verbal_comparison,
}


# =====================================================================
# SPLITS, DE-DUPLICATION & STATISTICS
# =====================================================================

def split_entity_pools(bank: Dict[str, List[str]], rng: random.Random) -> Dict[str, Dict[str, List[Entity]]]:
    """Partitions male / female / object names into disjoint train / val / test pools."""
    all_names = bank["male"] + bank["female"] + bank["objects"]
    assert len(all_names) == len(set(all_names)), "entity names must be unique within a language"

    pools = {s: {"people": [], "objects": []} for s in SPLIT_FRACTIONS}
    for key, gender in [("male", "m"), ("female", "f"), ("objects", None)]:
        names = bank[key][:]
        rng.shuffle(names)
        n_train = round(len(names) * SPLIT_FRACTIONS["train"])
        n_val = round(len(names) * SPLIT_FRACTIONS["val"])
        bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, len(names))}
        for split, (lo, hi) in bounds.items():
            pools[split]["objects" if gender is None else "people"] += [(n, gender) for n in names[lo:hi]]
    return pools


def build_dataset_split(L, split_name: str, num_samples: int, pool: Dict[str, List[Entity]],
                        rng: random.Random, seen: set) -> List[Dict[str, str]]:
    """Samples tasks by TASK_WEIGHTS and keeps only texts not seen in any split so far."""
    samples = []
    with tqdm(total=num_samples, desc=f"{L.lang.capitalize()} {split_name}") as pbar:
        while len(samples) < num_samples:
            task = rng.choices(TASKS, weights=TASK_WEIGHTS)[0]
            sample = GENERATORS[task](L, pool, rng)
            if sample["text"] in seen:
                continue
            seen.add(sample["text"])
            samples.append(sample)
            pbar.update(1)
    return samples


def split_stats(samples: List[Dict[str, str]], L) -> Dict:
    """Task mix and answer-type distribution for one split."""
    kinds = Counter("yes" if s["answer"] == L.yes else "no" if s["answer"] == L.no
                    else "equal" if s["answer"] == L.equal else "entity" for s in samples)
    return {"size": len(samples), "tasks": dict(Counter(s["task"] for s in samples)), "answer_types": dict(kinds)}


def save_jsonl(samples: List[Dict[str, str]], output_path: str):
    """Saves dataset samples to JSONL format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Synthetic Comparative-Reasoning Dataset Generator")
    parser.add_argument("--num-train", type=int, default=12000, help="Number of training samples per language")
    parser.add_argument("--num-val", type=int, default=1000, help="Number of validation samples per language")
    parser.add_argument("--num-test", type=int, default=1000, help="Number of test samples per language")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out-dir", type=str, default=None, help="Base directory for output JSONL files")
    args = parser.parse_args()

    base_path = args.out_dir if args.out_dir else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sizes = {"train": args.num_train, "val": args.num_val, "test": args.num_test}

    for lang, L in PHRASINGS.items():
        rng = random.Random(f"{args.seed}-{lang}")
        pools = split_entity_pools(ENTITY_BANKS[lang], rng)
        lang_dir = os.path.join(base_path, lang, "data", "reasoning")
        seen: set = set()
        stats = {"entity_pools": {s: {k: len(v) for k, v in p.items()} for s, p in pools.items()},
                 "task_weights": dict(zip(TASKS, TASK_WEIGHTS)), "splits": {}}

        for split, n in sizes.items():
            samples = build_dataset_split(L, split, n, pools[split], rng, seen)
            save_jsonl(samples, os.path.join(lang_dir, f"{split}.jsonl"))
            stats["splits"][split] = split_stats(samples, L)

        with open(os.path.join(lang_dir, "dataset_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(json.dumps(stats["splits"], ensure_ascii=False))

    print("\nPhase 3 Synthetic Reasoning Dataset Generation Complete for Hindi and Nepali!")


if __name__ == "__main__":
    main()
