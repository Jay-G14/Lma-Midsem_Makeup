#!/usr/bin/env python3
"""C1.1: the 20 Hindi word problems (single numeric answer) -> midsem/data/c1_problems.jsonl.

15 standard problems and 5 "modified" ones in the style of the drifting-boat example: each modified
problem M0k is a near-copy of standard problem S0k in which a key word changes what has to be
computed, so the familiar template's answer ("template_answer") becomes wrong.

  M01 बहते हुए (drifting)   the class example on S01's boat (engine 12 km/h, current 3 km/h): powered downstream for
                            6 h and back takes 15*6/9 = 10 h  ->  drifting (engine off) covers only 3*6 = 18 km: 18/9 = 2 h
  M02 खाली करता (empties)   two filling pipes 1/(1/4+1/6) = 2.4     ->  one pipe empties: 1/(1/4-1/6) = 12
  M03 को छोड़कर (except)     17 sheep, 9 die: 17-9 = 8               ->  all except 9 die: 9
  M04 होगा (will be)        sister is half of 10: 5                 ->  age gap stays 5, so at 40 she is 35 (not 20)
  M05 सूखती (dry)           6 workers 10 days -> 12 workers: 5      ->  shirts dry in parallel: still 10 hours (not 5 or 20)
"""

from midsem.part_a.common import DATA_DIR, write_jsonl

PROBLEMS = [
    # --- standard problems (S01-S05 have modified twins M01-M05) ---
    ("S01", "standard", "boat / streams",
     "एक नाव धारा के अनुकूल 15 किमी/घंटा और धारा के प्रतिकूल 9 किमी/घंटा की चाल से चलती है। शांत जल में नाव की चाल कितने किमी/घंटा है?",
     "12", "Downstream 15, upstream 9 km/h: speed in still water = (15+9)/2."),
    ("S02", "standard", "pipes / work",
     "नल A एक खाली टंकी को 4 घंटे में भरता है और नल B उसी टंकी को 6 घंटे में भरता है। दोनों नल एक साथ खोलने पर खाली टंकी कितने घंटे में भरेगी?",
     "2.4", "Pipe A fills in 4 h, pipe B fills in 6 h: together 1/(1/4+1/6) = 12/5 h."),
    ("S03", "standard", "subtraction",
     "एक किसान के पास 17 भेड़ें थीं। उनमें से 9 भेड़ें मर गईं। अब किसान के पास कितनी भेड़ें बची हैं?",
     "8", "17 sheep, 9 die: 17-9."),
    ("S04", "standard", "ages",
     "राम की आयु 10 वर्ष है। उसकी बहन की आयु राम की आयु की आधी है। बहन की आयु कितने वर्ष है?",
     "5", "Ram is 10, sister is half his age."),
    ("S05", "standard", "work / inverse proportion",
     "6 मज़दूर एक दीवार को 10 दिन में बनाते हैं। उसी दीवार को 12 मज़दूर कितने दिन में बनाएँगे?",
     "5", "6 workers take 10 days: 60 worker-days / 12 workers."),
    ("S06", "standard", "probability",
     "एक थैले में 20 गेंदें हैं, जिनमें से केवल 1 गेंद लाल है। बिना देखे एक गेंद निकालने पर उसके लाल होने की प्रायिकता कितनी है?",
     "0.05", "1 red ball out of 20: probability 1/20."),
    ("S07", "standard", "percentage",
     "एक कक्षा में 40 विद्यार्थी हैं। उनमें से 25% विद्यार्थी चश्मा पहनते हैं। कितने विद्यार्थी चश्मा पहनते हैं?",
     "10", "25% of 40 students."),
    ("S08", "standard", "speed",
     "एक बस 3 घंटे में 180 किमी की दूरी तय करती है। बस की औसत चाल कितने किमी/घंटा है?",
     "60", "180 km in 3 h."),
    ("S09", "standard", "simple interest",
     "5000 रुपये पर 8% वार्षिक साधारण ब्याज की दर से 3 वर्ष का ब्याज कितने रुपये होगा?",
     "1200", "Simple interest 5000 x 8% x 3."),
    ("S10", "standard", "average",
     "पाँच संख्याएँ 12, 15, 18, 21 और 24 हैं। इन संख्याओं का औसत कितना है?",
     "18", "Mean of 12, 15, 18, 21, 24 = 90/5."),
    ("S11", "standard", "fractions",
     "एक पिज़्ज़ा को 8 बराबर टुकड़ों में काटा गया। रीना ने उनमें से 2 टुकड़े खा लिए। पिज़्ज़ा का कितना भाग बचा है? उत्तर भिन्न या दशमलव में दीजिए।",
     "0.75", "8 slices, 2 eaten: 6/8 = 3/4 left."),
    ("S12", "standard", "area",
     "एक आयताकार खेत की लंबाई 25 मीटर और चौड़ाई 12 मीटर है। खेत का क्षेत्रफल कितने वर्ग मीटर है?",
     "300", "25 m x 12 m."),
    ("S13", "standard", "unit price",
     "5 किलो सेब का मूल्य 600 रुपये है। 3 किलो सेब का मूल्य कितने रुपये होगा?",
     "360", "600/5 = 120 per kg, times 3."),
    ("S14", "standard", "time",
     "एक फ़िल्म दोपहर 2:45 बजे शुरू होती है और शाम 5:15 बजे समाप्त होती है। फ़िल्म कितने मिनट लंबी है?",
     "150", "2:45 to 5:15 is 2 h 30 min."),
    ("S15", "standard", "ratio",
     "राम और श्याम ने 1500 रुपये 3:2 के अनुपात में आपस में बाँटे। राम को कितने रुपये मिले?",
     "900", "3/5 of 1500."),
    # --- modified problems: one key word changes the answer away from the twin's template ---
    ("M01", "modified", "boat / streams",
     "एक नाव का इंजन उसे शांत जल में 12 किमी/घंटा की चाल से चला सकता है। धारा की चाल 3 किमी/घंटा है। नाव 6 घंटे तक धारा के साथ बहते हुए स्थान A से स्थान B तक जाती है। स्थान B से स्थान A तक वापस आने में नाव को कितने घंटे लगेंगे?",
     "2", "The class's drifting-boat example on S01's boat: drifting (engine off) moves at the current's 3 km/h, so A-B is "
          "3*6 = 18 km and the return upstream at 12-3 = 9 km/h takes 2 h; the template (powered, 15 km/h downstream) "
          "gives 90 km and 10 h."),
    ("M02", "modified", "pipes / work",
     "नल A एक खाली टंकी को 4 घंटे में भरता है और नल B भरी हुई टंकी को 6 घंटे में खाली करता है। दोनों नल एक साथ खोलने पर खाली टंकी कितने घंटे में भरेगी?",
     "12", "Pipe B empties instead of fills: 1/(1/4-1/6) = 12 h, not 2.4 h."),
    ("M03", "modified", "subtraction",
     "एक किसान के पास 17 भेड़ें थीं। 9 भेड़ों को छोड़कर बाकी सभी भेड़ें मर गईं। अब किसान के पास कितनी भेड़ें बची हैं?",
     "9", "All except 9 die, so 9 are left, not 17-9 = 8."),
    ("M04", "modified", "ages",
     "राम की आयु 10 वर्ष है। उसकी बहन की आयु राम की आयु की आधी है। जब राम 40 वर्ष का होगा, तब उसकी बहन की आयु कितने वर्ष होगी?",
     "35", "The sister is 5 now; the 5-year gap stays, so 35 when Ram is 40, not half of 40 = 20."),
    ("M05", "modified", "work / inverse proportion",
     "6 कमीज़ें धूप में 10 घंटे में सूखती हैं। अगर 12 कमीज़ें एक साथ धूप में फैलाई जाएँ, तो वे कितने घंटे में सूखेंगी?",
     "10", "Shirts dry in parallel, so 12 shirts still take 10 h; the work template gives 5 (or 20 by direct proportion)."),
]

TEMPLATE_ANSWERS = {"M01": "10", "M02": "2.4", "M03": "8", "M04": "20", "M05": "5"}


def rows():
    out = []
    for pid, kind, topic, problem, answer, note in PROBLEMS:
        row = {"id": pid, "type": kind, "topic": topic, "problem": problem, "answer": answer, "english_gloss": note}
        if kind == "modified":
            row["twin"] = "S" + pid[1:]
            row["template_answer"] = TEMPLATE_ANSWERS[pid]
        out.append(row)
    return out


def main():
    data = rows()
    assert len(data) == 20 and sum(r["type"] == "modified" for r in data) == 5
    write_jsonl(DATA_DIR / "c1_problems.jsonl", data)
    print(f"Wrote {len(data)} problems to {DATA_DIR / 'c1_problems.jsonl'}")


if __name__ == "__main__":
    main()
