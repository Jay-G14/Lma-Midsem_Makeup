"""CL3410 make-up mid-semester assignment (variant 1). Self-contained: everything it uses lives inside midsem/.

Importing this package puts midsem/project/ first on sys.path. That folder holds the copies of the individual
project's code and data that Part A reuses (hindi/model, hindi/train, hindi/eval, scripts/generate_reasoning_data.py,
the tokenizer, the Phase 2/3 checkpoints and the data), so `import hindi...` always resolves to the copy inside
midsem/ and never to a project checkout next to it. Every script imports `midsem` before `hindi`.
"""

import sys
import warnings
from pathlib import Path

PROJECT = Path(__file__).resolve().parent / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
_hindi = sys.modules.get("hindi")
if _hindi is not None and not str(getattr(_hindi, "__file__", "") or "").startswith(str(PROJECT)):
    warnings.warn(f"`hindi` was imported from {getattr(_hindi, '__file__', '?')} before `midsem`; import midsem first "
                  f"so that the copy in {PROJECT} is used")
