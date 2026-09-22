"""Put the vendored generator tree on sys.path, or explain why it is absent.

`driftsense.generate` and `driftsense.presets` import from the vendored
generator (`src.sem_imaging`, `src.presets`), which lives at
`<repo>/generator` and is resolved relative to this file. That tree is not
part of the installed package: pyproject.toml packages `driftsense` alone,
because the generator's upstream terms are undetermined (see NOTICE).

So those two modules import from a source checkout and not from
site-packages. That is deliberate -- they are dataset-generation helpers and
nothing on the inference path touches them; `driftsense/__init__.py` pulls in
`driftsense.model` alone, so `pip install driftsense` yields a working
localiser either way.

What it is not is self-explanatory. Left alone, the symptom is

    ModuleNotFoundError: No module named 'src'

which names neither the cause nor the fix. This module makes it say both.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR_ROOT = os.path.join(REPO_ROOT, "generator")


class VendoredGeneratorMissing(ImportError):
    """The vendored generator tree is not beside the installed package."""


def ensure_generator_on_path() -> str:
    """Prepend the vendored generator to sys.path; return the path used.

    Raises VendoredGeneratorMissing if the tree is not there, which is the
    normal state of an installed wheel rather than a broken checkout.
    """
    if not os.path.isdir(GENERATOR_ROOT):
        raise VendoredGeneratorMissing(
            "the vendored generator tree is not present at "
            + GENERATOR_ROOT
            + "\n\nThis module imports the generator's own `src` package, "
            "which ships only in a source checkout -- it is not part of the "
            "installed `driftsense` distribution (see NOTICE for why).\n\n"
            "If you installed driftsense from a wheel: dataset generation "
            "needs the repository. Clone it and run from there.\n"
            "If you are in a checkout: generator/ is missing or empty; "
            "restore it with `git checkout -- generator`.\n\n"
            "Inference does not need this. register.py and phase3.py import "
            "neither driftsense.generate nor driftsense.presets."
        )
    if GENERATOR_ROOT not in sys.path:
        sys.path.insert(0, GENERATOR_ROOT)
    return GENERATOR_ROOT
