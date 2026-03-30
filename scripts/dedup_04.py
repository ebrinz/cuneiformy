from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "dedup",
    os.path.join(os.path.dirname(__file__), "04_deduplicate_corpus.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

deduplicate = _mod.deduplicate
deduplicate_with_stats = _mod.deduplicate_with_stats
