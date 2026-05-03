from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "clean",
    os.path.join(os.path.dirname(__file__), "05_clean_and_tokenize.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

clean_atf_line = _mod.clean_atf_line
build_corpus = _mod.build_corpus
normalize_transliteration = _mod.normalize_transliteration
