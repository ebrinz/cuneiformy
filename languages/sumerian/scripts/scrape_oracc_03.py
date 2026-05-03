from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "scrape_oracc",
    os.path.join(os.path.dirname(__file__), "03_scrape_oracc.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

extract_lemmas = _mod.extract_lemmas
extract_lines = _mod.extract_lines
save_texts = _mod.save_texts
parse_project_zip = _mod.parse_project_zip
