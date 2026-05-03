from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "scrape_etcsl",
    os.path.join(os.path.dirname(__file__), "01_scrape_etcsl.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

parse_etcsl_xml = _mod.parse_etcsl_xml
parse_translation_xml = _mod.parse_translation_xml
match_translations = _mod.match_translations
save_texts = _mod.save_texts
download_etcsl = _mod.download_etcsl
load_etcsl_files = _mod.load_etcsl_files
