from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "scrape_cdli",
    os.path.join(os.path.dirname(__file__), "02_scrape_cdli.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

parse_atf = _mod.parse_atf
save_texts = _mod.save_texts
download_cdli = _mod.download_cdli
