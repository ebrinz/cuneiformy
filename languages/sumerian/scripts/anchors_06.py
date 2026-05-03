from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "anchors",
    os.path.join(os.path.dirname(__file__), "06_extract_anchors.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

extract_epsd2_anchors = _mod.extract_epsd2_anchors
extract_cooccurrence_anchors = _mod.extract_cooccurrence_anchors
merge_anchors = _mod.merge_anchors
