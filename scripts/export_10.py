from importlib.util import spec_from_file_location, module_from_spec
import os

_spec = spec_from_file_location(
    "export",
    os.path.join(os.path.dirname(__file__), "10_export_production.py"),
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

project_all_vectors = _mod.project_all_vectors
