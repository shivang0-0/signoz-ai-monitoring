"""Architecture tests — verify module boundaries are not violated."""

import ast
import importlib
from pathlib import Path


def test_all_modules_importable() -> None:
    modules = ["routers", "models", "schemas", "services", "clients", "telemetry"]
    for m in modules:
        importlib.import_module(m)


def test_routers_do_not_import_models_directly() -> None:
    """Routers should not reach into models internals."""
    router_dir = Path("routers")
    for f in router_dir.glob("*.py"):
        if f.name == "__init__.py":
            continue
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("models.")
            ):
                raise AssertionError(
                    f"{f.name} imports from {node.module} — routers should use services"
                )


def test_services_do_not_import_routers() -> None:
    """Services should not depend on routers."""
    services_dir = Path("services")
    for f in services_dir.glob("*.py"):
        if f.name == "__init__.py":
            continue
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("routers")
            ):
                raise AssertionError(
                    f"{f.name} imports from {node.module} — services must not depend on routers"
                )
