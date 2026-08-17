import importlib
import importlib.abc
import sys
import unittest


class _HeavyDependencyBlocker(importlib.abc.MetaPathFinder):
    BLOCKED = ("agentscope", "shortuuid", "ollama")

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname in self.BLOCKED or fullname.startswith(
            tuple(f"{name}." for name in self.BLOCKED)
        ):
            raise ModuleNotFoundError(f"blocked optional dependency: {fullname}")
        return None


class LightweightStoreImportTest(unittest.TestCase):
    def test_m2_store_import_does_not_load_agent_dependencies(self):
        package_modules = [
            name
            for name in sys.modules
            if name == "AgeMem_code_agentscope"
            or name.startswith("AgeMem_code_agentscope.")
        ]
        optional_modules = [
            name
            for name in sys.modules
            if name in _HeavyDependencyBlocker.BLOCKED
            or name.startswith(
                tuple(f"{item}." for item in _HeavyDependencyBlocker.BLOCKED)
            )
        ]
        removed_names = package_modules + optional_modules
        saved = {name: sys.modules.pop(name) for name in removed_names}
        blocker = _HeavyDependencyBlocker()
        sys.meta_path.insert(0, blocker)
        try:
            module = importlib.import_module("AgeMem_code_agentscope.memory_store")
            package = importlib.import_module("AgeMem_code_agentscope")
            self.assertIs(package.InMemoryStore, module.InMemoryStore)
            self.assertFalse(
                any(
                    name == "agentscope" or name.startswith("agentscope.")
                    for name in sys.modules
                )
            )
            self.assertNotIn("shortuuid", sys.modules)
        finally:
            sys.meta_path.remove(blocker)
            for name in list(sys.modules):
                if name == "AgeMem_code_agentscope" or name.startswith(
                    "AgeMem_code_agentscope."
                ):
                    sys.modules.pop(name, None)
            sys.modules.update(saved)


if __name__ == "__main__":
    unittest.main()
