from typing import Dict, Any
from setuptools_rust import Binding, RustExtension, build_ext

rust_extensions = [RustExtension("yurki.yurki", binding=Binding.RustCPython)]


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {
            "packages": ["yurki"],
            "rust_extensions": rust_extensions,
            "cmdclass": dict(build_ext=build_ext),
            "zip_safe": False,
            "include_package_data": True,
        }
    )
