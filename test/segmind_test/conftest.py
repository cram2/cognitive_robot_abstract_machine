import os
import runpy
from pathlib import Path


def pytest_configure(config):
    # Only the xdist controller generates: workers run this hook too, and
    # concurrent writers would truncate the file while another process formats it.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    # Ensure ORM classes are generated before tests run
    repo_root = Path(__file__).resolve().parents[2]
    generate_orm_path = repo_root / "segmind" / "scripts" / "generate_orm.py"
    # Execute the ORM generation script as a standalone module
    runpy.run_path(str(generate_orm_path), run_name="__main__")