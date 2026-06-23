from pathlib import Path
import subprocess
import sys


def regenerate(generate_orm_path: Path) -> None:
    """
    Runs the provided generate_orm.py file in a subprocess.
    """
    generate_orm = generate_orm_path.resolve()
    folder = generate_orm.parent

    if generate_orm.name != "generate_orm.py":
        raise ValueError(f"Expected a generate_orm.py file, got: {generate_orm}")

    if not generate_orm.exists():
        raise FileNotFoundError(f"Generator not found: {generate_orm}")

    subprocess.run(
        [sys.executable, str(generate_orm)],
        cwd=folder,
        check=True,
    )


def clear_file(file_path: Path) -> None:
    """
    Deletes the contents of a file without deleting the file itself.
    """
    path = file_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    path.write_text("", encoding="utf-8")


clear_file(
    Path("../semantic_digital_twin/src/semantic_digital_twin/orm/ormatic_interface.py")
)
clear_file(Path("../coraplex/src/coraplex/orm/ormatic_interface.py"))
clear_file(Path("../experiments/src/experiments/orm/ormatic_interface.py"))

regenerate(Path("../semantic_digital_twin/scripts/generate_orm.py"))
regenerate(Path("../coraplex/scripts/generate_orm.py"))
regenerate(Path("../experiments/scripts/generate_orm.py"))
