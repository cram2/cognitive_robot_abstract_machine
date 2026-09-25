"""
The built wheel runs its console command and serves every packaged asset.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile


# %% installed distribution
def test_wheel_console_serves_its_own_assets(
    fixture_scene: Path, tmp_path: Path
) -> None:
    """
    Build and install outside the checkout, then exercise its actual console script.

    :param fixture_scene: Deterministic scene for the server's startup knowledge load.
    :param tmp_path: Isolated wheel output, installation and command working directory.
    """
    repository = Path(__file__).resolve().parents[2]
    wheels = tmp_path / "wheels"
    installed = tmp_path / "installed"
    commands = [
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            str(repository / "cramera"),
            "--wheel-dir",
            str(wheels),
        ],
    ]
    for command in commands:
        result = subprocess.run(
            command, cwd=tmp_path, capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stdout + result.stderr
    wheel = next(wheels.glob("cramera-*.whl"))
    with ZipFile(wheel) as archive:
        packaged_modules = {
            name
            for name in archive.namelist()
            if name.startswith("cramera/") and name.endswith(".py")
        }
    source_root = repository / "cramera" / "src"
    assert packaged_modules == {
        str(source.relative_to(source_root))
        for source in (source_root / "cramera").rglob("*.py")
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(installed),
            str(wheel),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    environment = dict(os.environ, PYTHONPATH=str(installed))
    probe = repository / "test" / "cramera_test" / "dataset" / "wheel_import.py"
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    imported = json.loads(result.stdout)
    assert imported["package"] == str(installed / "cramera")
    assert imported["providers"] == ["cramera.live.visualization:LiveVisualization"]
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    log = tmp_path / "viewer.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                str(installed / "bin" / "cramera"),
                str(port),
                "--no-browser",
            ],
            cwd=tmp_path,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 20
            endpoint = f"http://127.0.0.1:{port}"
            while True:
                assert process.poll() is None, log.read_text()
                try:
                    with urlopen(endpoint + "/index.html", timeout=1) as response:
                        page = response.read().decode()
                    break
                except URLError:
                    assert time.monotonic() < deadline, log.read_text()
                    time.sleep(0.05)
            assets = re.findall(r'(?:src|href)="([^\"]+)"', page)
            assert assets
            assets.extend(
                [
                    "vendor/checksums.json",
                    "vendor/licenses/three-MIT.txt",
                    "vendor/licenses/vis-network-MIT.txt",
                    "vendor/licenses/urdf-loader-Apache-2.0.txt",
                ]
            )
            for asset in assets:
                with urlopen(endpoint + "/" + asset, timeout=3) as response:
                    assert response.status == 200, asset
                    assert response.read(), asset
        finally:
            process.terminate()
            process.wait(timeout=10)
