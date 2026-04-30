#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

from ament_index_python import get_packages_with_prefixes

rk_default_modules = [
    os.path.join("annotators"),
    os.path.join("descriptors"),
    os.path.join("descriptors", "analysis_engines"),
]


def create_robokudo_package(package_name: str) -> bool:
    if package_name.find("/") != -1:
        print(f"Package name cannot contain a '/': '{package_name}'")
        return False

    packages = get_packages_with_prefixes().keys()
    if package_name in packages:
        print(f"Package already exists in this workspace: '{package_name}'")
        return False

    package_dir = os.path.join(os.getcwd(), package_name)
    if os.path.exists(package_dir):
        print(f"Directory already exists: '{package_dir}'")
        return False

    print(f"Creating RoboKudo package in: {package_dir}")

    try:
        subprocess.run(
            [
                "ros2",
                "pkg",
                "create",
                package_name,
                "--build-type",
                "ament_python",
                "--dependencies",
                "robokudo",
                "rclpy",
            ],
            check=True,
        )

        package_src_dir = os.path.join(package_dir, package_name)
        for module in rk_default_modules:
            os.makedirs(os.path.join(package_src_dir, module))
            os.mknod(os.path.join(package_src_dir, module, "__init__.py"))

    except subprocess.CalledProcessError as e:
        print(f"Failed to create package '{package_name}': {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package_name", type=str, help="Name of the package to create.")
    args = parser.parse_args()

    if not create_robokudo_package(args.package_name):
        sys.exit(1)


if __name__ == "__main__":
    main()
