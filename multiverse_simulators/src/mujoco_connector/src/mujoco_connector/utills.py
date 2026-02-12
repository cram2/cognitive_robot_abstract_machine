import os
import mujoco

def get_multiverse_connector_plugins():
    """
    Get the list of MuJoCo plugin files for the current MuJoCo version.
    The plugins are expected to be located in a directory named "mujoco_plugin/mujoco-{version}" relative to this file.

    :return: A list of file paths to the MuJoCo plugin files.
    """
    mujoco_version = str(mujoco.mj_version())
    possible_mujoco_plugins_dir = os.path.join(os.path.dirname(__file__), "..", "..", "mujoco_plugin", f"mujoco-{mujoco_version[0]}.{mujoco_version[1]}.{mujoco_version[2]}")
    file_ext = ".so" if os.name == "posix" else ".dll"
    if os.path.exists(possible_mujoco_plugins_dir):
        plugins_dir = possible_mujoco_plugins_dir
    else:
        raise FileNotFoundError(f"Could not find MuJoCo plugins directory at {possible_mujoco_plugins_dir}")
    return [os.path.join(plugins_dir, f) for f in os.listdir(plugins_dir) if os.path.isfile(os.path.join(plugins_dir, f)) and f.endswith(file_ext)]
