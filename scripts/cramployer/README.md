# cramployer

**cramployer** (CRAM + deployer) prepares a Docker environment for developing and running the CRAM
workspace, including RoboKudo. It reuses the repository's Docker and CI setup,
mounts editable host source, and supports ROS networking, NVIDIA GPUs, and GUI
windows. Open a shell and run whichever application you need.

cramployer aims to make CRAM easy to deploy, especially for external users of
our stack. ROS and workspace dependencies are installed inside Docker, minimizing
host changes and making the environment reproducible on machines that may be
wiped or reinstalled. Host prerequisites are listed below. Keep source changes
and configuration backed up outside any machine that may be erased; Docker
volumes and local files are stored on that machine too.

After preparing the initial configuration, two commands build the environment
and open a shell with CRAM and ROS set up: `cramployer setup` and `cramployer shell`.

Machines with the container deployed are called **cramployees**.

## Command reference

Run commands from this folder. Review `.env` before the first setup.

| Command | What it does |
| --- | --- |
| `./cramployer setup` | Build images, install dependencies, check the native extension, build ROS, and regenerate ORM interfaces. |
| `./cramployer shell` | Open a new interactive container with ROS and the Python environment activated. Exiting removes that container. |
| `./cramployer shell COMMAND ...` | Run a command in the same environment and remove the container when it finishes. |
| `./cramployer root-shell [CONTAINER]` | Open a root shell in an existing development container, with ROS and Python activated. |
| `./cramployer ros-build` | Rebuild the ROS workspace in a temporary container. |
| `./cramployer orm-build` | Regenerate ORM interfaces in a temporary container. |
| `./cramployer config` | Show the effective Compose configuration. |
| `./cramployer help` | Show the available commands. |

`./cramployer shell-offline` remains an alias for `./cramployer shell`. Setup and shell
commands do not start RoboKudo or another application automatically. The old
application service commands (`up`, `restart`, `logs`, `stop`, and `down`) have
been removed. Application output appears in the terminal where you launch it.

## Locations and first setup

The tools live in `scripts/cramployer/` within the CRAM checkout. The launcher
sets `CRAM_REPOSITORY` to the enclosing repository, and `.env.example` uses it as
the default `CRAM_SOURCE`. Host edits are mounted at `/workspace/cram`.
ROS sources default to `~/cramployer-workspace/ros_ws/src`.

Initialize the checkout's submodules with `git submodule update --init --recursive`.
For first use, copy the example configuration from the repository root:

```bash
cp scripts/cramployer/.env.example scripts/cramployer/.env
```

Review `scripts/cramployer/.env` for source locations, resource limits, and hardware
options. Preserve an existing `.env` when updating an installation. Then run:

```bash
./scripts/cramployer/cramployer setup
./scripts/cramployer/cramployer shell
```

Optionally inspect the resolved settings with
`./scripts/cramployer/cramployer config` before setup. If `.env` is missing, the
launcher creates it from the example and exits so you can review it before retrying.

Alternatively, enter `scripts/cramployer/` and use `./cramployer` as shown below.
Paths resolve from the launcher directory regardless of the caller's working
directory. Configure absolute `CRAM_SOURCE` and `ROS_SOURCE` paths in `.env` to
use other checkouts. Local `.env`, `container.env`, and `.state/` are ignored by Git.

Setup clones the configured repository and its submodules when `CRAM_SOURCE`
does not exist. Existing checkouts are used as-is; setup never pulls or resets
one. Set `REPOSITORY_URL` and `REPOSITORY_BRANCH` when configuring a separate
source checkout.
Existing host ROS sources are preserved.

The first setup can take substantial time and disk space. It:

1. Builds the checkout's `.github/docker/Dockerfile` and its setup scripts.
2. Seeds an empty host `ROS_SOURCE` from the image's ROS workspace sources.
3. Installs the whole Python workspace using the common CI installation step,
   selected package-specific CI steps, and configured workspace extras.
4. Compiles the `random_events` native extension into the image and tests its
   availability with the host source mounted.
5. Builds ROS with ordinary `python -m colcon build --merge-install` and
   regenerates ORM interfaces.

Setup then exits with the environment ready. Each shell sources both ROS
workspaces and activates `/opt/ros/cram-env`. It uses your host UID and GID, so
files you create in the mounted checkout belong to you.

## Dependencies and optional features

### Deployment-only apt packages

Add system packages needed by cramployer to `apt-packages.txt`, one name per line.
Blank lines and `#` comments are supported. These packages are installed into the
image in a separate layer after the extracted CI installation. Maintain this list
to add deployment dependencies without changing the shared ROS setup or CI workflow.

The list includes `ros-jazzy-rmw-cyclonedds-cpp`. Installing it does not change
the default middleware; select Cyclone DDS through `container.env` as described
below. After editing the apt list, run `./cramployer setup`, then open a new shell
to use the rebuilt image. Already-running containers retain their installed packages.

### Workspace and CI dependencies

CI YAML extraction remains in place. The common `Install dependencies` step
installs all workspace members and the root development extra. RoboKudo's extra
CI requirements remain enabled by default:

```dotenv
CI_PACKAGES=robokudo
TORCH_MODE=default
WORKSPACE_EXTRAS=
```

`CI_PACKAGES` selects additional installation steps from the checkout's current
`.github/workflows/ci_reusable.yml`:

| Selection | Extracted CI step |
| --- | --- |
| Always included | `Install dependencies` |
| `robokudo` | `Install robokudo optional requirements` |
| `semantic_digital_twin` | `Install drake` |

Set `CI_PACKAGES=robokudo,semantic_digital_twin` for both additional CI steps, or
leave it empty for just the common workspace installation. These switches select
optional dependency installation; they do not exclude packages from the workspace.

For other declared features, `WORKSPACE_EXTRAS` accepts comma-separated
`package:extra` pairs, for example:

```dotenv
WORKSPACE_EXTRAS=semantic_digital_twin:datasets,krrood:visualization
```

Each selection is validated against the chosen checkout's `pyproject.toml` files
and installed using `uv sync --active --inexact --package PACKAGE --extra EXTRA`.
This preserves the previously installed environment. Available extras depend on
the branch: consult the package's `[project.optional-dependencies]`. Optional
features such as Drake/IRIS, vision libraries, datasets, documentation tooling,
and graphical tools are selected explicitly instead of installing every extra.
Re-run `./cramployer setup` after changing dependency selections.

`TORCH_MODE=cpu` retains CI's explicit CPU index. `TORCH_MODE=default` removes
that index and uses the normal package index, allowing GPU-capable PyTorch wheels.
Verify compatibility with your host GPU and driver when enabling NVIDIA access.

Extraction checks expected CI step names and rejects embedded GitHub expressions
that need review. It is not a general GitHub Actions runner: tests, Actions, job
conditions, and unrelated steps are not executed. Selected installation commands
run as root during the image build, with CI's `sudo` calls treated as ordinary
commands. Use a checkout you trust. A changed workflow structure can require an
update to `installation.py`.

## Container environment variables

Use `.env` to configure cramployer itself (source paths, hardware options, build
settings). Use the optional, gitignored `container.env` for variables passed into
containers, including application settings and ROS middleware selection:

```bash
cp container.env.example container.env
```

For example, put these lines in `container.env`:

```dotenv
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
MY_APPLICATION_SETTING=value
```

Open a new `./cramployer shell` to apply changes; no image rebuild is needed.
The file is optional and can be empty. Leave `RMW_IMPLEMENTATION` unset to use
ROS's default middleware. Existing containers keep their original environment;
root shells inherit the environment of the container they attach to.

These variables are passed to all runtime containers, including ROS/ORM build
containers, but not to Docker image builds. Values explicitly configured in
Compose's `environment` section take precedence, so continue setting managed
values such as `ROS_DOMAIN_ID` and `BUILD_PARALLEL_WORKERS` through `.env`.
This feature requires Docker Compose 2.24 or newer for optional env files.

## Daily development

Open the environment:

```bash
./cramployer shell
```

Host edits are visible immediately. Restart the Python process you launched to
reload imported modules; there is no managed application to restart. For example,
inside the shell you can run RoboKudo:

```bash
python -u robokudo/src/robokudo/scripts/main.py _ae=foundation_stereo _headless
```

You choose its analysis engine and headless mode on the command line. For another
application, run its normal Python or ROS command. Output stays in this terminal.
You can also run one-off commands from the host:

```bash
./cramployer shell python -c 'import sys; print(sys.executable)'
```

Every `shell` invocation creates an independent container. Source files, ROS build
volumes, and the application cache are shared between them. Avoid concurrent ROS
builds, and stop applications that consume generated interfaces before regenerating
them. GUI/device changes apply to the next container you open. Existing shells
continue using the image and settings with which they started.

Ordinary Python changes require no image or ROS rebuild. Changes to ROS package
code, interfaces, or metadata use `./cramployer ros-build`; there is no symlink
installation. Changes to dependencies or the `random_events` C++ extension require
`./cramployer setup`. ORM changes use `./cramployer orm-build`.

Source changes persist on the host, and ROS build/install/log data and the user
cache live in Docker volumes. Ad hoc changes inside a container, including package
installations, disappear when you exit it. Lasting dependencies belong in the
installation configuration. The image's Python environment is owned by root;
perform normal development as the configured host user.

## Root access for temporary installations

Keep your normal development shell open. From another host terminal, run:

```bash
./cramployer root-shell
```

This attaches to the existing container as root and automatically sources
`/deployment/environment.sh`, preserving access to ROS and the CRAM virtualenv.
You can install system packages with `apt-get` directly. They become available
in the normal shell immediately and disappear when its container is removed.
Changes in host-mounted folders or shared volumes still persist.

If several development containers are running, the command lists their names
and exits. Rerun it with the selected container name:

```bash
./cramployer root-shell cramployer-cram-run-CONTAINER
```

The command attaches only to this project's running development containers.
Exit the root shell to return to your host terminal; the original development
shell keeps running. Put permanent dependencies in the image installation.

## NVIDIA, GUI, and host prerequisites

Use Ubuntu with Python 3.12 or newer, Bash, Git, Linux Docker Engine with Compose
and Buildx, and enough disk space for ROS, Python dependencies, build cache,
and optional CUDA libraries. Your account
must have Docker access. Optional NVIDIA support requires a host driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
configured for Docker.

ROS does not need to be installed on the host. The container provides the Ubuntu
and ROS environment, so deployment on other Linux distributions may also be
possible. Other host operating systems remain unvalidated; the current setup
assumes Linux host networking, IPC, user/group identifiers, and optional X11 access.

Enable hardware access independently in `.env`:

```dotenv
ENABLE_NVIDIA=true
ENABLE_GUI=true
```

The NVIDIA override exposes compute and graphics capabilities. Inside a shell,
use `nvidia-smi` and `python -c 'import torch; print(torch.cuda.is_available())'`
to check GPU access. Optional NVIDIA and GUI support remain disabled by default.

GUI mode supports X11 and XWayland and requires host `xauth`. Open shells from
your graphical desktop session with valid `DISPLAY` and `XAUTHORITY`. The script
copies the session cookie to `.state/Xauthority` without changing global X-server
permissions. Treat `.state/` as private credentials. Native Wayland-only sessions
are not configured. GUI forwarding makes windows possible; each application still
chooses whether to open them.

Host networking avoids container port mappings. Match `ROS_DOMAIN_ID` with the
robot/host and use compatible middleware, discovery, and QoS settings. Discovery
defaults to `SUBNET`; firewall and multicast restrictions still apply. Host IPC
supports DDS shared-memory communication, subject to participant permissions.

The default container limits are 16 GiB RAM and four CPUs, with two concurrent ROS
build packages. Adjust `.env` for the PC. These limits apply to shells and one-off
build containers; Docker image builds use the builder's resources.

## Updating and rebuilding

```bash
cd /path/to/cognitive_robot_abstract_machine
git pull
git submodule update --init --recursive
cd scripts/cramployer
./cramployer setup
./cramployer shell
```

Setup reads the checkout's current Dockerfile, CI commands, and dependency files.
Docker reuses matching build layers and its download cache. Python environments
are rebuilt in the image; an old virtualenv volume cannot persist obsolete
packages across base-image updates. Running shells continue using their old image
until you exit and open a new one.

Host ROS sources are not overwritten or automatically pulled. Review and update
them if upstream setup scripts change ROS repositories or revisions. Incremental
colcon builds do not remove artifacts from deleted packages. Use a fresh
`ROS_SOURCE`, an unused `VOLUME_PREFIX`, and `VOLUMES_EXTERNAL=false` in `.env`
for an independent source/build set.

To force unchanged base-image setup steps to run against remote repositories:

```bash
docker build --pull --no-cache -t cramployer-base:local \
  ../../.github/docker
./cramployer setup
```

Volumes survive shell exit. Avoid removing Docker volumes unless you intend to
discard their build products or cached data. If migrating from the old managed
RoboKudo service, stop/remove that old container separately; new shell commands do
not manage existing applications.

## Validation

Host tests live in `test/cramployer_test/` and run in the CI library matrix.
From the repository root, run the focused checks with:

```bash
python -m pytest --orm-build never \
  test/cramployer_test scripts/cramployer/container_tests
```

Compose configuration checks skip when Docker is unavailable; they do not need a
running Docker daemon. Native extension smoke tests also run during `setup`.

Focused pytest checks cover CI extraction, optional feature selection, isolated
installation working directories, extensible command and feature dispatch,
desktop authorization, source-container cleanup, and host bootstrap without site
packages. Commands that build or run containers are mocked; Compose configuration
checks run locally. No services or credentials are required. Setup also runs
container tests that check the compiled native extension with host source mounted.
Application-specific ROS, GPU, and GUI behaviour still requires a smoke test for
your application and hardware.

## Renaming an existing installation

Existing source paths in `.env` can remain unchanged. `VOLUME_PREFIX` selects the
Docker volumes to reuse: new installations default to `cramployer`, while migrated
installations can retain `cram-deployment` with `VOLUMES_EXTERNAL=true` to reuse
existing ROS build data and caches. New installations use `VOLUMES_EXTERNAL=false`.
The local folder and Docker image tags use `cramployer`. A Git hosting repository
rename is separate; update the Git remote URL after renaming it on the server.

## Installer architecture

The Bash entry point loads `.env` and starts the Python launcher. Host launcher
modules use only the Python standard library, Git, and Docker, without importing
workspace packages or requiring pip-installed host dependencies. Python
installation runs inside the ROS image, before workspace dependencies have been
installed. The installer modules therefore depend only
on the standard library and the image's existing PyYAML and typing_extensions
packages; they do not require an installed CRAM workspace.

`exceptions.py` deliberately defines standalone dataclass exceptions inheriting
from `ValueError`, with error descriptions and corrective advice. It cannot use
`krrood.exceptions.DataclassException`: krrood is one of the packages this code
is about to install, and importing it would require an already installed workspace.
This bootstrap requirement is the approved exception to the repository's
exception-base-class rule. Host launcher modules and shared exceptions also use
standard-library `typing.TYPE_CHECKING`, rather than requiring typing_extensions
before the development environment exists. This is the approved exception to the
repository's typing-import rule for host bootstrap code.

`workflow.py` parses CI steps, `workspace.py` validates optional dependency
selections, and `installation.py` selects and executes the installation. Shell
commands supplied by cramployer live in separate `.sh` files.

`launch.py` selects implementations through explicit registries.
`DeploymentCommand` in `commands.py` defines the deployment action contract, and
`ContainerFeature` in
`features.py` defines independently composed hardware features. Add an
implementation to the corresponding registry to extend either family.
`DependencyIndex` in `dependency_index.py` defines PyTorch index policies, with
separate implementations for processor-only and default package indexes.
