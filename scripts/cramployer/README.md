# cramployer

**cramployer** (CRAM + deployer) prepares a Docker environment for developing and
running the CRAM workspace. It mounts editable host source
and supports ROS networking, NVIDIA GPUs, and GUI windows.

ROS and workspace dependencies are installed inside Docker. Run `cramployer setup`
to build the environment, then `cramployer shell` to open it and launch your
application.

## Host prerequisites

Use Ubuntu with Python 3.12 or newer, Bash, Git, Linux Docker Engine with Compose
2.24 or newer and Buildx, and enough disk space for ROS, Python dependencies,
build cache, and optional CUDA libraries. ROS does not need to be installed on
the host. Other host operating systems remain unvalidated.

Before setup, verify that your account can access Docker without `sudo`:

```bash
docker info
docker compose version
docker buildx version
```

If `docker info` reports permission denied, follow Docker's
[Linux post-installation instructions](https://docs.docker.com/engine/install/linux-postinstall/)
to configure access, then retry.

## First setup

From the repository root:

```bash
git submodule update --init --recursive
cp scripts/cramployer/.env.example scripts/cramployer/.env
```

Review `scripts/cramployer/.env` for source locations, resource limits, and
hardware options. Preserve an existing `.env` when updating an installation.
Then run:

```bash
./scripts/cramployer/cramployer setup
./scripts/cramployer/cramployer shell
```

The first setup can take substantial time and disk space. It builds images,
installs dependencies, builds ROS, and generates ORM interfaces. The shell opens
with ROS and the Python environment activated, using your host UID and GID so
files you create in the mounted checkout belong to you.

Source directories on the host are mounted into the container:

| Source | Default host location | Container location |
| --- | --- | --- |
| CRAM (`CRAM_SOURCE`) | The checkout containing cramployer | `/workspace/cram` |
| ROS packages (`ROS_SOURCE`) | `~/cramployer-workspace/ros_ws/src` | `/workspace/ros_ws/src` |

Setup populates the ROS source directory when it is empty and leaves existing
contents untouched. Later setup runs warn if host source files differ from the
rebuilt image, including added or removed packages and changed files. Git metadata
and Python caches are ignored. Both host locations can be changed in `.env`;
they can share a parent directory. The container locations stay the same.

For the commands below, enter the launcher directory:

```bash
cd scripts/cramployer
```

Local `.env`, `container.env`, and `.state/` are ignored by Git.

## Command reference

| Command | What it does |
| --- | --- |
| `./cramployer setup` | Build images, install dependencies, build ROS, and regenerate ORM interfaces. |
| `./cramployer shell` | Open a new interactive container with ROS and Python activated. Exiting removes that container. |
| `./cramployer shell COMMAND ...` | Run a command in a temporary container. |
| `./cramployer root-shell [CONTAINER]` | Open a root shell in an existing development container, with ROS and Python activated. |
| `./cramployer ros-build` | Rebuild the ROS workspace. |
| `./cramployer orm-build` | Regenerate ORM interfaces. |
| `./cramployer config` | Show the effective Compose configuration. |
| `./cramployer help` | Show the available commands. |

## Daily development

Open the environment:

```bash
./cramployer shell
```

Launch your application inside the shell. For example, to run RoboKudo:

```bash
python -u robokudo/src/robokudo/scripts/main.py _ae=demo _headless
```

Host edits are visible immediately. Restart the Python process to reload imported
modules. Application output stays in this terminal. You can also run one-off
commands from the host:

```bash
./cramployer shell python -c 'import sys; print(sys.executable)'
```

Each shell creates an independent container. Source files, ROS build volumes, and
the application cache are shared. Avoid concurrent ROS builds, and stop applications
that consume generated interfaces before regenerating them. Changes to images,
environment settings, or hardware access apply to newly opened containers.

| Change | Action |
| --- | --- |
| Python source | Restart the application. |
| ROS package code, interfaces, or metadata | Run `./cramployer ros-build`. |
| ORM definitions | Run `./cramployer orm-build`. |
| Dependencies or the `random_events` C++ extension | Run `./cramployer setup`, then open a new shell. |

Source changes persist on the host. ROS build/install/log data and the user cache
persist in Docker volumes. Packages installed inside a running container disappear
when you exit it; put lasting dependencies in the installation configuration.
Perform normal development as your configured host user.

### Temporary root access

Keep your development shell open. From another host terminal, run:

```bash
./cramployer root-shell
```

This attaches as root with ROS and Python activated. Install system packages with
`apt-get` directly; they become available in the development shell immediately.
Exit the root shell to return to the host without closing the development shell.

If several development containers are running, the command lists their names.
Select one explicitly:

```bash
./cramployer root-shell cramployer-cram-run-CONTAINER
```

## Dependencies and optional features

### System packages

Add lasting system dependencies to `apt-packages.txt`, one package per line.
Blank lines and `#` comments are supported. Run `./cramployer setup` after changes.

### Python dependencies

CI and cramployer use the same `scripts/install_dependencies.sh` in the selected
checkout. All workspace packages and the root development dependencies are installed.
Optional dependencies are configured in `.env`:

```dotenv
CI_PACKAGES=robokudo
TORCH_MODE=default
WORKSPACE_EXTRAS=
```

`CI_PACKAGES=robokudo` adds RoboKudo's optional dependencies.
Use `CI_PACKAGES=robokudo,semantic_digital_twin` to also install Drake, or leave
it empty for the common workspace dependencies. This setting selects optional
dependencies; all workspace packages are still installed.

`WORKSPACE_EXTRAS` accepts comma-separated `package:extra` pairs, for example:

```dotenv
WORKSPACE_EXTRAS=semantic_digital_twin:datasets,krrood:visualization
```

Consult each package's `[project.optional-dependencies]` in `pyproject.toml` for
available extras. `TORCH_MODE=cpu` selects CPU-only PyTorch wheels;
`TORCH_MODE=default` uses the normal package index, allowing GPU-capable wheels.
Run `./cramployer setup` after changing dependency selections. Setup executes
installation commands from the selected checkout, so use a checkout you trust.

## Container environment variables

Use `.env` for cramployer settings such as source paths, hardware options,
`ROS_DOMAIN_ID`, and `BUILD_PARALLEL_WORKERS`. Use the optional `container.env`
for application settings and ROS middleware selection:

```bash
cp container.env.example container.env
```

For example:

```dotenv
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
MY_APPLICATION_SETTING=value
```

Open a new shell to apply changes; no image rebuild is needed. Leave
`RMW_IMPLEMENTATION` unset to use ROS's default middleware. These variables also
apply to ROS/ORM build containers, but not to image builds. Settings explicitly
managed by Compose take precedence over values in `container.env`.

## NVIDIA, GUI, and ROS networking

NVIDIA support requires a host driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
configured for Docker. Enable GPU and GUI access independently in `.env` before
setup or opening a shell:

```dotenv
ENABLE_NVIDIA=true
ENABLE_GUI=true
```

Both options are disabled by default. Inside a shell, use `nvidia-smi` and
`python -c 'import torch; print(torch.cuda.is_available())'` to check GPU access.

GUI mode supports X11 and XWayland and requires host `xauth`. Open shells from
your graphical desktop session with valid `DISPLAY` and `XAUTHORITY`.
Treat `.state/` as private credentials. Native Wayland-only sessions are not
configured.

Containers use host networking and IPC. Match `ROS_DOMAIN_ID` with the robot/host
and use compatible middleware, discovery, and QoS settings. Discovery defaults
to `SUBNET`; firewall and multicast restrictions still apply.

The default container limits are 16 GiB RAM and four CPUs, with two concurrent ROS
build packages. Adjust `.env` for the PC. These limits apply to shells and one-off
build containers; Docker image builds use the builder's resources.

## Updating and rebuilding

From the repository root:

```bash
git pull
git submodule update --init --recursive
cd scripts/cramployer
./cramployer setup
./cramployer shell
```

Docker reuses matching build layers and its download cache. Host ROS sources are
not overwritten or automatically pulled. If setup warns that sources differ from
the image, review your local changes and update the sources manually, or select a
fresh `ROS_SOURCE` and `VOLUME_PREFIX` in `.env` and rerun setup. This keeps the old
sources and build data available. Incremental ROS builds retain artifacts from
deleted packages.

Volumes survive shell exit. Remove them only if you intend to discard their build
products or cached data.

## Advanced usage

### Separate source checkouts and storage

Set absolute `CRAM_SOURCE` and `ROS_SOURCE` paths in `.env` to use other checkouts.
Setup clones `REPOSITORY_URL` at `REPOSITORY_BRANCH`, including submodules, if
`CRAM_SOURCE` does not exist. Existing checkouts are used as-is.

`VOLUME_PREFIX` controls the Docker volume names for ROS build/install/log data
and the user cache; it defaults to `cramployer`. Use a fresh `ROS_SOURCE` and an
unused prefix for independent ROS sources and build storage. Compose creates the
named volumes when needed and reuses them on later runs.

### Rebuilding without cached layers

From `scripts/cramployer/`, force the base image to rebuild against remote
repositories with:

```bash
docker build --pull --no-cache -t cramployer-base:local \
  ../../.github/docker
./cramployer setup
```
