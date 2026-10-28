# %% CI dependency image
ARG BASE_IMAGE=cramployer-base:local
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

# %% Repository installation
ARG TORCH_MODE=default
ARG CI_PACKAGES=robokudo
ARG WORKSPACE_EXTRAS=
COPY installation.py dependency_index.py exceptions.py vocabulary.py workflow.py workspace.py installation.sh workspace-extra.sh /deployment/scripts/cramployer/
RUN --mount=type=bind,from=repository,target=/workspace/cram,rw \
    --mount=type=cache,target=/opt/ros/.uv_cache \
    source /opt/ros/overlay_ws/install/setup.bash && \
    source /opt/ros/cram-env/bin/activate && \
    cd /deployment && \
    python -m scripts.cramployer.installation /workspace/cram "${TORCH_MODE}" "${CI_PACKAGES}" "${WORKSPACE_EXTRAS}"

# %% Native extension retained outside the temporary source mount
RUN --mount=type=bind,from=repository,target=/workspace/cram,rw \
    source /opt/ros/cram-env/bin/activate && \
    cd /workspace/cram/random_events && \
    python setup.py build_ext --build-lib "$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"

# %% Deployment-only system dependencies
COPY apt-packages.txt /deployment/apt-packages.txt
RUN apt-get update && \
    sed 's/#.*//' /deployment/apt-packages.txt | \
    xargs -r env DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# %% Host user and writable workspace
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd --non-unique --gid "${HOST_GID}" cramployer && \
    useradd --non-unique --uid "${HOST_UID}" --gid cramployer --create-home --shell /bin/bash cramployer && \
    mkdir -p /workspace/ros_ws/{src,build,install,log} /home/cramployer/.cache && \
    chown -R "${HOST_UID}:${HOST_GID}" /workspace/ros_ws && \
    chown "${HOST_UID}:${HOST_GID}" /home/cramployer/.cache && \
    git config --system --add safe.directory /workspace/cram && \
    printf '\nsource /deployment/environment.sh\n' >> /home/cramployer/.bashrc
COPY environment.sh entrypoint.sh /deployment/
USER cramployer
WORKDIR /workspace/cram
ENTRYPOINT ["bash", "/deployment/entrypoint.sh"]
CMD ["bash"]
