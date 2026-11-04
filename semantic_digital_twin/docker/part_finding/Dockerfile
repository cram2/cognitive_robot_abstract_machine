# The part finding server, built from the checkout it is in. From the repository root:
#   docker build -f semantic_digital_twin/docker/part_finding/Dockerfile -t cram-part-finding .
#   docker run -d --name part-finding --restart unless-stopped --gpus all -p 8790:8790 \
#     -v part-finding-cache:/root/.cache cram-part-finding
# The volume keeps the model weights and the answers across restarts and rebuilds.
ARG FROM_IMAGE=ghcr.io/cram2/cognitive_robot_abstract_machine:jazzy
FROM $FROM_IMAGE

# The CUDA version of the torch build; it has to be supported by the host's driver.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

SHELL ["/bin/bash", "-c"]
ENV UV_CACHE_DIR=/opt/ros/.uv_cache
COPY . /opt/cram
WORKDIR /opt/cram
RUN source /opt/ros/cram-env/bin/activate \
    && uv sync --active \
    && (cd random_events && python setup.py build_ext --inplace) \
    && uv pip install torch torchvision --index-url $TORCH_INDEX \
    && uv pip install -e "semantic_digital_twin[vision]" \
    && rm -rf $UV_CACHE_DIR

EXPOSE 8790
CMD ["semantic-digital-twin-part-finding-server"]
