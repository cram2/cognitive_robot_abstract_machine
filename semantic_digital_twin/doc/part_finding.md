# Finding the parts of an object

An object often comes as a single mesh, while a robot needs its parts: the handle of a
mug to grasp it, the legs of a table to avoid them. {py:class}`~semantic_digital_twin.part_finding.annotation_part_finder.AnnotationPartFinder`
gives an annotation the parts its part-whole relationships call for, by finding them in
the shape of its root body and splitting each one off into a body and annotation of its
own.

```python
from semantic_digital_twin.part_finding.annotation_part_finder import AnnotationPartFinder

parts = AnnotationPartFinder().find_parts(mug)
mug.handle  # the handle that was found, or still None
```

In a pipeline, the step
{py:class}`~semantic_digital_twin.pipeline.part_finding.FindParts` does the same for
every annotation of a type, such as `FindParts(whole_type=Mug)`.

Only relationships the annotation has no part in yet are looked at. An optional part,
such as a mug's handle, and each of any number of parts, such as a table's legs, is
added only when the finder is at least `minimum_confidence` sure of it.

A relationship that holds a single part, such as a mug's handle, gets the piece the
finder is most sure of. Annotations that may have several parts of a kind hold them in
a list, such as a pot's `handles`, and get each piece found as a part of its own.

## Collision shapes

The parts are cut out of the visual shapes, and the pieces collide as they look. A body
whose collision shapes differ from its visual shapes, such as the convex decompositions
most RoboCasa objects come with, is not split, since that would throw its collision
shapes away. Either find the parts before decomposing, so that the decomposition runs
on each piece:

```python
from semantic_digital_twin.pipeline.mesh_decomposition.coacd import COACDMeshDecomposer

AnnotationPartFinder().find_parts(mug)
COACDMeshDecomposer().apply(mug._world)
```

or let the finder decompose the pieces again:

```python
AnnotationPartFinder(decomposer=COACDMeshDecomposer()).find_parts(mug)
```

## How the parts are found

The parts are found by a combination of Qwen3-VL and SAM 2.1. In a benchmark on 93
kitchen objects this found 78 of 87 handles and marked nothing on any of six cups
without one, at about 5 seconds per object on an A40.

Every setting is a field of
{py:class}`~semantic_digital_twin.part_finding.vision_language.VisionLanguagePartLocator`
and the classes it is made of.

## Where the models run

By default `AnnotationPartFinder` asks the part finding server that runs on an internal
server of the institute, which needs nothing beyond this package. Set the environment
variable `SEMANTIC_DIGITAL_TWIN_PART_FINDING_SERVER_URL` to its address, which you get
from the institute, or give a server directly:

```python
from semantic_digital_twin.adapters.part_finding.client import RemotePartLocator

AnnotationPartFinder(locator=RemotePartLocator(base_url="http://localhost:8790"))
```

The server keeps every answer, so the same question about the same shape is answered
from its cache.

To run the models on your own GPU instead, install the `vision` extra, after installing
the torch build that matches your driver's CUDA version (`nvidia-smi` shows it):

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e "semantic_digital_twin[vision]"
```

```python
from semantic_digital_twin.adapters.part_finding.vision_language_models import (
    QwenBoxDetector, QwenModel, Quantization, Sam2BoxSegmenter,
)
from semantic_digital_twin.part_finding.vision_language import (
    VisionLanguagePartLocator,
)

locator = VisionLanguagePartLocator(
    detector=QwenBoxDetector(model=QwenModel.QWEN3_VL_4B, quantization=Quantization.FOUR_BIT),
    segmenter=Sam2BoxSegmenter(),
)
AnnotationPartFinder(locator=locator)
```

The 8B model needs about 19 GB of GPU memory; the 4B model in 4 bits fits into 8 GB
and found 73 of the 87 handles in the benchmark. On a CPU (`device=Device.CPU`) a
single object takes minutes.

## Running the server

The server is the same locator behind http. Start it with
`semantic-digital-twin-part-finding-server` (see `--help` for the model, quantization,
device, port and cache), or build the image in
`semantic_digital_twin/docker/part_finding`. It checks no credentials, so run it only
inside a trusted network.
