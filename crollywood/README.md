# crollywood: narrated, subtitled videos

`crollywood` makes narrated, subtitled mp4 videos out of scenes you write: the
results of an experiment, a demonstration of a robot, a walk through a piece of code, an
explainer of an idea. A
video is a `Storyboard`: scenes in order, each with the lines of narration that start
with it. `VideoProduction` then:

- holds each slide long enough for its lines to be said,
- lays the scenes on a timeline, dissolving each into the next,
- has the narrator say the lines,
- burns the subtitles into the picture, or adds them as a track the viewer can switch
  off,
- encodes the picture as H.264 to a byte budget,
- and joins the narration to it as AAC.

## A minimal video

```python
from datetime import timedelta
from pathlib import Path

from crollywood.narration import NarratedLine, NarratedScene, Storyboard
from crollywood.production import VideoProduction
from crollywood.slides import TabularDataRow, TabularDataSlide, TextCard
from crollywood.timeline import Transition

storyboard = Storyboard(
    [
        NarratedScene(
            role_taker=TextCard(
                "Pouring in simulation", "12 runs", held_for=timedelta(seconds=2)
            ),
            lines=(NarratedLine("The robot pours into a cup twelve times."),),
        ),
        NarratedScene(
            role_taker=TabularDataSlide(
                "Results", (TabularDataRow("Poured without spilling", "11 of 12"),)
            ),
            lines=(NarratedLine("It spilled once."),),
            transition=Transition.CUT,
        ),
    ]
)
VideoProduction(storyboard).written_to(Path("pouring.mp4"))
```

`test/crollywood_test/test_production.py` makes a video just like this one, with
a test voice in place of the speech model.

## Videos and scenes

Everything that plays is a `video.Video`: it has a `duration` and a `frame_at(moment)`,
both in `timedelta`s. A slide, footage, a timeline and a video file on disk
(`encoding.VideoFile`) are all videos. A new kind of picture is a subclass of
`video.Video`; a slide subclasses `slides.HeldSlide` instead, which lasts as long as it is
held (`held_for`), and the storyboard holds it longer when its lines need the time. A
video whose length comes from what it shows, such as footage, keeps that length, and a
line that would still be being said when the next begins is refused with
`NarrationOverrun`.

A `timeline.TimelineScene` is the part a video plays in a timeline: `TimelineScene(role_taker=video)`.
Any video can be a scene, a whole timeline included, so a chapter can be one scene of a
longer video. Each scene says how the scene before passes into it (`transition`):
`Transition.DISSOLVE` by default, or `Transition.CUT`, for example where the two would
write different text in one place. `narration.NarratedScene` is a scene with the lines
that start with it.

Videos that come with the package:

- `slides.SingleFrameSlide`: one picture held.
- `slides.TextCard`: a heading and a line. Use it as a title card or an end card.
- `slides.TabularDataSlide`: a table of named rows and their values, such as results.
- `footage_playback.FootagePlayback`: a stretch of a recording filling the frame, sped
  up with the speed shown on a badge, and optionally a second view of the same stretch
  in the corner.
- `footage_playback.TitleOverFootage`: a title over footage, fading out.

`footage.ViewOfTheRun` is a stretch of any video, for example an `encoding.VideoFile` of
what `semantic_digital_twin.adapters.mujoco_video_recording` records of a simulated run.
`footage.Framing` cuts margins off its pictures.

Drawing is on `frame.Frame`, the pixels of one picture: pasting and fitting pictures,
shading, outlines, lines and arrows. Colours are `Color`, positions `Point2` and
rectangles `PlanarBoundingBox` of `semantic_digital_twin`, in pixels from the top left
corner (`Frame.box` makes a box from its left, top, width and height).
`typesetting.Typesetting` writes text, with the three letter sizes in
`typesetting.TextSize`, and `code.CodeTypesetting` writes a line of Python coloured as
an editor would. `Resolution.standard()` is 1280 × 720. Every scene leaves the bottom
band of the frame clear for the subtitles (`Resolution.stage_height`).

## The look

A `theme.VideoTheme` holds the colours and fonts the slides and captions are drawn
in: the page, text, muted, hairline, panel, marker and accent colours, how code is
coloured, and a `lettering.Typeface` (DejaVu by default, since matplotlib ships it). Slides
take one as `theme=`. `VideoTheme.neutral()`, dark text on white, is the default;
`VideoTheme.dark()` is light text on a near-black page. Footage keeps white text and dark
badges whatever the theme.

Burned-in subtitles are drawn in the band every scene keeps clear
(`captions.BandCaption`, given the video's theme), or in white on a shaded box over
the bottom of the picture (`captions.BoxedCaption`); pass the style to
`VideoProduction(captions=...)`. What a caption is read on over a picture is a
`shading.Shade`, given as `shade=`: an `EvenShade`, its corners rounded by
`corner_radius`, or a `GradientShade` that deepens from its top edge.

The frame rate, the dissolve between scenes and what is drawn over every frame are a
`timeline.TimelineStyle`, passed as `VideoProduction(timeline_style=...)`. Anything
drawn over the whole video rather than one scene is a `timeline.Overlay`, passed as
`TimelineStyle(overlays=[...])`. `overlays.ProgressBar(theme=...)` is a thin
bar along the top that fills in the accent colour as the video plays, across every
scene.

## Keeping slow parts

Slow parts of a scene, such as decoded recordings or rendered pictures, can be stored
with `cache.SceneCache`. It keeps them in the directory the environment variable
`CROLLYWOOD_CACHE` names (`cache.EnvironmentVariable.CACHE_ROOT`), or in
`~/.cache/crollywood` by default, so a second render does not recompute them.

## The narrator

By default the lines are spoken by the Kokoro speech model, run locally. Install it
with the `narrator` extra (`pip install crollywood[narrator]`). Then download
`kokoro-v1.0.onnx` and `voices-v1.0.bin` from the
[kokoro-onnx model release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)
into the `voice` folder of the cache. `KokoroTextToSpeech(voice=..., speed=...)` picks the
voice and its pace. Each line is trimmed of the silence around it, keeping a short
breath, and levelled to one loudness. Every line it says is kept in the cache, so
re-rendering a video reuses the same speech.

Any subclass of `speech.TextToSpeech`, which says a line with `speaks(text, pace) -> SoundChannel`,
can narrate instead. A storyboard with no lines makes a silent video, and then no
speech model is needed.

`NarratedLine(written, spoken=...)` is subtitled as written and said as spoken, for when the
two differ (for example numbers). `NarratedLine(..., pace=0.9)` says a line slower than the
voice's own speed, for one the viewer needs time to follow. To time steps of a scene
to the narration, measure the lines with `NarratedLine.start_times`. `Narration.report()` prints when each line
starts and how much silence follows it.

## Size

Sizes are `krrood.units.DataSize` and bit rates `krrood.units.BitRate` (`DataSize.of_bytes(40_000)`,
`BitRate(DataSize(64_000))` for 64 kilobits a second). The file may take
`VideoProduction.picture_bit_rate` over its length; for a video that must fit another
size, such as a venue's, subclass `VideoProduction` and override `size_allowed_for`.
`encoder=H264Encoder(preset=EncoderPreset.FAST)` encodes a quick draft.
