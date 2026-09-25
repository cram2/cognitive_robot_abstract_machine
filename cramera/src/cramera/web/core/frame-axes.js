/* ============================================================================
 * core/frame-axes.js — the in-scene frame display: which frames the viewer draws
 * an axis triad on, how big, and whether their names show.
 *
 * The frames come from the loaded models themselves: every URDF link is a frame,
 * and so is every loose object the bundle or the live bridge put in the scene.
 * Because a triad is parented to the link's own object, it follows the world
 * without any pose plumbing — in recorded playback and live alike.
 *
 * Pure state and lookup rules, no DOM and no three.js, so they are testable
 * under node.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const VISIBLE_KEY = 'cramera.frame-axes-visible';
  /* localStorage key of whether the frame display is on. */

  const NAMES_KEY = 'cramera.frame-axes-names';
  /* localStorage key of whether frame names are drawn next to the triads. */

  const SIZE_KEY = 'cramera.frame-axes-size';
  /* localStorage key of the triad arm length, in metres. */

  const HIDDEN_SOURCES_KEY = 'cramera.frame-axes-hidden-sources';
  /* localStorage key of the sources whose frames are hidden, as a JSON array. */

  const HIDDEN_FRAMES_KEY = 'cramera.frame-axes-hidden-frames';
  /* localStorage key of the individually hidden frames, as a JSON array. */

  const OBJECT_SOURCE = 'objects';
  /* The source name the loose objects share; models use their own name. */

  const DEFAULT_SIZE = 0.15;
  /* Triad arm length a fresh viewer starts at, in metres. */

  const MIN_SIZE = 0.02;
  const MAX_SIZE = 1.0;
  /* The range the size control offers, in metres. */

  /* The three arms of a triad, in the axis-to-colour convention RViz uses. */
  const AXES = [
    { axis: 'x', color: '#ff5c6c' },
    { axis: 'y', color: '#4bd38a' },
    { axis: 'z', color: '#5b8cff' },
  ];

  function readList(storage, key) {
    try {
      const stored = JSON.parse(storage.getItem(key) || 'null');
      return Array.isArray(stored) ? stored : [];
    } catch (error) {
      return [];
    }
  }

  function lookupSet(storage, key) {
    const set = {};
    readList(storage, key).forEach(function (entry) { set[entry] = true; });
    return set;
  }

  function storeSet(storage, key, set) {
    storage.setItem(key, JSON.stringify(Object.keys(set).sort()));
  }

  /* What the display is currently hiding: whole sources, and individual frames. */
  function hidden(storage) {
    return {
      sources: lookupSet(storage, HIDDEN_SOURCES_KEY),
      frames: lookupSet(storage, HIDDEN_FRAMES_KEY),
    };
  }

  /* Hide or show one source and every frame under it; returns the updated state.

     Ticking a source clears the individual choices inside it, so a source whose
     frames were picked one by one is not silently half-hidden the next time it is
     switched back on. */
  function setSourceHidden(storage, source, isHidden, frameIds) {
    const set = lookupSet(storage, HIDDEN_SOURCES_KEY);
    if (isHidden) set[source] = true;
    else delete set[source];
    storeSet(storage, HIDDEN_SOURCES_KEY, set);
    const frames = lookupSet(storage, HIDDEN_FRAMES_KEY);
    (frameIds || []).forEach(function (id) { delete frames[id]; });
    storeSet(storage, HIDDEN_FRAMES_KEY, frames);
    return hidden(storage);
  }

  /* Hide or show one frame on its own; returns the updated state.

     A frame ticked back on also un-hides its source, since a hidden source would
     otherwise keep it off screen and the tick would look ignored. */
  function setFrameHidden(storage, frame, isHidden, source) {
    const frames = lookupSet(storage, HIDDEN_FRAMES_KEY);
    if (isHidden) frames[frame] = true;
    else delete frames[frame];
    storeSet(storage, HIDDEN_FRAMES_KEY, frames);
    if (!isHidden && source) {
      const sources = lookupSet(storage, HIDDEN_SOURCES_KEY);
      delete sources[source];
      storeSet(storage, HIDDEN_SOURCES_KEY, sources);
    }
    return hidden(storage);
  }

  function readFlag(storage, key) {
    return storage.getItem(key) === 'true';
  }

  /* The size within the range the control offers; anything unreadable is the default. */
  function clampSize(value) {
    const size = parseFloat(value);
    if (!isFinite(size)) return DEFAULT_SIZE;
    return Math.min(MAX_SIZE, Math.max(MIN_SIZE, size));
  }

  /* The whole frame display state, as the viewer last left it. */
  function settings(storage) {
    return {
      visible: readFlag(storage, VISIBLE_KEY),
      names: readFlag(storage, NAMES_KEY),
      size: storage.getItem(SIZE_KEY) === null
        ? DEFAULT_SIZE
        : clampSize(storage.getItem(SIZE_KEY)),
    };
  }

  function setVisible(storage, visible) {
    storage.setItem(VISIBLE_KEY, visible ? 'true' : 'false');
    return settings(storage);
  }

  function setNames(storage, names) {
    storage.setItem(NAMES_KEY, names ? 'true' : 'false');
    return settings(storage);
  }

  function setSize(storage, size) {
    storage.setItem(SIZE_KEY, String(clampSize(size)));
    return settings(storage);
  }

  /* The id a frame is tracked by. Two models can each carry a link of the same name
     — a URDF root is routinely called ``world_root`` — so the name alone would let one
     model's frame stand in for the other's. */
  function frameId(source, name) {
    return source + '\u0000' + name;
  }

  /* Every frame of one loaded model: its URDF links, named the way the world names
     them, so a link of a second model with the same link name still reads clearly. */
  function framesOfModel(model) {
    const frames = [];
    model.obj.traverse(function (child) {
      if (!child.isURDFLink || !child.name) return;
      const name = model.prefix ? model.prefix + '/' + child.name : String(child.name);
      frames.push({
        id: frameId(model.name, name),
        name: name,
        source: model.name,
        object: child,
      });
    });
    return frames;
  }

  /* How much of a source is on screen, as its tick box shows it. */
  const SourceState = { ALL: 'all', SOME: 'some', NONE: 'none' };

  /* Every source the display can draw: each loaded model, and the loose objects. */
  function sourcesOf(models, objectMeshes) {
    const sources = (models || []).map(function (model) { return model.name; });
    if (Object.keys(objectMeshes || {}).length) sources.push(OBJECT_SOURCE);
    return sources.sort();
  }

  /* Every frame in the scene: the links of every loaded model, then the loose
     objects, sorted by name so the display is stable across rebuilds. */
  function framesOf(models, objectMeshes) {
    const frames = [];
    (models || []).forEach(function (model) {
      framesOfModel(model).forEach(function (frame) { frames.push(frame); });
    });
    const objects = objectMeshes || {};
    Object.keys(objects).forEach(function (key) {
      frames.push({
        id: frameId(OBJECT_SOURCE, key),
        name: key,
        source: OBJECT_SOURCE,
        object: objects[key],
      });
    });
    return frames.sort(function (one, other) {
      return one.name < other.name ? -1 : one.name > other.name ? 1 : 0;
    });
  }

  /* Only the frames that are neither hidden themselves nor part of a hidden source. */
  function visibleFrames(frames, hiddenState) {
    const state = hiddenState || {};
    const hiddenSources = state.sources || {};
    const hiddenIds = state.frames || {};
    return (frames || []).filter(function (frame) {
      return !hiddenSources[frame.source] && !hiddenIds[frame.id];
    });
  }

  /* The frames of one source, so the settings list can offer them one by one. */
  function framesOfSource(frames, source) {
    return (frames || []).filter(function (frame) { return frame.source === source; });
  }

  /* How a source's own tick box reads: every frame under it on, none of them, or
     some. A source ticked off is 'none' whatever its frames say. */
  function sourceState(frames, source, hiddenState) {
    const own = framesOfSource(frames, source);
    const shown = visibleFrames(own, hiddenState).length;
    if (!shown) return SourceState.NONE;
    return shown === own.length ? SourceState.ALL : SourceState.SOME;
  }

  global.FrameAxes = {
    VISIBLE_KEY: VISIBLE_KEY,
    NAMES_KEY: NAMES_KEY,
    SIZE_KEY: SIZE_KEY,
    HIDDEN_SOURCES_KEY: HIDDEN_SOURCES_KEY,
    HIDDEN_FRAMES_KEY: HIDDEN_FRAMES_KEY,
    SourceState: SourceState,
    OBJECT_SOURCE: OBJECT_SOURCE,
    DEFAULT_SIZE: DEFAULT_SIZE,
    MIN_SIZE: MIN_SIZE,
    MAX_SIZE: MAX_SIZE,
    AXES: AXES,
    clampSize: clampSize,
    settings: settings,
    setVisible: setVisible,
    setNames: setNames,
    setSize: setSize,
    frameId: frameId,
    framesOf: framesOf,
    visibleFrames: visibleFrames,
    framesOfSource: framesOfSource,
    sourceState: sourceState,
    sourcesOf: sourcesOf,
    hidden: hidden,
    setSourceHidden: setSourceHidden,
    setFrameHidden: setFrameHidden,
  };
})(typeof window !== 'undefined' ? window : this);
