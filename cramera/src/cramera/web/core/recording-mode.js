/* ============================================================================
 * core/recording-mode.js — the lifecycle of one captured live run.
 *
 * Attaching to a running demo starts capturing it in the background (see
 * cramera.live.visualization); this module only decides what the recording control
 * shows and does for a given /recording status -- one button that ends the capture and
 * then opens the panel naming, trimming and keeping it -- mirroring how live-mode.js
 * centralizes the live-view control's own logic.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const SCENE_NAME = '__recording__';
  /* Name of the reserved scene a finalized recording is bundled under.
     Must match cramera.paths.RECORDING_SCENE_NAME. */

  const STATE = { IDLE: 'idle', RECORDING: 'recording', FINALIZED: 'finalized' };
  /* Mirrors cramera.live.recording.RecordingState's values. */

  const NAME_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;
  /* Mirrors cramera.onboard.scene_index.SCENE_NAME_PATTERN. */

  const SPEED_OPTIONS = [0.5, 1, 2, 4];
  /* Playback speed multipliers the speed selector offers. */

  const STOP = 'stop';
  /* Capture is running: the control ends it. */

  const SAVE = 'save';
  /* Capture has ended: the control opens the panel that names, trims and keeps it. */

  const DESTINATION = { LOCAL: 'local', SHARED: 'shared' };
  /* Mirrors cramera.live.recording_storage.SceneDestination's values: whether a saved
     episode stays on this machine or lands in the scenes root others read too. */

  function isRecordingScene(sceneName) {
    return sceneName === SCENE_NAME;
  }

  global.RecordingMode = {
    SCENE_NAME: SCENE_NAME,
    STATE: STATE,
    SPEED_OPTIONS: SPEED_OPTIONS,
    STOP: STOP,
    SAVE: SAVE,
    DESTINATION: DESTINATION,
    isRecordingScene: isRecordingScene,

    /* Whether the recording controls have anything to show at all. */
    controlsVisible: function (status) {
      return !!status && status.state !== STATE.IDLE;
    },

    /* What the stop control says, depending on whether capture is still running. */
    stopButtonLabel: function (state) {
      return state === STATE.RECORDING ? '⏹ Stop recording' : '⏹ Stopped';
    },

    /* Whether pressing stop does anything right now. */
    canStop: function (state) {
      return state === STATE.RECORDING;
    },

    /* Whether the recording can be saved: only once capture has been finalized. */
    canSave: function (state) {
      return state === STATE.FINALIZED;
    },

    /* Whether there is anything left to throw away. */
    canDiscard: function (state) {
      return state !== STATE.IDLE;
    },

    /* Whether a user-typed name is safe to save the recording under. */
    isValidSaveName: function (name) {
      return typeof name === 'string' && NAME_PATTERN.test(name) && name !== SCENE_NAME;
    },

    /* What the single recording control does right now: end the capture, then keep
       what it captured. Nothing to do once the run has been saved or thrown away. */
    controlAction: function (state) {
      if (state === STATE.RECORDING) return STOP;
      if (state === STATE.FINALIZED) return SAVE;
      return null;
    },

    /* What that control says it will do. */
    controlLabel: function (state) {
      return state === STATE.RECORDING ? '⏹ Stop recording' : '💾 Save episode…';
    },

    /* Whether the episode can be trimmed from the scene `sceneName` is showing. Only
       the episode's own scene will do: the frames a trim names are indices into the
       run being cut, so another scene's trajectory would cut it at the wrong places.
       An episode the viewer has no trajectory for can only be saved whole. */
    canTrim: function (sceneName, frameCount) {
      return isRecordingScene(sceneName) && frameCount > 0;
    },

    /* The trim a freshly finalized run opens with: all of it. */
    wholeTrim: function (frameCount) {
      return { first: 0, last: frameCount - 1 };
    },

    /* Whether a trim selects a real stretch of a run of `frameCount` frames. */
    isValidTrim: function (trim, frameCount) {
      return !!trim && frameCount > 0
        && trim.first >= 0 && trim.last < frameCount && trim.first <= trim.last;
    },

    /* Whether a trim keeps everything, so saving can skip re-bundling the run. */
    isWholeTrim: function (trim, frameCount) {
      return !!trim && trim.first === 0 && trim.last === frameCount - 1;
    },

    /* How much of the run a trim keeps, for the save panel to show. */
    trimSummary: function (trim, frameCount, framesPerSecond) {
      const kept = trim.last - trim.first + 1;
      const seconds = framesPerSecond > 0 ? kept / framesPerSecond : 0;
      return kept + ' frames · ' + seconds.toFixed(2) + ' s';
    },

    /* A playback speed multiplier, or the default when it is not one of the offered
       options (an unrecognized stored preference, or nothing chosen yet). */
    clampSpeed: function (speed) {
      return SPEED_OPTIONS.indexOf(speed) >= 0 ? speed : 1;
    },
  };
})(window);
