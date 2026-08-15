/* ============================================================================
 * core/recording-mode.js — the lifecycle of one captured live run.
 *
 * Attaching to a running demo starts capturing it in the background (see
 * cramera.live.visualization); this module only decides what the recording
 * controls (stop / discard / save) show and do for a given /recording status,
 * mirroring how live-mode.js centralizes the live-view control's own logic.
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

  function isRecordingScene(sceneName) {
    return sceneName === SCENE_NAME;
  }

  global.RecordingMode = {
    SCENE_NAME: SCENE_NAME,
    STATE: STATE,
    SPEED_OPTIONS: SPEED_OPTIONS,
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

    /* A playback speed multiplier, or the default when it is not one of the offered
       options (an unrecognized stored preference, or nothing chosen yet). */
    clampSpeed: function (speed) {
      return SPEED_OPTIONS.indexOf(speed) >= 0 ? speed : 1;
    },
  };
})(window);
