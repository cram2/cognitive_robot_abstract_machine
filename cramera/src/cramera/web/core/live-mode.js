/* ============================================================================
 * core/live-mode.js — where watching a running demo is allowed to happen.
 *
 * A recorded scene and a live demo are two different worlds, and the two things a
 * user can do to an object in them are not the same act: dragging in a recording
 * offsets a playback pose on the client, while dragging live posts to the bridge and
 * moves the actual simulated world. Mixing them would mean one arrangement silently
 * overwriting the other, so the live pose stream is only ever attached on the reserved
 * live scene -- which the bridge bundles fresh from the demo's current world on every
 * attach. From anywhere else, going live means going *to* that scene.
 *
 * Keeping the rule here makes it one decision instead of a condition repeated at every
 * site that can turn the stream on.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const SCENE_NAME = '__live__';
  /* Name of the reserved scene the bridge bundles the running demo into.
     Must match cramera.paths.LIVE_SCENE_NAME. */

  const NAVIGATE = 'navigate';
  /* Going live from here means loading the live scene first. */

  const TOGGLE = 'toggle';
  /* Already on the live scene: the pose stream can be switched on and off in place. */

  function isLiveScene(sceneName) {
    return sceneName === SCENE_NAME;
  }

  global.LiveMode = {
    SCENE_NAME: SCENE_NAME,
    NAVIGATE: NAVIGATE,
    TOGGLE: TOGGLE,
    isLiveScene: isLiveScene,

    /* Whether the live pose stream may drive the scene `sceneName` is showing. */
    attachable: function (sceneName) {
      return isLiveScene(sceneName);
    },

    /* What pressing the live control does on the scene `sceneName` is showing. */
    actionFor: function (sceneName) {
      return isLiveScene(sceneName) ? TOGGLE : NAVIGATE;
    },

    /* What the live control says it will do, so a scene the stream cannot attach to does
       not offer a toggle it would refuse. */
    labelFor: function (sceneName, attached) {
      if (!isLiveScene(sceneName)) return '◉ Live view';
      return attached ? '◉ LIVE — attached' : '◉ Live';
    },

    titleFor: function (sceneName) {
      if (!isLiveScene(sceneName)) {
        return 'Switch to the live view of the running demo — this leaves the recorded '
          + 'scene, and an arrangement made here does not carry over';
      }
      return 'Attach to the running demo (cramera-live bridge) — renders the live world '
        + 'instead of the recording';
    },

    /* Whether the live scene must be rebuilt and reloaded because this page's bundle
       no longer describes the running demo. Two ways that happens:

       - the bridge reports a bundle signature (`info.bundleSignature`, a digest of
         the world model a bundle built right now would serialize) that differs from
         the one this page's bundle carries (`scene.bundleSignature`) — the demo
         attached a different world, its model changed mid-run, or anything else that
         makes the loaded bundle describe a world that no longer runs;
       - the bundle was built before the demo's world was attached
         (`scene.worldBound` not true — a bundle without the flag was written before
         the flag existed and is just as suspect): the moment `info.running` reports
         an attached world, the bundle must be rebuilt against it.

       Only ever true on the live scene while the stream is attached: anywhere else the
       page is showing a recording, which no live event may reload away. */
    needsLiveSceneReload: function (sceneName, attached, scene, info) {
      if (!isLiveScene(sceneName) || !attached || !scene || !info) return false;
      if (typeof scene.bundleSignature === 'string' && typeof info.bundleSignature === 'string'
          && scene.bundleSignature !== info.bundleSignature) return true;
      return scene.worldBound !== true && info.running === true;
    },

    /* Whether the stream should attach by itself right now. A URL naming a recorded
       scene is a deliberate choice and is never auto-attached away from; the live
       scene and the blank landing page are fair game. By default this fires once per
       page (a manual detach stays detached); with the always-live setting it fires
       whenever the stream is down, so the next demo run attaches without a click.

       `hasExplicitScene` is whether the URL names a scene at all; `attachedOnceBefore`
       is whether this page already auto-attached once. */
    shouldAutoAttach: function (sceneName, hasExplicitScene, attached, attachedOnceBefore, alwaysLive) {
      if (attached) return false;
      if (hasExplicitScene && !isLiveScene(sceneName)) return false;
      return alwaysLive ? true : !attachedOnceBefore;
    },

    /* Whether a freshly built live bundle describes the same world as the one this
       page loaded: same models (names, prefixes, robot flags) and the same robot
       identity. A page can sit on the live scene across demo runs, so attaching must
       compare rather than assume. With nothing to compare (page not loaded yet, or
       the rebundle failed) the answer is "same" — reloading on no evidence would
       loop. */
    sameBundle: function (loadedScene, freshScene) {
      if (!loadedScene || !freshScene) return true;
      const identity = function (scene) {
        return JSON.stringify({
          models: scene.models || [],
          robot: scene.robot || null,
        });
      };
      return identity(loadedScene) === identity(freshScene);
    },
  };
})(window);
