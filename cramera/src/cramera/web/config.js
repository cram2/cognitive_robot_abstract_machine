/* ============================================================================
 * config.js — *which* panels are shown *where*. This is the file you edit to swap
 * a visualization: remove an id, add your own (define it via Panels.define in
 * a new panels/<name>/panel.js and include that script in index.html).
 *
 * Slots are the data-slot elements in index.html ('left', 'right'); a slot
 * with several panel ids stacks them vertically.
 * ==========================================================================*/
// The 3D scene stands alone when another page embeds it (another scene view,
// the replay popup) or when it was popped out into its own window (?layout=scene);
// SceneContext reads both off the URL. A ?scene in a *top-level* tab keeps the full
// page, so the user isn't stranded on a bare scene with no way to navigate back.
var _sceneOnly = SceneContext.sceneOnly(window.self !== window.top);
if (_sceneOnly) {
  document.documentElement.classList.add(SceneContext.SCENE_ONLY_CLASS);
}
if (SceneContext.poppedOut()) {
  document.documentElement.classList.add(SceneContext.POPPED_OUT_CLASS);
}
window.CRAMERA_CONFIG = {
  layout: _sceneOnly
    ? { left: ['robot-scene'] }
    : {
        left: ['robot-scene'],
        right: ['eql', 'graph'],
      },
};
