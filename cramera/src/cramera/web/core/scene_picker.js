/* ============================================================================
 * core/scene_picker.js — resolves the header's robot/environment dropdowns
 * to one onboarded scene bundle.
 *
 * There is no independent robot/environment mixing: each onboarded scene is a
 * fixed (robot, environment) recording. This module only looks up, among the
 * scenes actually onboarded, which one matches a chosen (robot, environment)
 * pair — so the two dropdowns in robot_scene/panel.js look independent while
 * only ever landing on a combination that was actually recorded.
 *
 * `scenes` is `scenes/index.json`'s `scenes` array: entries shaped like
 * {name, robot, environment}, where `environment` is null for a bench-only
 * scene (no environment model, e.g. a robot recorded alone on a bench).
 * ==========================================================================*/
(function () {
  'use strict';

  function uniq(values) {
    const seen = {}, out = [];
    values.forEach(function (v) {
      const key = String(v);
      if (!seen[key]) { seen[key] = true; out.push(v); }
    });
    return out;
  }

  // every robot with at least one onboarded scene
  function robots(scenes) {
    return uniq((scenes || []).map(function (s) { return s.robot; }));
  }

  // every environment onboarded together with the given robot (null included
  // for a bench-only recording), in the order scenes first offers them
  function environments(scenes, robot) {
    return uniq(
      (scenes || [])
        .filter(function (s) { return s.robot === robot; })
        .map(function (s) { return s.environment; })
    );
  }

  // the scene bundle recorded for this exact (robot, environment) pair, or
  // null if that combination was never onboarded
  function sceneFor(scenes, robot, environment) {
    const env = environment || null;
    const match = (scenes || []).find(function (s) {
      return s.robot === robot && (s.environment || null) === env;
    });
    return match ? match.name : null;
  }

  // the (robot, environment) a named scene was recorded with, or null if the
  // name isn't in the index
  function describe(scenes, name) {
    const match = (scenes || []).find(function (s) { return s.name === name; });
    return match ? { robot: match.robot, environment: match.environment || null } : null;
  }

  // every scene name in the index, alphabetically — sceneFor() only ever resolves a
  // (robot, environment) pair to its *first* match, so this is what lets the user
  // choose among several scenes that share one (e.g. multiple saved live recordings
  // of the same demo, which all share the same robot and environment model name)
  function names(scenes) {
    return (scenes || []).map(function (s) { return s.name; }).sort();
  }

  window.ScenePicker = {
    robots: robots,
    environments: environments,
    sceneFor: sceneFor,
    describe: describe,
    names: names,
  };
})();
