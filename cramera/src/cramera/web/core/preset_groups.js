/* ============================================================================
 * core/preset_groups.js — the ready-made questions, under the heading each belongs to.
 *
 * A demo knows more than one thing: what is true of the run in progress, and what its
 * finished runs recorded. Questions about them are written the same way but answered in
 * different places, so they are offered apart.
 *
 * A scene whose questions are all about one thing keeps a single unlabelled row —
 * a heading over the only group there is says nothing.
 * ==========================================================================*/
(function () {
  'use strict';

  const DEFAULT_SCOPE = 'current_state';

  function scopeOf(preset) {
    return preset.scope || DEFAULT_SCOPE;
  }

  // the demo names its scopes and their headings; a payload without them (a recorded
  // scene) still reads back a heading from the scope name itself
  function labelOf(name, described) {
    const known = (described || []).filter(function (scope) { return scope.name === name; })[0];
    if (known && known.label) return known.label;
    return name.split('_').map(function (word) {
      return word.charAt(0).toUpperCase() + word.slice(1);
    }).join(' ') + ' Queries';
  }

  function orderOf(presets, described) {
    const names = (described || []).map(function (scope) { return scope.name; });
    presets.forEach(function (preset) {
      if (names.indexOf(scopeOf(preset)) < 0) names.push(scopeOf(preset));
    });
    return names;
  }

  function of(presets, described) {
    const all = presets || [];
    if (!all.length) return [];
    const groups = orderOf(all, described).map(function (name) {
      return {
        name: name,
        label: labelOf(name, described),
        presets: all.filter(function (preset) { return scopeOf(preset) === name; }),
      };
    }).filter(function (group) { return group.presets.length; });
    if (groups.length === 1) groups[0].label = null;
    return groups;
  }

  window.PresetGroups = { of: of };
})();
