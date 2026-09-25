/* ============================================================================
 * core/environment-theme.js — link-name -> material look for environment models.
 *
 * Bundled environment URDFs (apartments, kitchens, warehouses, ...) carry whatever
 * grey the authoring tool exported. This maps a link's name to a plausible furniture
 * color/finish using the regex vocabulary those models are named with, so the same
 * lookup applies to any onboarded environment rather than one hand-tuned scene.
 *
 * Pure string -> descriptor logic, no THREE/DOM dependency, so it is usable both from
 * the browser panel and from node:test.
 * ==========================================================================*/
(function () {
  'use strict';

  //: a run of identically-named siblings (e.g. shelved books) cycles through this
  //: palette by their trailing index, so the row doesn't render as one solid block
  var VARIED_PALETTE = [
    0xd7263d, 0x1b998b, 0xf4a261, 0x3a86ff, 0xffbe0b, 0x8338ec, 0x2ec4b6, 0xe63946,
  ];

  //: [pattern, look] tried in order; first match wins. ``texture`` names a procedural
  //: map the caller supplies (see panel.js's WOOD_COUNTER/WOOD_TABLE); null means a
  //: flat color.
  var RULES = [
    [/cooktop|hotplate|ceran|stove/, { color: 0x0a0b0d, roughness: 0.18, metalness: 0.15, texture: null }],
    [/island_countertop|countertop|worktop/, { color: 0xffffff, roughness: 0.55, metalness: 0.02, texture: 'counter' }],
    [/coffee_table|table_area|bedside_table|table_top|dining|nightstand/, { color: 0xffffff, roughness: 0.5, metalness: 0.02, texture: 'table' }],
    [/bookshelf/, { color: 0x6b4226, roughness: 0.65, metalness: 0.0, texture: null }],
    [/sofa|couch|armchair/, { color: 0xa85c48, roughness: 0.88, metalness: 0.0, texture: null }],
    [/chair_/, { color: 0x8a5a3b, roughness: 0.6, metalness: 0.0, texture: null }],
    [/bed_/, { color: 0xe8dcc8, roughness: 0.82, metalness: 0.0, texture: null }],
    [/plant|flower/, { color: 0x3f7d43, roughness: 0.7, metalness: 0.0, texture: null }],
    [/handle|tap_body|tap_handle|sink|faucet/, { color: 0xc6ccd4, roughness: 0.28, metalness: 0.85, texture: null }],
    [/cabinet|drawer|door|wardrobe|dishwasher|oven|coffe_machine|island_back|island_waterfall|side_[ab]|fridge/, { color: 0x1b1d21, roughness: 0.42, metalness: 0.12, texture: null }],
    [/toilet|basin|paper_holder/, { color: 0xf2f2f0, roughness: 0.3, metalness: 0.0, texture: null }],
    [/radiator|handrail|pipe/, { color: 0xb8bcc2, roughness: 0.35, metalness: 0.6, texture: null }],
    [/trash_can/, { color: 0x2a2d31, roughness: 0.5, metalness: 0.1, texture: null }],
    [/floor_lamp/, { color: 0xd4c9a8, roughness: 0.5, metalness: 0.3, texture: null }],
    [/wall/, { color: 0xd9d4cb, roughness: 0.95, metalness: 0.0, texture: null }],
  ];

  function trailingIndex(name) {
    var match = /(\d+)$/.exec(name);
    return match ? parseInt(match[1], 10) : 0;
  }

  //: a link's furniture look, or null when nothing in the vocabulary matches
  function lookOf(linkName) {
    var name = (linkName || '').toLowerCase();
    if (/^book_\d+$/.test(name)) {
      return {
        color: VARIED_PALETTE[trailingIndex(name) % VARIED_PALETTE.length],
        roughness: 0.75,
        metalness: 0.0,
        texture: null,
      };
    }
    for (var i = 0; i < RULES.length; i++) {
      if (RULES[i][0].test(name)) return RULES[i][1];
    }
    return null;
  }

  window.EnvironmentTheme = { lookOf: lookOf, VARIED_PALETTE: VARIED_PALETTE };
})();
