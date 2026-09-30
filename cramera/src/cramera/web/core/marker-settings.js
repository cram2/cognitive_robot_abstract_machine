/* ============================================================================
 * core/marker-settings.js — the viewer's RViz-like marker settings state.
 *
 * Two persisted choices: which marker namespaces are hidden (a client-side filter,
 * like collapsing a namespace checkbox in RViz), and which topics the user watched
 * or dropped beyond the demo's defaults (re-applied to the bridge on every attach,
 * since each demo process starts from its own default subscriptions).
 *
 * Pure state rules, no DOM, so they are testable under node.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const HIDDEN_NAMESPACES_KEY = 'cramera.hidden-marker-namespaces';
  /* localStorage key of the hidden namespace list, as a JSON array. */

  const TOPIC_OVERRIDES_KEY = 'cramera.marker-topic-overrides';
  /* localStorage key of the user's topic choices, as JSON {topic: bool}. */

  function readList(storage, key) {
    try {
      const stored = JSON.parse(storage.getItem(key) || 'null');
      return Array.isArray(stored) ? stored : [];
    } catch (error) {
      return [];
    }
  }

  function readMap(storage, key) {
    try {
      const stored = JSON.parse(storage.getItem(key) || 'null');
      return stored && typeof stored === 'object' && !Array.isArray(stored) ? stored : {};
    } catch (error) {
      return {};
    }
  }

  /* The hidden namespaces as a lookup set. */
  function hiddenNamespaces(storage) {
    const hidden = {};
    readList(storage, HIDDEN_NAMESPACES_KEY).forEach(function (ns) { hidden[ns] = true; });
    return hidden;
  }

  /* Hide or show one namespace; returns the updated lookup set. */
  function setNamespaceHidden(storage, ns, hidden) {
    const set = hiddenNamespaces(storage);
    if (hidden) set[ns] = true;
    else delete set[ns];
    storage.setItem(HIDDEN_NAMESPACES_KEY, JSON.stringify(Object.keys(set).sort()));
    return set;
  }

  /* Only the markers whose namespace is not hidden. */
  function visibleMarkers(markers, hidden) {
    return (markers || []).filter(function (marker) { return !hidden[marker.ns]; });
  }

  /* Every namespace a payload carries, sorted and unique. */
  function namespacesOf(markers) {
    const seen = {};
    (markers || []).forEach(function (marker) { seen[marker.ns] = true; });
    return Object.keys(seen).sort();
  }

  /* The user's topic choices beyond the demo's defaults: {topic: subscribed}. */
  function topicOverrides(storage) {
    return readMap(storage, TOPIC_OVERRIDES_KEY);
  }

  /* Remember one topic choice; a choice matching the demo's own state still sticks,
     because the next demo process starts from its defaults again. */
  function setTopicOverride(storage, topic, subscribed) {
    const overrides = topicOverrides(storage);
    overrides[topic] = subscribed;
    storage.setItem(TOPIC_OVERRIDES_KEY, JSON.stringify(overrides));
    return overrides;
  }

  global.MarkerSettings = {
    HIDDEN_NAMESPACES_KEY: HIDDEN_NAMESPACES_KEY,
    TOPIC_OVERRIDES_KEY: TOPIC_OVERRIDES_KEY,
    hiddenNamespaces: hiddenNamespaces,
    setNamespaceHidden: setNamespaceHidden,
    visibleMarkers: visibleMarkers,
    namespacesOf: namespacesOf,
    topicOverrides: topicOverrides,
    setTopicOverride: setTopicOverride,
  };
})(typeof window !== 'undefined' ? window : this);
