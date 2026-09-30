/* ============================================================================
 * core/replay.js — replaying a recorded slice of the live demo.
 *
 * The pure logic of the replay popup: reading the ?replay= window out of a URL,
 * building the popup URL that carries the window (and an explicit bridge address)
 * into a fresh viewer, and mapping wall-clock playback time onto recorded frames.
 * The popup itself is the ordinary viewer page mounted in replay mode; nothing
 * here touches the DOM.
 * ==========================================================================*/
(function () {
  'use strict';

  // how long the playback rests on the last frame before looping, in seconds
  const LOOP_HOLD_SECONDS = 1.0;

  // ?replay=<start>,<end> (epoch seconds) -> {start, end}, or null when absent
  // or unusable. An unusable window is treated as "not a replay page" rather
  // than an error: the viewer then simply behaves as the ordinary live page.
  function fromSearch(search) {
    const match = /[?&]replay=([^&]+)/.exec(search || '');
    if (!match) return null;
    const parts = decodeURIComponent(match[1]).split(',');
    if (parts.length !== 2) return null;
    const start = parseFloat(parts[0]);
    const end = parseFloat(parts[1]);
    if (!isFinite(start) || !isFinite(end) || end <= start) return null;
    return { start: start, end: end };
  }

  // the URL a popup replays `window_` at. What the opener was looking at travels with
  // it: the scene whose recording holds the moment, and an explicit live= bridge
  // address so the popup asks the same bridge.
  function popupUrl(pathname, search, window_) {
    const carried = ['scene=[\\w-]+', 'live=[\\w.:-]+']
      .map(function (pattern) { return new RegExp('[?&](' + pattern + ')').exec(search || ''); })
      .filter(Boolean)
      .map(function (match) { return '&' + match[1]; })
      .join('');
    return pathname + '?replay=' + window_.start + ',' + window_.end + carried;
  }

  // how long the recorded clip runs, in seconds
  function duration(frames) {
    if (!frames || !frames.length) return 0;
    return frames[frames.length - 1].at - frames[0].at;
  }

  // the frame on screen after `elapsed` seconds of looping playback: the newest
  // frame not later than the playback time, holding the last frame for
  // LOOP_HOLD_SECONDS before starting over
  function frameAt(frames, elapsed) {
    if (!frames || !frames.length) return null;
    const at = frames[0].at + (elapsed % (duration(frames) + LOOP_HOLD_SECONDS));
    let shown = frames[0];
    for (let index = 0; index < frames.length; index++) {
      if (frames[index].at > at) break;
      shown = frames[index];
    }
    return shown;
  }

  // '12:00:25 – 12:00:35' — what the popup's badge names the clip
  function label(window_) {
    function clock(at) {
      const date = new Date(at * 1000);
      function two(value) { return (value < 10 ? '0' : '') + value; }
      return two(date.getHours()) + ':' + two(date.getMinutes()) + ':' + two(date.getSeconds());
    }
    return clock(window_.start) + ' – ' + clock(window_.end);
  }

  window.Replay = {
    fromSearch: fromSearch,
    popupUrl: popupUrl,
    duration: duration,
    frameAt: frameAt,
    label: label,
  };
})();
