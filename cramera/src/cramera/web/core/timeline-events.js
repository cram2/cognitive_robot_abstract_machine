/* ============================================================================
 * core/timeline-events.js — the key moments of a recording, as the replay
 * timeline marks them.
 *
 * A bundle records its plan steps as segments (see cramera.onboard.demo's
 * derive_segments), and a manipulation segment additionally knows the frame its
 * object was picked up and the frame it was let go. Those are the moments worth
 * jumping to, so the scrubber marks them and previews what the scene looks like
 * there.
 *
 * Pure derivation, no DOM and no three.js, so it is testable under node.
 * ==========================================================================*/
(function (global) {
  'use strict';

  /* What kind of moment a mark on the timeline is. */
  const EventKind = {
    STEP: 'step',
    PICK: 'pick',
    RELEASE: 'release',
  };

  /* How each kind reads on the timeline. */
  const KIND_STYLE = {
    step: { color: '#5b8cff', verb: 'starts' },
    pick: { color: '#ffb648', verb: 'picked up' },
    release: { color: '#4bd38a', verb: 'let go of' },
  };

  /* A step name as recorded ('transport_milk') reads as a phrase. */
  function readableStep(step) {
    return String(step || 'step').replace(/_/g, ' ');
  }

  /* The object a manipulation segment carries, without its mesh extension. */
  function objectOf(segment) {
    return String(segment.picks || '').replace(/\.[^.]+$/, '');
  }

  function eventsOfSegment(segment) {
    const events = [{
      frame: segment.start || 0,
      kind: EventKind.STEP,
      label: readableStep(segment.step),
      step: segment.step,
    }];
    const object = objectOf(segment);
    if (!object) return events;
    if (typeof segment.attach === 'number') {
      events.push({ frame: segment.attach, kind: EventKind.PICK, label: object,
                    step: segment.step });
    }
    if (typeof segment.detach === 'number') {
      events.push({ frame: segment.detach, kind: EventKind.RELEASE, label: object,
                    step: segment.step });
    }
    return events;
  }

  /* Every key moment of one scene bundle, in the order they happen.

     Two moments on the same frame would sit on top of each other on the timeline,
     so the more specific one wins: a step that starts exactly where its object is
     picked up is marked as the pick. */
  function of(scene) {
    const segments = (scene && scene.segments) || [];
    const byFrame = {};
    segments.forEach(function (segment) {
      eventsOfSegment(segment).forEach(function (event) {
        const taken = byFrame[event.frame];
        if (taken && taken.kind !== EventKind.STEP) return;
        byFrame[event.frame] = event;
      });
    });
    return Object.keys(byFrame)
      .map(function (frame) { return byFrame[frame]; })
      .sort(function (one, other) { return one.frame - other.frame; });
  }

  /* Where a frame sits along the timeline, as a percentage of the whole run. */
  function positionOf(frame, frameCount) {
    if (!frameCount || frameCount < 2) return 0;
    const clamped = Math.min(Math.max(frame, 0), frameCount - 1);
    return (clamped / (frameCount - 1)) * 100;
  }

  /* What one event reads as: 'picked up milk', 'transport milk starts'. */
  function describe(event) {
    const style = KIND_STYLE[event.kind] || KIND_STYLE.step;
    return event.kind === EventKind.STEP
      ? event.label + ' ' + style.verb
      : style.verb + ' ' + event.label;
  }

  /* The colour a mark is drawn in. */
  function colorOf(event) {
    return (KIND_STYLE[event.kind] || KIND_STYLE.step).color;
  }

  /* Where the preview card sits, in pixels along the track.

     The card is centred on its mark, so a mark near either end would push it past the
     panel edge; it stops short instead, and only the card moves — the mark stays put. */
  function previewOffset(percent, trackWidth, cardWidth) {
    const centre = (percent / 100) * trackWidth;
    const half = cardWidth / 2;
    if (cardWidth >= trackWidth) return trackWidth / 2;
    return Math.min(Math.max(centre, half), trackWidth - half);
  }

  /* The run's timestamp of a frame, as m:ss. */
  function timeOf(frame, framesPerSecond) {
    const totalSeconds = Math.max(0, frame) / (framesPerSecond || 30);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60);
    return minutes + ':' + (seconds < 10 ? '0' : '') + seconds;
  }

  global.TimelineEvents = {
    EventKind: EventKind,
    of: of,
    positionOf: positionOf,
    previewOffset: previewOffset,
    describe: describe,
    colorOf: colorOf,
    timeOf: timeOf,
  };
})(typeof window !== 'undefined' ? window : this);
