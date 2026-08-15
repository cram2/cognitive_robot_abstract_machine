// Unit tests for core/timeline-events.js (node:test): which moments of a recording the
// replay timeline marks, and how each one reads.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function loadTimelineEvents() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/timeline-events.js'), 'utf8'))(scope);
  return scope.TimelineEvents;
}

// a scene bundle's segments, in the shape cramera.onboard.demo derives them
function transportScene() {
  return {
    segments: [
      { step: 'prepare', action: 'ParkArmsAction', start: 0, end: 120 },
      { step: 'transport_milk', action: 'TransportAction', arm: 'left',
        start: 120, end: 640, picks: 'milk.stl', attach: 210, detach: 590 },
    ],
  };
}

const framesOf = function (events) {
  return events.map(function (event) { return event.frame; });
};

const kindsOf = function (events) {
  return events.map(function (event) { return event.kind; });
};

// %% which moments are marked
test('a scene with no segments has nothing to mark', function () {
  const TimelineEvents = loadTimelineEvents();

  assert.deepStrictEqual(TimelineEvents.of({ segments: [] }), []);
});

test('a scene that was never onboarded with segments is handled', function () {
  const TimelineEvents = loadTimelineEvents();

  assert.deepStrictEqual(TimelineEvents.of(null), []);
});

test('every step start is marked', function () {
  const TimelineEvents = loadTimelineEvents();

  const events = TimelineEvents.of({
    segments: [{ step: 'prepare', start: 0, end: 120 },
               { step: 'transport_milk', start: 120, end: 640 }],
  });

  assert.deepStrictEqual(framesOf(events), [0, 120]);
  assert.deepStrictEqual(kindsOf(events), [
    TimelineEvents.EventKind.STEP, TimelineEvents.EventKind.STEP]);
});

test('a manipulation segment also marks its pick and its release', function () {
  const TimelineEvents = loadTimelineEvents();

  const events = TimelineEvents.of(transportScene());

  assert.deepStrictEqual(framesOf(events), [0, 120, 210, 590]);
  assert.deepStrictEqual(kindsOf(events), [
    TimelineEvents.EventKind.STEP,
    TimelineEvents.EventKind.STEP,
    TimelineEvents.EventKind.PICK,
    TimelineEvents.EventKind.RELEASE,
  ]);
});

test('a segment that moves nothing marks only its start', function () {
  const TimelineEvents = loadTimelineEvents();

  const events = TimelineEvents.of({
    segments: [{ step: 'navigate', start: 40, end: 90, attach: 50, detach: 80 }],
  });

  assert.deepStrictEqual(kindsOf(events), [TimelineEvents.EventKind.STEP]);
});

test('a pick on the very frame a step starts is marked as the pick', function () {
  const TimelineEvents = loadTimelineEvents();

  const events = TimelineEvents.of({
    segments: [{ step: 'transport_milk', start: 210, end: 640,
                 picks: 'milk.stl', attach: 210, detach: 590 }],
  });

  assert.deepStrictEqual(framesOf(events), [210, 590]);
  assert.strictEqual(events[0].kind, TimelineEvents.EventKind.PICK);
});

test('the marks come out in the order they happen', function () {
  const TimelineEvents = loadTimelineEvents();

  const events = TimelineEvents.of({
    segments: [{ step: 'second', start: 300, end: 400 },
               { step: 'first', start: 10, end: 300 }],
  });

  assert.deepStrictEqual(framesOf(events), [10, 300]);
});

// %% how a mark reads
test('a step reads as the step starting', function () {
  const TimelineEvents = loadTimelineEvents();
  const events = TimelineEvents.of(transportScene());

  assert.strictEqual(TimelineEvents.describe(events[1]), 'transport milk starts');
});

test('a pick and a release name the object without its mesh file extension', function () {
  const TimelineEvents = loadTimelineEvents();
  const events = TimelineEvents.of(transportScene());

  assert.strictEqual(TimelineEvents.describe(events[2]), 'picked up milk');
  assert.strictEqual(TimelineEvents.describe(events[3]), 'let go of milk');
});

test('each kind of mark is drawn in its own colour', function () {
  const TimelineEvents = loadTimelineEvents();
  const events = TimelineEvents.of(transportScene());
  const colors = events.map(TimelineEvents.colorOf);

  assert.strictEqual(new Set(colors).size, 3);          // step, pick, release
  assert.strictEqual(colors[0], colors[1]);             // both steps alike
});

// %% where a mark sits, and when it happens
test('a mark sits where its frame falls in the run', function () {
  const TimelineEvents = loadTimelineEvents();

  assert.strictEqual(TimelineEvents.positionOf(0, 101), 0);
  assert.strictEqual(TimelineEvents.positionOf(50, 101), 50);
  assert.strictEqual(TimelineEvents.positionOf(100, 101), 100);
});

test('a frame beyond the run stays on the timeline', function () {
  const TimelineEvents = loadTimelineEvents();

  assert.strictEqual(TimelineEvents.positionOf(500, 101), 100);
  assert.strictEqual(TimelineEvents.positionOf(-5, 101), 0);
});

test('a single-frame recording puts its mark at the start', function () {
  const TimelineEvents = loadTimelineEvents();

  assert.strictEqual(TimelineEvents.positionOf(0, 1), 0);
});

// %% where the preview card sits
test('a preview centres on its mark when there is room', function () {
  const TimelineEvents = loadTimelineEvents();

  assert.strictEqual(TimelineEvents.previewOffset(50, 400, 180), 200);
});

test('a preview near either end stops short of the edge', function () {
  const TimelineEvents = loadTimelineEvents();

  assert.strictEqual(TimelineEvents.previewOffset(0, 400, 180), 90);
  assert.strictEqual(TimelineEvents.previewOffset(100, 400, 180), 310);
});

test('a preview wider than the track sits in the middle of it', function () {
  const TimelineEvents = loadTimelineEvents();

  assert.strictEqual(TimelineEvents.previewOffset(0, 120, 180), 60);
});

test('a mark reports the run time of its frame', function () {
  const TimelineEvents = loadTimelineEvents();

  assert.strictEqual(TimelineEvents.timeOf(185, 30), '0:06');
  assert.strictEqual(TimelineEvents.timeOf(1800, 30), '1:00');
});
