'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const ScenePanelFunctions = require('./scene_panel_functions');

// %% refresh-independent playback
class PlaybackLoop extends ScenePanelFunctions {
  constructor(refreshRate, speed) {
    super({
      running: true, models: [], replayClip: null, highlightArrows: {},
      clock: {getDelta() { return 1 / refreshRate; }},
      controls: {update() { return false; }},
      playing: true, traj: {frames: Array(1000).fill({}), framesPerSecond: 30},
      liveOn: false, playhead: 0, playbackSpeedMultiplier: speed, playheadCbs: [],
      follow: false, needsRender: false, requestAnimationFrame() {},
    });
    this.scope.applyFrame = () => {};
    this.scope.renderFrame = () => {};
  }

  advanceFrames(count) {
    for (let index = 0; index < count; index++) this.scope.tick();
    return this.scope.playhead;
  }
}

for (const refreshRate of [30, 60, 144]) {
  for (const speed of [0.5, 1, 2]) {
    test(`one second at ${refreshRate} Hz advances playback at ${speed}×`, () => {
      const playback = new PlaybackLoop(refreshRate, speed);
      const expected = playback.scope.traj.framesPerSecond * speed;
      assert.ok(Math.abs(playback.advanceFrames(refreshRate) - expected) < 1e-9);
    });
  }
}
