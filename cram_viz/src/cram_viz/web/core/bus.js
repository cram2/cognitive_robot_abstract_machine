/* ============================================================================
 * core/bus.js — the event bus panels talk over.
 *
 * Panels never call each other directly: they publish and subscribe here, so
 * any panel can be removed (or a new one added) without breaking the rest —
 * an event nobody listens to simply goes nowhere.
 *
 * The event contract between the built-in panels:
 *
 *   event                payload                        emitted by → consumed by
 *   -------------------- ------------------------------ ------------------------
 *   scene:part-clicked   {id}                           robot-scene → eql
 *   scene:step           {step}   ('__done__' at end)   robot-scene → eql, graph
 *   live:changed         {on, url}                      robot-scene → graph
 *   entity:highlight     {ids, focus?}                  eql → robot-scene, graph
 *   entity:select        {id, detail, relations}        graph → eql
 *   kb:ready             {payload}                      eql → anyone
 *
 * A new panel is free to define further events; document them in its header.
 * ==========================================================================*/
(function () {
  'use strict';

  const handlers = {};   // event name -> [callback]

  window.Bus = {
    on: function (event, cb) {
      (handlers[event] = handlers[event] || []).push(cb);
      return cb;
    },
    off: function (event, cb) {
      const list = handlers[event];
      if (!list) return;
      const i = list.indexOf(cb);
      if (i >= 0) list.splice(i, 1);
    },
    emit: function (event, payload) {
      (handlers[event] || []).slice().forEach(function (cb) {
        try {
          cb(payload);
        } catch (err) {
          // one broken listener must not take the other panels down
          console.error('[bus] listener for "' + event + '" failed:', err);
        }
      });
    },
  };
})();
