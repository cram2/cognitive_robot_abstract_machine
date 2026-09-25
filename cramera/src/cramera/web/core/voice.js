/* ============================================================================
 * core/voice.js — capturing one spoken question as text.
 *
 * Wraps the browser's SpeechRecognition (Chrome ships it prefixed as
 * webkitSpeechRecognition). A capture turns one press of the record button into
 * one transcript and hands it to its callbacks; it knows nothing about what the
 * text is for — the panel publishes it on the bus, and any consumer takes it
 * from there.
 *
 * The recognizer constructor is injectable, so the capture's state machine is
 * unit-testable without a browser.
 * ==========================================================================*/
(function () {
  'use strict';

  function browserRecognizer() {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
  }

  // options:
  //   onTranscript(text)     one recognized utterance, as text
  //   onState(listening)     the microphone opened or closed
  //   onError(message)       recognition failed (no mic, no permission, no speech)
  //   recognizer             SpeechRecognition-shaped constructor; the browser's
  //                          own when absent, null for an unsupported browser
  function create(options) {
    const Recognizer =
      options.recognizer === undefined ? browserRecognizer() : options.recognizer;
    let active = null;   // the in-flight recognition, while listening

    const capture = {
      supported: !!Recognizer,
      listening: false,

      // opens the microphone for one utterance; false when unsupported or
      // already listening
      start: function () {
        if (!capture.supported || capture.listening) return false;
        const recognition = new Recognizer();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        recognition.onresult = function (event) {
          const transcript = event.results[0][0].transcript;
          if (options.onTranscript) options.onTranscript(transcript);
        };
        recognition.onerror = function (event) {
          if (options.onError) {
            options.onError((event && event.error) || 'speech recognition failed');
          }
        };
        // fires after result and error alike: the one place listening ends
        recognition.onend = function () {
          capture.listening = false;
          active = null;
          if (options.onState) options.onState(false);
        };
        active = recognition;
        capture.listening = true;
        if (options.onState) options.onState(true);
        recognition.start();
        return true;
      },

      // asks the in-flight recognition to wrap up; a capture that is not
      // listening has nothing to stop
      stop: function () {
        if (active) active.stop();
      },
    };
    return capture;
  }

  window.VoiceCapture = { create: create };
})();
