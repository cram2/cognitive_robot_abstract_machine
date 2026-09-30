/* ============================================================================
 * core/completion.js — what the query box offers for the word being typed.
 *
 * Three questions, none of which touch the DOM: which word is under the caret
 * (and whether it follows a dot, so members are wanted rather than names),
 * which of the vocabulary's entries match it, and what the box reads after one
 * is accepted. The vocabulary itself comes from whichever source answers the
 * queries (core/query_source.js), shaped as the /vocabulary payload's entries:
 * {name, kind, detail, module, type}.
 * ==========================================================================*/
(function () {
  'use strict';

  const NAME_CHARACTER = /[A-Za-z0-9_]/;

  // the order names are offered in: the ones a query is built out of first, the
  // workspace's several thousand classes last. Members of a type follow the same
  // rule among themselves — its own data before what it can do.
  const KIND_ORDER = [
    'variable', 'field', 'property', 'method',
    'factory', 'entity_type', 'value', 'class',
  ];

  const PREFIX_MATCH = 0, INITIALS_MATCH = 1, CONTAINS_MATCH = 2, NO_MATCH = 3;

  function kindRank(kind) {
    const rank = KIND_ORDER.indexOf(kind);
    return rank < 0 ? KIND_ORDER.length : rank;
  }

  // the word being typed at the caret: what precedes it decides whether the box
  // is naming something (`Bo`) or reaching into it (`scene_object.na`)
  function tokenAt(text, caret) {
    const source = String(text == null ? '' : text);
    const end = Math.max(0, Math.min(caret, source.length));
    let start = end;
    while (start > 0 && NAME_CHARACTER.test(source[start - 1])) start--;
    return {
      prefix: source.slice(start, end),
      start: start,
      end: end,
      owner: ownerBefore(source, start),
    };
  }

  // the name a dot immediately before `start` reaches into, or '' when the word
  // being typed is not a member of anything. A dotted chain answers with its
  // nearest name, which is the one whose members are being typed.
  function ownerBefore(source, start) {
    if (start === 0 || source[start - 1] !== '.') return '';
    let end = start - 1;
    let ownerStart = end;
    while (ownerStart > 0 && NAME_CHARACTER.test(source[ownerStart - 1])) ownerStart--;
    return source.slice(ownerStart, end);
  }

  // how well one name answers to what has been typed, lower being closer
  function matchOf(name, prefix) {
    if (!prefix) return PREFIX_MATCH;
    const lowerName = name.toLowerCase(), lowerPrefix = prefix.toLowerCase();
    if (lowerName.indexOf(lowerPrefix) === 0) return PREFIX_MATCH;
    if (initialsOf(name).indexOf(prefix.toUpperCase()) === 0) return INITIALS_MATCH;
    if (lowerName.indexOf(lowerPrefix) > 0) return CONTAINS_MATCH;
    return NO_MATCH;
  }

  // the capitals of a name, so `BCC` reaches BodyCollisionCheck
  function initialsOf(name) {
    return name.replace(/[^A-Z]/g, '');
  }

  // the entries worth offering for one token, closest match first
  function suggest(entries, token, limit) {
    const prefix = (token && token.prefix) || '';
    const ranked = [];
    (entries || []).forEach(function (entry, position) {
      const match = matchOf(entry.name, prefix);
      if (match !== NO_MATCH) ranked.push({ entry: entry, match: match, position: position });
    });
    ranked.sort(function (left, right) {
      return left.match - right.match ||
        kindRank(left.entry.kind) - kindRank(right.entry.kind) ||
        left.entry.name.length - right.entry.name.length ||
        left.position - right.position;
    });
    return ranked.slice(0, limit || ranked.length).map(function (scored) { return scored.entry; });
  }

  // the box's text and caret once an entry is accepted for the typed token
  function applied(text, token, entry) {
    const source = String(text == null ? '' : text);
    const inserted = entry.name;
    return {
      text: source.slice(0, token.start) + inserted + source.slice(token.end),
      caret: token.start + inserted.length,
    };
  }

  window.Completion = {
    tokenAt: tokenAt,
    suggest: suggest,
    applied: applied,
  };
})();
