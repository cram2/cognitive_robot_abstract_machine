/* ============================================================================
 * core/model-constraints.js — how the Models tab's constraint rows become the
 * workbench API's payload.
 *
 * A row constrains one variable: a numeric one to a closed interval, a symbolic one
 * to a selection of its elements. Several rows on the same variable are united
 * server-side, mirroring the desktop GUI's semantics. Pure data mapping, no DOM, so
 * the payload rules are testable under node.
 * ==========================================================================*/
(function (global) {
  'use strict';

  /* One row's state -> its API constraint, or null when the row is incomplete.
     Row state: {variable, kind, low, high, values} where low/high are the numeric
     inputs' strings and values is the symbolic selection. */
  function constraintOf(row) {
    if (!row || !row.variable) return null;
    if (row.kind === 'symbolic') {
      const values = (row.values || []).filter(function (value) { return value !== ''; });
      if (!values.length) return null;
      return { variable: row.variable, values: values };
    }
    const low = parseFloat(row.low);
    const high = parseFloat(row.high);
    if (!isFinite(low) || !isFinite(high)) return null;
    return { variable: row.variable, intervals: [[Math.min(low, high), Math.max(low, high)]] };
  }

  /* Every complete row as the API's constraint list; incomplete rows are skipped so
     an empty or half-filled row means "unconstrained" rather than an error. */
  function payload(rows) {
    return (rows || []).map(constraintOf).filter(function (constraint) { return constraint !== null; });
  }

  /* A row's constraint as the short text shown in results and summaries. */
  function describe(constraint) {
    if (!constraint) return '';
    if (constraint.values) return constraint.variable + ' ∈ {' + constraint.values.join(', ') + '}';
    const interval = constraint.intervals[0];
    return constraint.variable + ' ∈ [' + interval[0] + ', ' + interval[1] + ']';
  }

  global.ModelConstraints = { constraintOf: constraintOf, payload: payload, describe: describe };
})(typeof window !== 'undefined' ? window : this);
