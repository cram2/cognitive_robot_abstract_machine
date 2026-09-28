/* ============================================================================
 * core/answer_table.js — an EQL answer as one table.
 *
 * Answer rows are objects whose keys depend on what the query asked for: a set_of()
 * names its own columns, an entity query answers with a named thing and its fields.
 * Both read far better as a table with stable columns than as a run of chips, so this
 * settles the columns once and classifies every value by what it is, leaving only the
 * markup and the colours to the panel.
 *
 * A __key__ is the answer's own bookkeeping rather than something asked for, so any of
 * them is kept out of the columns whether or not this viewer knows what it means: the
 * server it is talking to may be newer than the page.
 * ==========================================================================*/
(function () {
  'use strict';

  const ENTITY_NAME = '__entity__';
  const ENTITY_TYPE = '__type__';
  const BOOKKEEPING_KEY = /^__.+__$/;
  const NAME_COLUMN = 'name';
  const EMPTY_CELL = '—';

  // the columns of an entity answer are its name followed by its own fields; of any
  // other answer, every key any row carries, in the order the rows introduce them
  function columnsOf(rows) {
    const columns = [];
    rows.forEach(function (row) {
      if (row[ENTITY_NAME] !== undefined && columns.indexOf(NAME_COLUMN) < 0) {
        columns.push(NAME_COLUMN);
      }
      Object.keys(row).forEach(function (key) {
        if (BOOKKEEPING_KEY.test(key)) return;
        if (columns.indexOf(key) < 0) columns.push(key);
      });
    });
    return columns;
  }

  function cellOf(row, column) {
    const named = row[ENTITY_NAME];
    if (column === NAME_COLUMN && named !== undefined) {
      return { text: String(named), kind: 'name' };
    }
    return valueCell(row[column]);
  }

  function valueCell(value) {
    if (value === null || value === undefined || value === '') {
      return { text: EMPTY_CELL, kind: 'empty' };
    }
    if (typeof value === 'boolean') return { text: String(value), kind: String(value) };
    if (typeof value === 'number') return { text: String(value), kind: 'number' };
    if (typeof value === 'object') return { text: JSON.stringify(value), kind: 'text' };
    return { text: String(value), kind: 'text' };
  }

  // `replay` is the answer's parallel list of replay windows, one entry per row and
  // null for a row naming no moment; an answer that carries none replays nothing.
  function of(rows, replay) {
    const all = rows || [];
    const windows = replay || [];
    const columns = columnsOf(all);
    return {
      columns: columns,
      rows: all.map(function (row, index) {
        return {
          type: row[ENTITY_TYPE] === undefined ? null : String(row[ENTITY_TYPE]),
          replay: windows[index] === undefined ? null : windows[index],
          cells: columns.map(function (column) { return cellOf(row, column); }),
        };
      }),
    };
  }

  window.AnswerTable = { of: of };
})();
