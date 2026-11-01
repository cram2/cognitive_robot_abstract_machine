// Draws the map that the parameter-mixin-map directive embeds: coraplex actions on the
// left, parameter mixins in the middle, giskardpy nodes on the right, and the same
// relation as a matrix below.
(() => {
  const root = document.querySelector(".parameter-mixin-map");
  if (!root) return;

  // %% indices
  const map = JSON.parse(root.querySelector(".parameter-mixin-map-data").textContent);
  const mixins = map.parameter_mixins;
  const mixinByName = Object.fromEntries(mixins.map(mixin => [mixin.name, mixin]));
  const classByName = Object.fromEntries(map.classes.map(takingClass => [takingClass.name, takingClass]));
  const usersOf = Object.fromEntries(mixins.map(mixin => [
    mixin.name, map.classes.filter(takingClass => takingClass.parameter_mixins.includes(mixin.name)),
  ]));
  const actions = map.classes.filter(takingClass => takingClass.framework === "coraplex");
  const nodes = map.classes.filter(takingClass => takingClass.framework === "giskardpy");
  const isShared = mixin => mixin.framework === "giskardpy" && usersOf[mixin.name].some(user => user.framework === "coraplex");
  const commonMixins = new Set(mixins
    .filter(mixin => usersOf[mixin.name].filter(user => user.framework === "giskardpy").length > nodes.length / 2)
    .map(mixin => mixin.name));
  const edgeKind = (takingClass, mixin) => mixin.framework === "coraplex"
    ? "coraplex" : (takingClass.framework === "giskardpy" ? "giskardpy" : "shared");

  // %% text
  const escape = text => String(text).replace(/[&<>"]/g, character => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[character]));
  const code = text => `<code>${escape(text)}</code>`;
  const documentation = text => escape(text)
    .replace(/:[a-z]+:`~?([^`]+)`/g, (_, target) => code(target.split(".").pop()))
    .replace(/``([^`]+)``/g, (_, literal) => `<code>${literal}</code>`)
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");

  // %% diagram
  const diagram = root.querySelector(".diagram");
  const details = root.querySelector(".details");
  const showCommon = root.querySelector(".show-common");
  root.querySelector(".show-common-label").innerHTML =
    `Show ${[...commonMixins].map(name => code(name)).join(", ")}, which most nodes take`;
  const svgNamespace = "http://www.w3.org/2000/svg";
  let selected = null;
  let hovered = null;

  function element(tag, attributes, parent) {
    const created = document.createElementNS(svgNamespace, tag);
    for (const key in attributes) created.setAttribute(key, attributes[key]);
    parent.appendChild(created);
    return created;
  }

  // Shortens text with an ellipsis until it fits into the given width, or leaves it empty.
  function fitText(textElement, text, width) {
    textElement.textContent = text;
    for (let length = text.length - 1; length > 0 && textElement.getComputedTextLength() > width; length--) {
      textElement.textContent = `${text.slice(0, length)}…`;
    }
    if (textElement.getComputedTextLength() > width) textElement.textContent = "";
  }

  // Orders a column by the mean height of the mixins its classes take, so that edges cross less.
  function orderByMixinHeight(column, heightOf) {
    for (const takingClass of column) {
      const heights = takingClass.shownMixins.map(heightOf);
      takingClass.meanHeight = heights.reduce((sum, height) => sum + height, 0) / heights.length;
    }
    return column.sort((first, second) => first.meanHeight - second.meanHeight || first.name.localeCompare(second.name));
  }

  function render() {
    const shownMixins = mixins.filter(mixin => showCommon.checked || !commonMixins.has(mixin.name));
    const shownNames = new Set(shownMixins.map(mixin => mixin.name));
    for (const takingClass of map.classes) {
      takingClass.shownMixins = takingClass.parameter_mixins.filter(name => shownNames.has(name));
    }

    const rowHeight = 17, top = 46, width = 1000;
    const actionX = 228, mixinLeft = 330, mixinRight = 640, nodeX = 742;
    const mixinY = {};
    const sections = [];
    let y = top;
    let previousSection = null;
    for (const mixin of shownMixins) {
      const section = `${mixin.framework} · ${mixin.section}`;
      if (section !== previousSection) {
        y += previousSection ? 14 : 0;
        sections.push({ y, label: section });
        y += 13;
        previousSection = section;
      }
      mixinY[mixin.name] = y;
      y += rowHeight + 3;
    }
    const left = orderByMixinHeight(actions.filter(action => action.shownMixins.length), name => mixinY[name]);
    const right = orderByMixinHeight(nodes.filter(node => node.shownMixins.length), name => mixinY[name]);
    const height = Math.max(y, top + 13 + left.length * rowHeight, top + 13 + right.length * rowHeight) + 16;
    for (const column of [left, right]) {
      const step = Math.max(rowHeight, (height - top - 30) / Math.max(column.length, 1));
      const start = top + 13 + Math.max(0, (height - top - 30 - step * column.length) / 2);
      column.forEach((takingClass, index) => { takingClass.y = start + index * step + step / 2 - rowHeight / 2 + 8; });
    }

    diagram.setAttribute("viewBox", `0 0 ${width} ${height}`);
    diagram.replaceChildren();
    element("text", { x: actionX, y: 22, "text-anchor": "end", class: "column-heading" }, diagram).textContent = "CORAPLEX ACTIONS";
    element("text", { x: (mixinLeft + mixinRight) / 2, y: 22, "text-anchor": "middle", class: "column-heading" }, diagram).textContent = "PARAMETER MIXINS";
    element("text", { x: nodeX, y: 22, class: "column-heading" }, diagram).textContent = "GISKARDPY NODES";

    const edges = element("g", {}, diagram);
    const curve = (x1, y1, x2, y2) => {
      const handle = (x2 - x1) * 0.5;
      return `M${x1},${y1} C${x1 + handle},${y1} ${x2 - handle},${y2} ${x2},${y2}`;
    };
    for (const action of left) for (const name of action.shownMixins) {
      element("path", {
        d: curve(actionX + 8, action.y - 4, mixinLeft, mixinY[name] + 4),
        class: `edge ${edgeKind(action, mixinByName[name])}`, "data-class": action.name, "data-mixin": name,
      }, edges);
    }
    for (const node of right) for (const name of node.shownMixins) {
      element("path", {
        d: curve(mixinRight, mixinY[name] + 4, nodeX - 8, node.y - 4),
        class: "edge giskardpy", "data-class": node.name, "data-mixin": name,
      }, edges);
    }
    for (const section of sections) {
      element("text", { x: mixinLeft + 4, y: section.y + 6, class: "section-heading" }, diagram).textContent = section.label;
    }

    const item = name => element("g", { class: "item", tabindex: 0, role: "button", "data-name": name, "aria-label": name }, diagram);
    for (const mixin of shownMixins) {
      const group = item(mixin.name);
      element("rect", {
        x: mixinLeft, y: mixinY[mixin.name] - 5, width: mixinRight - mixinLeft, height: rowHeight, rx: 8,
        class: `pill ${mixin.framework}${isShared(mixin) ? " shared" : ""}`,
      }, group);
      const mixinName = element("text", { x: mixinLeft + 10, y: mixinY[mixin.name] + 7 }, group);
      mixinName.textContent = mixin.name;
      const parameterNames = element("text", { x: mixinRight - 8, y: mixinY[mixin.name] + 7, "text-anchor": "end", class: "parameter-names" }, group);
      fitText(parameterNames, mixin.parameters.map(parameter => parameter.name).join(", "),
        mixinRight - 8 - (mixinLeft + 10 + mixinName.getComputedTextLength() + 12));
    }
    for (const action of left) {
      const group = item(action.name);
      element("circle", { cx: actionX + 8, cy: action.y - 4, r: 2.5, class: "dot coraplex" }, group);
      element("text", { x: actionX, y: action.y, "text-anchor": "end" }, group).textContent = action.name;
    }
    for (const node of right) {
      const group = item(node.name);
      element("circle", { cx: nodeX - 8, cy: node.y - 4, r: 2.5, class: "dot giskardpy" }, group);
      element("text", { x: nodeX, y: node.y }, group).textContent = node.name;
    }
    applyFocus();
  }

  function related(name) {
    if (mixinByName[name]) return new Set([name, ...usersOf[name].map(user => user.name)]);
    return new Set([name, ...classByName[name].parameter_mixins]);
  }

  function applyFocus() {
    const name = hovered || selected;
    diagram.classList.toggle("focus", !!name);
    const relatedNames = name ? related(name) : new Set();
    diagram.querySelectorAll("g.item").forEach(group => {
      group.classList.toggle("on", relatedNames.has(group.dataset.name));
      group.classList.toggle("selected", group.dataset.name === name);
    });
    diagram.querySelectorAll(".edge").forEach(edge => edge.classList.toggle(
      "on", !!name && (edge.dataset.class === name || edge.dataset.mixin === name)));
    showDetails(name);
  }

  // %% details
  function parameterItem(parameter, framework, origin) {
    const signature = `<b>${escape(parameter.name)}</b>: ${escape(parameter.type_hint)}${parameter.default === null ? "" : ` = ${escape(parameter.default)}`}`;
    return `<li class="${framework}"><div class="signature">${signature}</div>`
      + (origin ? `<div class="origin">${origin}</div>` : "")
      + (parameter.documentation ? `<div class="parameter-documentation">${documentation(parameter.documentation)}</div>` : "")
      + "</li>";
  }

  function chip(name) {
    const framework = (mixinByName[name] || classByName[name]).framework;
    return `<button type="button" class="chip ${framework}" data-pick="${escape(name)}">${escape(name)}</button>`;
  }

  function showDetails(name) {
    if (!name) {
      details.innerHTML = `<div class="eyebrow">Details</div>
        <p class="hint">Hover over or tap an action, mixin or node to see the parameters it declares or takes. Tap it again or press Escape to clear.</p>
        <p class="hint">Purple outlines mark the giskardpy mixins that coraplex actions take too.</p>`;
      return;
    }
    const mixin = mixinByName[name];
    if (mixin) {
      details.innerHTML = `<div class="eyebrow ${mixin.framework}">${mixin.framework} mixin · ${escape(mixin.section)}</div>
        <h4>${escape(mixin.name)}</h4><p class="hint">${documentation(mixin.documentation)}</p>
        <ul class="parameters">${mixin.parameters.map(parameter => parameterItem(parameter, mixin.framework, "")).join("")}</ul>
        <div class="kind">Taken by ${usersOf[name].length}</div>
        <div class="chips">${usersOf[name].map(user => chip(user.name)).join("")}</div>`;
      return;
    }
    const takingClass = classByName[name];
    const redeclared = Object.fromEntries(takingClass.redeclared_parameters.map(declaration => [declaration.parameter.name, declaration]));
    const items = takingClass.parameter_mixins.flatMap(mixinName => mixinByName[mixinName].parameters.map(parameter => {
      const declaration = redeclared[parameter.name];
      return declaration
        ? parameterItem(declaration.parameter, mixinByName[mixinName].framework, `from ${escape(mixinName)}, declared again by ${escape(declaration.declaring_class)}`)
        : parameterItem(parameter, mixinByName[mixinName].framework, `from ${escape(mixinName)}`);
    }));
    const kind = takingClass.framework === "giskardpy" ? "giskardpy node" : "coraplex action";
    details.innerHTML = `<div class="eyebrow ${takingClass.framework}">${kind} · ${escape(takingClass.defining_module)}</div>
      <h4>${escape(takingClass.name)}</h4>
      ${takingClass.behaviours.length ? `<div class="kind">Behaviours: ${takingClass.behaviours.map(code).join(", ")}</div>` : ""}
      <div class="kind">${takingClass.parameter_mixins.length} parameter mixins, ${items.length} parameters</div>
      <ul class="parameters">${items.join("")}</ul>
      <div class="chips">${takingClass.parameter_mixins.map(chip).join("")}</div>`;
  }

  function select(name, toggles) {
    selected = toggles && selected === name ? null : name;
    hovered = null;
    applyFocus();
  }

  diagram.addEventListener("pointerover", event => {
    const group = event.target.closest("g.item");
    if (group && event.pointerType === "mouse") { hovered = group.dataset.name; applyFocus(); }
  });
  diagram.addEventListener("pointerout", event => {
    const group = event.target.closest("g.item");
    if (group && event.pointerType === "mouse") { hovered = null; applyFocus(); }
  });
  diagram.addEventListener("click", event => {
    const group = event.target.closest("g.item");
    select(group ? group.dataset.name : null, true);
  });
  diagram.addEventListener("keydown", event => {
    const group = event.target.closest("g.item");
    if (group && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); select(group.dataset.name, true); }
  });
  document.addEventListener("keydown", event => { if (event.key === "Escape") select(null, false); });
  details.addEventListener("click", event => {
    const picked = event.target.closest("[data-pick]");
    if (picked) select(picked.dataset.pick, false);
  });
  showCommon.addEventListener("change", render);

  // %% matrix
  const matrix = root.querySelector(".matrix");
  const tabs = [...root.querySelectorAll(".tabs button")];
  const shortName = mixin => mixin.name.replace(/^Has/, "").replace(/Parameters?$/, "");

  function renderMatrix(framework) {
    const rows = map.classes.filter(takingClass => takingClass.framework === framework);
    const columns = mixins.filter(mixin => usersOf[mixin.name].some(user => user.framework === framework));
    const groups = {};
    for (const row of rows) (groups[row.defining_module] ??= []).push(row);
    const head = `<thead><tr><th></th>${columns.map(mixin => `<th class="column-name ${mixin.framework}" title="${escape(mixin.parameters.map(parameter => parameter.name).join(", "))}"><span>${escape(shortName(mixin))}</span></th>`).join("")}</tr></thead>`;
    const body = Object.keys(groups).sort().map(group =>
      `<tr class="group"><th colspan="${columns.length + 1}">${escape(group.replace(/_/g, " "))}</th></tr>`
      + groups[group].map(row => `<tr><th>${escape(row.name)}<small>${row.parameter_mixins.length}</small></th>${columns.map(mixin => row.parameter_mixins.includes(mixin.name)
        ? `<td title="${escape(mixin.name)}"><i class="${edgeKind(row, mixin)}"></i></td>` : "<td></td>").join("")}</tr>`).join(""),
    ).join("");
    matrix.innerHTML = `${head}<tbody>${body}</tbody>`;
    tabs.forEach(tab => tab.setAttribute("aria-selected", tab.dataset.rows === framework));
  }
  tabs.forEach(tab => tab.addEventListener("click", () => renderMatrix(tab.dataset.rows)));

  render();
  renderMatrix("coraplex");
})();
