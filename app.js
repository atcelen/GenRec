/* GenRec project page — vanilla JS, no build step.
 * Reads window.sectionA / window.sectionB from data.js, renders rows,
 * wires up the slider crossfade and paired video scrub. */

(function () {
  "use strict";

  const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Build a <picture> that prefers a sibling .webp over the original .jpg/.png.
   * If the source is not a local image (e.g. a placehold.co URL) we just emit
   * a plain <img> — there's no point asking for a webp variant of a remote PNG. */
  function pictureFor(src, alt) {
    const img = new Image();
    img.src = src;
    img.alt = alt || "";
    img.loading = "lazy";
    img.decoding = "async";
    return img;
  }

  function el(tag, opts = {}, children = []) {
    const node = document.createElement(tag);
    if (opts.className) node.className = opts.className;
    if (opts.text != null) node.textContent = opts.text;
    if (opts.attrs) {
      for (const [k, v] of Object.entries(opts.attrs)) {
        if (v === false || v == null) continue;
        if (v === true) node.setAttribute(k, "");
        else node.setAttribute(k, String(v));
      }
    }
    if (opts.style) {
      for (const [k, v] of Object.entries(opts.style)) {
        node.style.setProperty(k, v);
      }
    }
    for (const c of [].concat(children)) {
      if (c == null || c === false) continue;
      node.appendChild(c.nodeType ? c : document.createTextNode(c));
    }
    return node;
  }

  /* ----------------------------- Carousel ------------------------------- */

  function setupCarousel(carouselId, indicatorId) {
    const container = document.getElementById(carouselId);
    if (!container) return;
    const rows = container.querySelectorAll(".carousel-viewport .row");
    if (rows.length === 0) return;
    const indicator = document.getElementById(indicatorId);
    const prevBtn = container.querySelector(".carousel-prev");
    const nextBtn = container.querySelector(".carousel-next");
    let current = 0;

    function show(index) {
      // Pause videos in the current slide before switching
      rows[current].querySelectorAll("video").forEach(function (v) { v.pause(); });
      current = Math.max(0, Math.min(rows.length - 1, index));
      rows.forEach(function (r, i) { r.classList.toggle("active", i === current); });
      if (indicator) {
        indicator.textContent = (current + 1) + " / " + rows.length;
      }
      if (prevBtn) prevBtn.disabled = current === 0;
      if (nextBtn) nextBtn.disabled = current === rows.length - 1;
    }

    if (prevBtn) prevBtn.addEventListener("click", function () { show(current - 1); });
    if (nextBtn) nextBtn.addEventListener("click", function () { show(current + 1); });
    show(0);
  }

  /* ----------------------------- Section A ----------------------------- */

  function renderSectionA() {
    const cfg = window.sectionA;
    if (!cfg) return;
    const baselineHeader = document.getElementById("section-a-baseline-header");
    if (baselineHeader && cfg.baselineLabel) {
      baselineHeader.textContent = cfg.baselineLabel;
    }

    const re10kHost = document.getElementById("section-a-re10k-rows");
    const dl3dvHost = document.getElementById("section-a-dl3dv-rows");
    if (!re10kHost || !dl3dvHost) return;
    re10kHost.replaceChildren();
    dl3dvHost.replaceChildren();

    cfg.rows.forEach((row) => {
      const node = buildSectionARow(row, cfg.baselineLabel || "Baseline");
      if ((row.sceneName || "").toLowerCase().includes("dl3dv")) {
        dl3dvHost.appendChild(node);
      } else {
        re10kHost.appendChild(node);
      }
    });

    setupCarousel("carousel-a-re10k", "carousel-a-re10k-indicator");
    setupCarousel("carousel-a-dl3dv", "carousel-a-dl3dv-indicator");
  }

  function buildSectionARow(row, baselineLabel) {
    const gtSamples = Array.isArray(row.groundTruth) ? row.groundTruth : [];
    const ourSamples = Array.isArray(row.ourSamples) ? row.ourSamples : [];
    const blSamples = Array.isArray(row.baselineSamples) ? row.baselineSamples : [];
    const N = Math.max(gtSamples.length, ourSamples.length, blSamples.length);

    const inputCell = el("div", { className: "cell" }, [
      el("span", { className: "cell-label", text: "Input" }),
      el("div", { className: "media" }, [
        pictureFor(row.input, `${row.altPrefix} input`),
      ]),
    ]);

    const gtResult = buildSampleStack(gtSamples, row.altPrefix, "ground truth");
    const oursResult = buildSampleStack(ourSamples, row.altPrefix, "our output");
    const blResult = buildSampleStack(blSamples, row.altPrefix, "baseline output");

    const gtCell = el("div", { className: "cell" }, [
      el("span", { className: "cell-label", text: "Ground Truth" }),
      gtResult.container,
    ]);
    const oursCell = el("div", { className: "cell" }, [
      el("span", { className: "cell-label", text: "GenRec (ours)" }),
      oursResult.container,
    ]);
    const blCell = el("div", { className: "cell" }, [
      el("span", { className: "cell-label", text: baselineLabel }),
      blResult.container,
    ]);

    const children = [inputCell, gtCell, oursCell, blCell];
    if (N > 1) {
      children.push(buildSharedControls(N, row, [gtResult, oursResult, blResult]));
    }

    return el("article", { className: "row" }, [
      el("div", { className: "row-grid row-grid-4col" }, children),
    ]);
  }

  function buildSampleStack(samples, altPrefix, label) {
    if (samples.length === 0) {
      var container = el("div", { className: "media" }, [
        el("div", {
          attrs: { "aria-label": "No " + label + " available" },
          style: {
            display: "grid",
            "place-items": "center",
            width: "100%",
            height: "100%",
            color: "var(--fg-muted)",
            "font-size": "0.85rem",
          },
          text: "No sample",
        }),
      ]);
      return { container: container, elements: [] };
    }

    if (samples.length === 1) {
      var container = el("div", { className: "media" }, [
        pictureFor(samples[0], altPrefix + " " + label),
      ]);
      return { container: container, elements: [] };
    }

    var stack = el("div", { className: "sample-stack media" });
    var elements = samples.map(function (src, i) {
      var node = pictureFor(src, altPrefix + " " + label + " sample " + (i + 1));
      if (i === 0) node.classList.add("active");
      stack.appendChild(node);
      return node;
    });
    return { container: stack, elements: elements };
  }

  function buildSharedControls(N, row, stacks) {
    const counter = el("span", { text: `Target View 1 / ${N}` });
    const hint = el("span", { className: "muted", text: "use \u2190 \u2192 keys" });
    const counterRow = el("div", { className: "sample-counter" }, [counter, hint]);

    const dots = el("div", { className: "dots", attrs: { role: "presentation" } });
    const dotEls = [];
    for (let i = 0; i < N; i++) {
      const d = el("span", { className: i === 0 ? "dot active" : "dot" });
      dots.appendChild(d);
      dotEls.push(d);
    }

    const range = el("input", {
      className: "range",
      attrs: {
        type: "range",
        min: 0,
        max: N - 1,
        step: 1,
        value: 0,
        "aria-label": `Sample selector \u2014 ${row.sceneName || row.altPrefix}`,
      },
    });
    range.style.setProperty("--fill", "0%");

    function setIndex(i) {
      i = Math.max(0, Math.min(N - 1, i));
      stacks.forEach(function (s) {
        s.elements.forEach(function (node, k) {
          node.classList.toggle("active", k === i);
        });
      });
      dotEls.forEach(function (d, k) { d.classList.toggle("active", k === i); });
      counter.textContent = `Target View ${i + 1} / ${N}`;
      const pct = N === 1 ? 0 : (i / (N - 1)) * 100;
      range.style.setProperty("--fill", `${pct}%`);
    }

    range.addEventListener("input", function () { setIndex(parseInt(range.value, 10)); });

    dotEls.forEach(function (d, k) {
      d.addEventListener("click", function () {
        range.value = String(k);
        setIndex(k);
      });
      d.style.cursor = "pointer";
    });

    return el("div", { className: "sample-controls" }, [
      el("div", { style: { display: "flex", "flex-direction": "column", gap: "8px" } }, [counterRow, dots]),
      range,
    ]);
  }

  /* ----------------------------- Section D (Ablation) -------------------- */

  function renderSectionD() {
    const cfg = window.sectionD;
    if (!cfg || !cfg.rows || cfg.rows.length === 0) return;
    const host = document.getElementById("section-d-rows");
    if (!host) return;
    host.replaceChildren();

    cfg.rows.forEach(function (row) {
      host.appendChild(buildAblationRow(row, cfg));
    });

    setupCarousel("carousel-d", "carousel-d-indicator");
  }

  /* Reusable before/after comparison slider. opts:
   *   topSrc / bottomSrc   — images (top is revealed on the left side)
   *   topLabel / bottomLabel — overlay chips (either may be empty to omit)
   *   altPrefix, className — alt text prefix, extra class on the frame */
  function makeCompareSlider(opts) {
    var frame = el("div", {
      className: "ablation-zoom" + (opts.className ? " " + opts.className : ""),
    });

    var imgBottom = new Image();
    imgBottom.src = opts.bottomSrc;
    imgBottom.alt = opts.altPrefix + " " + (opts.bottomLabel || "after");
    imgBottom.className = "ablation-img-bottom";
    imgBottom.loading = "lazy";

    var imgTop = new Image();
    imgTop.src = opts.topSrc;
    imgTop.alt = opts.altPrefix + " " + (opts.topLabel || "before");
    imgTop.className = "ablation-img-top";
    imgTop.loading = "lazy";

    var sliderLine = el("div", { className: "ablation-slider-line" });
    var sliderHandle = el("div", {
      className: "ablation-slider-handle",
      attrs: { tabindex: "0", role: "slider", "aria-label": "Comparison slider", "aria-valuenow": "50", "aria-valuemin": "0", "aria-valuemax": "100" }
    });
    sliderHandle.innerHTML = '<svg viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><polyline points="6 2 3 5 6 8"/></svg><svg viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><polyline points="4 2 7 5 4 8"/></svg>';

    frame.appendChild(imgBottom);
    frame.appendChild(imgTop);
    frame.appendChild(sliderLine);
    frame.appendChild(sliderHandle);
    if (opts.topLabel) {
      frame.appendChild(el("span", { className: "ablation-label ablation-label-left", text: opts.topLabel }));
    }
    if (opts.bottomLabel) {
      frame.appendChild(el("span", { className: "ablation-label ablation-label-right", text: opts.bottomLabel }));
    }

    var pos = 50;
    var dragging = false;

    function setPos(p, animate) {
      pos = Math.max(0, Math.min(100, p));
      if (animate) {
        imgTop.style.transition = "clip-path 180ms cubic-bezier(0.25,0.1,0.25,1)";
        sliderLine.style.transition = "left 180ms cubic-bezier(0.25,0.1,0.25,1)";
        sliderHandle.style.transition = "left 180ms cubic-bezier(0.25,0.1,0.25,1)";
        setTimeout(function () {
          imgTop.style.transition = "";
          sliderLine.style.transition = "";
          sliderHandle.style.transition = "";
        }, 200);
      }
      imgTop.style.clipPath = "inset(0 " + (100 - pos) + "% 0 0)";
      sliderLine.style.left = pos + "%";
      sliderHandle.style.left = pos + "%";
      sliderHandle.setAttribute("aria-valuenow", String(Math.round(pos)));
    }

    function getPosFromEvent(e) {
      var rect = frame.getBoundingClientRect();
      return ((e.clientX - rect.left) / rect.width) * 100;
    }

    frame.addEventListener("pointerdown", function (e) {
      if (e.target === sliderHandle || sliderHandle.contains(e.target)) {
        dragging = true;
      } else {
        setPos(getPosFromEvent(e), true);
        dragging = true;
      }
      frame.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    frame.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      setPos(getPosFromEvent(e), false);
    });
    frame.addEventListener("pointerup", function (e) {
      dragging = false;
      frame.releasePointerCapture(e.pointerId);
    });

    sliderHandle.addEventListener("keydown", function (e) {
      var step = e.shiftKey ? 10 : 2;
      if (e.key === "ArrowLeft") { setPos(pos - step, false); e.preventDefault(); }
      else if (e.key === "ArrowRight") { setPos(pos + step, false); e.preventDefault(); }
      else if (e.key === "Home") { setPos(0, true); e.preventDefault(); }
      else if (e.key === "End") { setPos(100, true); e.preventDefault(); }
    });

    setPos(50, false);
    return frame;
  }

  function buildAblationRow(row, cfg) {
    var sourceImg = new Image();
    sourceImg.src = row.sourceImage;
    sourceImg.alt = row.altPrefix + " source view";
    sourceImg.loading = "lazy";
    sourceImg.decoding = "async";

    var cells = [el("div", { className: "ablation-source" }, [sourceImg])];

    var hasFull = Boolean(row.fullWith && row.fullWithout);
    if (hasFull) {
      cells.push(makeCompareSlider({
        topSrc: row.fullWith,
        bottomSrc: row.fullWithout,
        topLabel: cfg.withLabel,
        bottomLabel: cfg.withoutLabel,
        altPrefix: row.altPrefix,
        className: "ablation-zoom-full",
      }));
    }

    cells.push(makeCompareSlider({
      topSrc: row.cropWith,
      bottomSrc: row.cropWithout,
      topLabel: hasFull ? "w/" : cfg.withLabel,
      bottomLabel: hasFull ? "w/o" : cfg.withoutLabel,
      altPrefix: row.altPrefix + " zoom",
      className: hasFull ? "ablation-zoom-crop" : "",
    }));

    var grid = el("div", {
      className: "ablation-row-grid" + (hasFull ? " ablation-row-grid-3" : ""),
    }, cells);

    var caption = row.caption ? el("p", { className: "ablation-caption", text: row.caption }) : null;

    var article = el("article", { className: "row" }, [grid]);
    if (caption) article.appendChild(caption);
    return article;
  }

  /* ----------------------------- Section B ----------------------------- */

  function renderSectionB() {
    const cfg = window.sectionB;
    if (!cfg) return;
    const withHeader = document.getElementById("section-b-with-header");
    const withoutHeader = document.getElementById("section-b-without-header");
    if (withHeader && cfg.withLabel) withHeader.textContent = cfg.withLabel;
    if (withoutHeader && cfg.withoutLabel) withoutHeader.textContent = cfg.withoutLabel;

    const host = document.getElementById("section-b-rows");
    if (!host) return;
    host.replaceChildren();

    cfg.rows.forEach((row) => {
      host.appendChild(buildSectionBRow(row, cfg));
    });

    setupCarousel("carousel-b", "carousel-b-indicator");
  }

  function buildInputBlock(inputs, altPrefix) {
    const arr = Array.isArray(inputs) ? inputs : [inputs];
    if (arr.length <= 1) {
      return el("div", { className: "input-grid-1" }, [
        el("div", { className: "media" }, [
          pictureFor(arr[0], `${altPrefix} input`),
        ]),
      ]);
    }
    if (arr.length === 2) {
      return el(
        "div",
        { className: "input-grid-2" },
        arr.map((src, i) =>
          el("div", { className: "media" }, [
            pictureFor(src, `${altPrefix} input ${i + 1}`),
          ])
        )
      );
    }
    return el(
      "div",
      { className: "input-grid-quad" },
      arr.slice(0, 4).map((src, i) =>
        el("div", { className: "media" }, [
          pictureFor(src, `${altPrefix} input ${i + 1}`),
        ])
      )
    );
  }

  function buildVideoCell(src, poster, label, altPrefix) {
    const v = document.createElement("video");
    v.src = src;
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    v.preload = "metadata";
    v.setAttribute("aria-label", `${altPrefix} ${label}`);
    if (poster) v.poster = poster;
    return el("div", { className: "cell" }, [
      el("span", { className: "cell-label", text: label }),
      el("div", { className: "media" }, [v]),
    ]);
  }

  function buildSectionBRow(row, cfg) {
    const inputCell = el("div", { className: "cell" }, [
      el("span", { className: "cell-label", text: "Input" }),
      buildInputBlock(row.inputs, row.altPrefix),
    ]);

    const withCell = buildVideoCell(
      row.withScm,
      row.poster,
      cfg.withLabel,
      row.altPrefix
    );
    const withoutCell = buildVideoCell(
      row.withoutScm,
      row.poster,
      cfg.withoutLabel,
      row.altPrefix
    );

    const videos = [
      withCell.querySelector("video"),
      withoutCell.querySelector("video"),
    ];

    const playBtn = el("button", {
      className: "ghost-btn",
      attrs: { type: "button", "aria-pressed": "false" },
      text: "Play",
    });
    const restartBtn = el("button", {
      className: "ghost-btn",
      attrs: { type: "button" },
      text: "Restart",
    });
    const scrub = el("input", {
      className: "range",
      attrs: {
        type: "range",
        min: 0,
        max: 1000,
        step: 1,
        value: 0,
        "aria-label": `Scrub ${row.sceneName || row.altPrefix} videos`,
      },
    });
    scrub.style.setProperty("--fill", "0%");

    let dragging = false;
    let primary = videos[0];

    function reflectPlayState() {
      const isPlaying = !primary.paused && !primary.ended;
      playBtn.textContent = isPlaying ? "Pause" : "Play";
      playBtn.setAttribute("aria-pressed", isPlaying ? "true" : "false");
    }

    function syncFollowers() {
      // Keep secondaries in lock-step with the primary's currentTime/play state.
      const drift = 0.08;
      videos.forEach((v) => {
        if (v === primary) return;
        if (Math.abs(v.currentTime - primary.currentTime) > drift) {
          try { v.currentTime = primary.currentTime; } catch (_) {}
        }
        if (primary.paused && !v.paused) v.pause();
        if (!primary.paused && v.paused) v.play().catch(() => {});
      });
    }

    primary.addEventListener("timeupdate", () => {
      if (dragging) return;
      const d = primary.duration;
      if (!isFinite(d) || d <= 0) return;
      const pct = (primary.currentTime / d) * 100;
      scrub.value = String(Math.round((primary.currentTime / d) * 1000));
      scrub.style.setProperty("--fill", `${pct}%`);
      syncFollowers();
    });
    primary.addEventListener("play", () => { reflectPlayState(); syncFollowers(); refreshSectionBPlayAll(); });
    primary.addEventListener("pause", () => { reflectPlayState(); syncFollowers(); refreshSectionBPlayAll(); });

    scrub.addEventListener("pointerdown", () => { dragging = true; });
    scrub.addEventListener("pointerup", () => { dragging = false; });
    scrub.addEventListener("input", () => {
      const d = primary.duration;
      if (!isFinite(d) || d <= 0) return;
      const t = (parseInt(scrub.value, 10) / 1000) * d;
      videos.forEach((v) => { try { v.currentTime = t; } catch (_) {} });
      scrub.style.setProperty("--fill", `${(t / d) * 100}%`);
    });

    playBtn.addEventListener("click", () => {
      if (primary.paused) primary.play().catch(() => {});
      else primary.pause();
    });
    restartBtn.addEventListener("click", () => {
      videos.forEach((v) => { try { v.currentTime = 0; } catch (_) {} });
      primary.play().catch(() => {});
    });

    /* Tag for IntersectionObserver + section-level Play All. */
    videos.forEach((v) => v.dataset.sectionB = "1");

    const controls = el("div", { className: "video-controls" }, [
      playBtn,
      restartBtn,
      scrub,
    ]);

    return el("article", { className: "row" }, [
      el("div", { className: "row-grid" }, [inputCell, withCell, withoutCell, controls]),
    ]);
  }

  /* ----------------------------- Section C ----------------------------- */

  function renderSectionC() {
    const cfg = window.sectionC;
    if (!cfg || !cfg.rows || cfg.rows.length === 0) return;
    const host = document.getElementById("section-c-rows");
    if (!host) return;
    host.replaceChildren();

    cfg.rows.forEach((row) => {
      host.appendChild(buildSectionCRow(row));
    });

    setupCarousel("carousel-c", "carousel-c-indicator");
  }

  function buildSectionCRow(row) {
    const inputCell = el("div", { className: "cell" }, [
      el("span", { className: "cell-label", text: "Input" }),
      el("div", { className: "media" }, [
        pictureFor(row.input, `${row.altPrefix} input`),
      ]),
    ]);

    const v = document.createElement("video");
    v.src = row.video;
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    v.preload = "metadata";
    v.setAttribute("aria-label", `${row.altPrefix} 3DGS render`);
    v.dataset.sectionC = "1";

    const videoCell = el("div", { className: "cell" }, [
      el("span", { className: "cell-label", text: "3DGS Render" }),
      el("div", { className: "media" }, [v]),
    ]);

    const playBtn = el("button", {
      className: "ghost-btn",
      attrs: { type: "button", "aria-pressed": "false" },
      text: "Play",
    });
    const restartBtn = el("button", {
      className: "ghost-btn",
      attrs: { type: "button" },
      text: "Restart",
    });
    const scrub = el("input", {
      className: "range",
      attrs: {
        type: "range",
        min: 0,
        max: 1000,
        step: 1,
        value: 0,
        "aria-label": `Scrub ${row.sceneName || row.altPrefix} video`,
      },
    });
    scrub.style.setProperty("--fill", "0%");

    let dragging = false;

    function reflectPlayState() {
      const isPlaying = !v.paused && !v.ended;
      playBtn.textContent = isPlaying ? "Pause" : "Play";
      playBtn.setAttribute("aria-pressed", isPlaying ? "true" : "false");
    }

    v.addEventListener("timeupdate", () => {
      if (dragging) return;
      const d = v.duration;
      if (!isFinite(d) || d <= 0) return;
      const pct = (v.currentTime / d) * 100;
      scrub.value = String(Math.round((v.currentTime / d) * 1000));
      scrub.style.setProperty("--fill", `${pct}%`);
    });
    v.addEventListener("play", () => { reflectPlayState(); refreshSectionCPlayAll(); });
    v.addEventListener("pause", () => { reflectPlayState(); refreshSectionCPlayAll(); });

    scrub.addEventListener("pointerdown", () => { dragging = true; });
    scrub.addEventListener("pointerup", () => { dragging = false; });
    scrub.addEventListener("input", () => {
      const d = v.duration;
      if (!isFinite(d) || d <= 0) return;
      const t = (parseInt(scrub.value, 10) / 1000) * d;
      v.currentTime = t;
      scrub.style.setProperty("--fill", `${(t / d) * 100}%`);
    });

    playBtn.addEventListener("click", () => {
      if (v.paused) v.play().catch(() => {});
      else v.pause();
    });
    restartBtn.addEventListener("click", () => {
      v.currentTime = 0;
      v.play().catch(() => {});
    });

    const controls = el("div", { className: "video-controls" }, [
      playBtn,
      restartBtn,
      scrub,
    ]);

    return el("article", { className: "row" }, [
      el("div", { className: "row-grid section-c-grid" }, [inputCell, videoCell, controls]),
    ]);
  }

  /* ------------------- IntersectionObserver autoplay ------------------- */

  function setupAutoplayObserver() {
    if (REDUCED_MOTION) return;
    if (!("IntersectionObserver" in window)) return;

    const rows = document.querySelectorAll("#section-b-rows .row, #section-c-rows .row");
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const vids = entry.target.querySelectorAll("video");
          if (entry.isIntersecting) {
            vids.forEach((v) => v.play().catch(() => {
              /* autoplay blocked — user must press Play. */
            }));
          } else {
            vids.forEach((v) => v.pause());
          }
        });
      },
      { threshold: 0.25 }
    );
    rows.forEach((r) => io.observe(r));
  }

  /* ----------------------- Section-level Play All ---------------------- */

  let _playAllRefreshScheduled = false;
  function refreshSectionBPlayAll() {
    if (_playAllRefreshScheduled) return;
    _playAllRefreshScheduled = true;
    requestAnimationFrame(() => {
      _playAllRefreshScheduled = false;
      const btn = document.getElementById("play-all");
      if (!btn) return;
      const vids = document.querySelectorAll('video[data-section-b="1"]');
      const anyPlaying = Array.from(vids).some((v) => !v.paused && !v.ended);
      btn.textContent = anyPlaying ? "Pause all" : "Play all";
      btn.setAttribute("aria-pressed", anyPlaying ? "true" : "false");
    });
  }

  function setupSectionBPlayAll() {
    const btn = document.getElementById("play-all");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const vids = Array.from(document.querySelectorAll('video[data-section-b="1"]'));
      const anyPlaying = vids.some((v) => !v.paused && !v.ended);
      if (anyPlaying) vids.forEach((v) => v.pause());
      else vids.forEach((v) => v.play().catch(() => {}));
    });
    refreshSectionBPlayAll();
  }

  /* ------------------- Section C Play All ------------------- */

  let _playAllCRefreshScheduled = false;
  function refreshSectionCPlayAll() {
    if (_playAllCRefreshScheduled) return;
    _playAllCRefreshScheduled = true;
    requestAnimationFrame(() => {
      _playAllCRefreshScheduled = false;
      const btn = document.getElementById("play-all-c");
      if (!btn) return;
      const vids = document.querySelectorAll('video[data-section-c="1"]');
      const anyPlaying = Array.from(vids).some((v) => !v.paused && !v.ended);
      btn.textContent = anyPlaying ? "Pause all" : "Play all";
      btn.setAttribute("aria-pressed", anyPlaying ? "true" : "false");
    });
  }

  function setupSectionCPlayAll() {
    const btn = document.getElementById("play-all-c");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const vids = Array.from(document.querySelectorAll('video[data-section-c="1"]'));
      const anyPlaying = vids.some((v) => !v.paused && !v.ended);
      if (anyPlaying) vids.forEach((v) => v.pause());
      else vids.forEach((v) => v.play().catch(() => {}));
    });
    refreshSectionCPlayAll();
  }

  /* ------------------------- Quantitative results ----------------------- */

  function numericValue(raw) {
    if (raw == null) return NaN;
    return parseFloat(String(raw).replace(/[^0-9.\-]/g, ""));
  }

  function buildResultsTable(bench) {
    var cols = bench.columns;

    // Contiguous column groups → colspan header cells.
    var groups = [];
    cols.forEach(function (c) {
      var g = c.group || "";
      var last = groups[groups.length - 1];
      if (last && last.name === g) last.span += 1;
      else groups.push({ name: g, span: 1 });
    });

    var headRow1 = el("tr", {}, [
      el("th", { className: "col-method", attrs: { rowspan: 2, scope: "col" }, text: "Method" }),
    ].concat(groups.map(function (g) {
      return el("th", {
        className: "col-group",
        attrs: { colspan: g.span, scope: g.span > 1 ? "colgroup" : "col" },
        text: g.name,
      });
    })));

    var headRow2 = el("tr", {}, cols.map(function (c) {
      return el("th", { className: "col-metric", attrs: { scope: "col" } }, [
        c.label,
        c.dir ? el("span", { className: "metric-dir", text: c.dir === "up" ? " ↑" : " ↓" }) : null,
      ]);
    }));

    // Best value per ranked column.
    var best = cols.map(function (c, ci) {
      if (!c.dir) return null;
      var bestVal = null;
      bench.rows.forEach(function (r) {
        var v = numericValue(r.values[ci]);
        if (!isFinite(v)) return;
        if (bestVal == null || (c.dir === "up" ? v > bestVal : v < bestVal)) bestVal = v;
      });
      return bestVal;
    });

    var body = el("tbody", {}, bench.rows.map(function (r) {
      return el("tr", { className: r.ours ? "row-ours" : "" }, [
        el("th", { className: "col-method", attrs: { scope: "row" } }, [
          r.method,
          r.ours ? el("span", { className: "ours-chip", text: "ours" }) : null,
        ]),
      ].concat(cols.map(function (c, ci) {
        var raw = r.values[ci];
        var v = numericValue(raw);
        var isBest = best[ci] != null && isFinite(v) && v === best[ci];
        return el("td", { className: isBest ? "best" : "", text: raw == null ? "–" : String(raw) });
      })));
    }));

    var table = el("table", { className: "results-table" }, [
      el("thead", {}, [headRow1, headRow2]),
      body,
    ]);

    var block = el("div", { className: "results-table-block" }, [
      el("div", { className: "table-scroll" }, [table]),
    ]);
    if (bench.note) block.appendChild(el("p", { className: "results-note", text: bench.note }));
    return block;
  }

  function renderResults() {
    var cfg = window.resultsData;
    if (!cfg || !Array.isArray(cfg.benchmarks) || cfg.benchmarks.length === 0) return;
    var tabBar = document.getElementById("results-tabs");
    var host = document.getElementById("results-table-host");
    if (!tabBar || !host) return;
    tabBar.replaceChildren();

    var tabs = cfg.benchmarks.map(function (bench, i) {
      var btn = el("button", {
        className: "results-tab",
        attrs: {
          type: "button",
          role: "tab",
          "aria-selected": i === 0 ? "true" : "false",
          tabindex: i === 0 ? "0" : "-1",
        },
        text: bench.label,
      });
      if (bench.badge) {
        btn.appendChild(el("span", { className: "results-tab-badge", text: bench.badge }));
      }
      btn.addEventListener("click", function () { select(i); });
      tabBar.appendChild(btn);
      return btn;
    });

    tabBar.addEventListener("keydown", function (e) {
      var current = tabs.findIndex(function (t) { return t.getAttribute("aria-selected") === "true"; });
      var next = -1;
      if (e.key === "ArrowRight") next = (current + 1) % tabs.length;
      else if (e.key === "ArrowLeft") next = (current - 1 + tabs.length) % tabs.length;
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = tabs.length - 1;
      if (next >= 0) {
        select(next);
        tabs[next].focus();
        e.preventDefault();
      }
    });

    function select(i) {
      tabs.forEach(function (t, k) {
        t.setAttribute("aria-selected", k === i ? "true" : "false");
        t.setAttribute("tabindex", k === i ? "0" : "-1");
      });
      host.replaceChildren(buildResultsTable(cfg.benchmarks[i]));
    }

    select(0);
  }

  function renderComponentAblation() {
    var cfg = window.componentAblation;
    if (!cfg || !Array.isArray(cfg.datasets)) return;
    var host = document.getElementById("component-ablation-host");
    if (!host) return;
    host.replaceChildren();

    var grid = el("div", { className: "ablation-tables" }, cfg.datasets.map(function (ds) {
      var card = el("div", { className: "ablation-table-card" }, [
        el("h3", { className: "carousel-group-title", text: ds.label }),
        buildResultsTable({ columns: cfg.columns, rows: ds.rows, note: ds.delta }),
      ]);
      return card;
    }));
    host.appendChild(grid);
  }

  /* ------------------------ Sticky nav + scroll spy ---------------------- */

  function setupSiteNav() {
    var nav = document.getElementById("site-nav");
    if (!nav) return;

    var hero = document.querySelector(".hero");
    if (hero && "IntersectionObserver" in window) {
      var visIo = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          nav.classList.toggle("visible", !entry.isIntersecting);
        });
      }, { rootMargin: "-60px 0px 0px 0px" });
      visIo.observe(hero);
    } else {
      nav.classList.add("visible");
    }

    var links = Array.from(nav.querySelectorAll("a[href^='#']")).filter(function (a) {
      return a.getAttribute("href").length > 1;
    });
    var sections = links
      .map(function (a) { return document.getElementById(a.getAttribute("href").slice(1)); })
      .filter(Boolean);
    if (!("IntersectionObserver" in window) || sections.length === 0) return;

    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        links.forEach(function (a) {
          a.classList.toggle("active", a.getAttribute("href") === "#" + entry.target.id);
        });
      });
    }, { rootMargin: "-30% 0px -60% 0px" });
    sections.forEach(function (s) { spy.observe(s); });
  }

  /* ----------------------------- Scroll reveal --------------------------- */

  function setupReveal() {
    if (REDUCED_MOTION || !("IntersectionObserver" in window)) return;
    var targets = document.querySelectorAll(
      ".section-header, .figure-plate, .figure-caption, .legend-chips, " +
      ".results-tabs, #results-table-host, .ablation-table-card"
    );
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        io.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -5% 0px" });
    targets.forEach(function (t) {
      t.classList.add("reveal");
      io.observe(t);
    });
  }

  /* --------------------------------- Boot ------------------------------- */

  function boot() {
    renderSectionA();
    renderSectionD();
    renderSectionC();
    renderResults();
    renderComponentAblation();
    setupAutoplayObserver();
    setupSectionCPlayAll();
    setupBibtexCopy();
    setupSiteNav();
    setupReveal();
  }

  function setupBibtexCopy() {
    document.querySelectorAll(".bibtex-copy").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const target = document.getElementById(btn.dataset.target);
        if (!target) return;
        const text = target.innerText;
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
          } else {
            const ta = document.createElement("textarea");
            ta.value = text;
            ta.style.position = "fixed";
            ta.style.opacity = "0";
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
          }
          const original = btn.textContent;
          btn.textContent = "Copied";
          btn.classList.add("copied");
          setTimeout(() => {
            btn.textContent = original;
            btn.classList.remove("copied");
          }, 1500);
        } catch (_) {
          /* clipboard blocked — leave button as-is */
        }
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
