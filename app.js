/**
 * AEGIS Architectural Presentation Site
 * Pure Vanilla JavaScript — Zero Framework, Zero Dependencies
 * Features:
 *  - Procedural Topographic/Contour Canvas Mesh
 *  - Eased Number Counter (requestAnimationFrame)
 *  - Scroll-Pinned Pipeline Progress Thread
 *  - IntersectionObserver Staggered Reveals
 *  - Interactive Historical Map Layer Toggle
 *  - Prefers-Reduced-Motion Compliance
 */

(function () {
  'use strict';

  // Check user preference for reduced motion
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ==========================================================================
     1. Sticky Navbar State
     ========================================================================== */
  const navbar = document.getElementById('navbar');
  function handleScrollNav() {
    if (!navbar) return;
    if (window.scrollY > 40) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  }
  window.addEventListener('scroll', handleScrollNav, { passive: true });
  handleScrollNav();

  /* ==========================================================================
     2. Procedural Topographic Contour Background Canvas
     ========================================================================== */
  const canvas = document.getElementById('bg-canvas');
  if (canvas && !prefersReducedMotion) {
    const ctx = canvas.getContext('2d');
    let width, height;
    let animationFrameId;
    let time = 0;

    function resizeCanvas() {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Generate smooth undulating contour isolines
    function drawContours() {
      ctx.clearRect(0, 0, width, height);

      // We draw 9 flowing isolines representing bathymetric & elevation contours
      const lineCount = 8;
      const baseSpacing = height / (lineCount + 1);

      ctx.lineWidth = 1;

      for (let i = 0; i < lineCount; i++) {
        const yBase = baseSpacing * (i + 1);
        const opacity = 0.04 + (i / lineCount) * 0.08;
        
        // Sea-glass cyan tint
        ctx.strokeStyle = `rgba(56, 189, 248, ${opacity})`;
        ctx.beginPath();

        const step = 25;
        for (let x = 0; x <= width + step; x += step) {
          // Multi-frequency sine harmonics to evoke natural topography
          const wave1 = Math.sin(x * 0.0018 + time * 0.0006 + i * 0.7) * 45;
          const wave2 = Math.cos(x * 0.0035 - time * 0.0004 + i) * 25;
          const wave3 = Math.sin(x * 0.0008 + time * 0.0002) * 60;
          
          const y = yBase + wave1 + wave2 + wave3;
          
          if (x === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }

      time += 1.2;
      animationFrameId = requestAnimationFrame(drawContours);
    }

    // Start canvas loop
    drawContours();

    // Pause when tab is inactive to preserve battery
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        cancelAnimationFrame(animationFrameId);
      } else {
        drawContours();
      }
    });
  }

  /* ==========================================================================
     3. Eased requestAnimationFrame Number Counter
     ========================================================================== */
  function easeOutQuart(x) {
    return 1 - Math.pow(1 - x, 4);
  }

  function animateCounter(el) {
    if (el.dataset.hasAnimated === 'true') return;

    // Cache the verified static fallback text from the HTML node
    const originalText = el.textContent.trim();

    // Read target value: prefer data-counter attribute, or parse directly from hardcoded HTML text node
    let targetVal = parseFloat(el.getAttribute('data-counter'));
    if (isNaN(targetVal)) {
      const match = originalText.match(/[-+]?[0-9]*\.?[0-9]+/);
      if (match) targetVal = parseFloat(match[0]);
    }

    if (isNaN(targetVal)) return;

    el.dataset.hasAnimated = 'true';

    // Parse decimal precision: prioritize data-decimals, fallback to detecting from target or text node
    let decimals = el.hasAttribute('data-decimals')
      ? parseInt(el.getAttribute('data-decimals'), 10)
      : 0;
    if (isNaN(decimals) || !el.hasAttribute('data-decimals')) {
      const decMatch = originalText.match(/\.([0-9]+)/);
      decimals = decMatch ? decMatch[1].length : 0;
    }

    // Determine suffix and prefix: prioritize attributes, fallback to extracting from text
    let suffix = el.getAttribute('data-suffix');
    if (suffix === null) {
      const sufMatch = originalText.match(/[^0-9.]*$/);
      suffix = sufMatch ? sufMatch[0] : '';
    }

    let prefix = el.getAttribute('data-prefix');
    if (prefix === null) {
      const preMatch = originalText.match(/^[^0-9.]*/);
      prefix = preMatch ? preMatch[0] : '';
    }

    // Respect prefers-reduced-motion
    if (prefersReducedMotion) {
      el.textContent = `${prefix}${targetVal.toFixed(decimals)}${suffix}`;
      return;
    }

    try {
      const duration = 1800; // ms
      let startTime = null;

      function step(timestamp) {
        try {
          if (!startTime) startTime = timestamp;
          const elapsed = timestamp - startTime;
          const progress = Math.min(elapsed / duration, 1);
          const easedProgress = easeOutQuart(progress);
          const currentVal = easedProgress * targetVal;

          el.textContent = `${prefix}${currentVal.toFixed(decimals)}${suffix}`;

          if (progress < 1) {
            requestAnimationFrame(step);
          } else {
            el.textContent = `${prefix}${targetVal.toFixed(decimals)}${suffix}`;
          }
        } catch (stepErr) {
          // If frame update is interrupted or fails, restore static verified HTML fallback
          el.textContent = originalText;
        }
      }

      // Initialize animation from 0 up to verified target value
      el.textContent = `${prefix}${(0).toFixed(decimals)}${suffix}`;
      requestAnimationFrame(step);
    } catch (err) {
      // Fall back immediately to hardcoded static text if animation initialization fails
      el.textContent = originalText;
    }
  }

  /* ==========================================================================
     4. IntersectionObserver for Reveal & Stagger Animations
     ========================================================================== */
  const revealElements = document.querySelectorAll('.reveal');
  const stageCards = document.querySelectorAll('.stage-card');
  const counterElements = document.querySelectorAll('[data-counter]');

  if ('IntersectionObserver' in window) {
    // 4a. General Section / Card Reveals
    const revealObserver = new IntersectionObserver((entries, obs) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          obs.unobserve(entry.target);
        }
      });
    }, {
      rootMargin: '0px 0px -80px 0px',
      threshold: 0.1
    });

    revealElements.forEach(el => revealObserver.observe(el));

    // 4b. Pipeline Stage Cards (for staggered spec items)
    const stageObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          // Trigger counters inside this stage card
          const cardCounters = entry.target.querySelectorAll('[data-counter]');
          cardCounters.forEach(animateCounter);
        }
      });
    }, {
      threshold: 0.25
    });

    stageCards.forEach(card => stageObserver.observe(card));

    // 4c. Standalone Metric Counters (Hero, Validation)
    const counterObserver = new IntersectionObserver((entries, obs) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          obs.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.3
    });

    counterElements.forEach(el => {
      // If element is not inside a stage card (which handles its own counters), observe directly
      if (!el.closest('.stage-card')) {
        counterObserver.observe(el);
      }
    });

  } else {
    // Fallback if IntersectionObserver is not available
    revealElements.forEach(el => el.classList.add('is-visible'));
    stageCards.forEach(card => card.classList.add('is-visible'));
    counterElements.forEach(animateCounter);
  }

  /* ==========================================================================
     5. Persistent Pipeline Connecting Thread Progress Tracker
     ========================================================================== */
  const threadProgress = document.getElementById('thread-progress');
  const threadSteps = document.querySelectorAll('.thread-step');
  const pipelineSection = document.getElementById('pipeline');

  function updatePipelineProgress() {
    if (!pipelineSection || !threadProgress || stageCards.length === 0) return;

    const pipelineRect = pipelineSection.getBoundingClientRect();
    const windowHeight = window.innerHeight;

    // Calculate how far we've scrolled inside the pipeline section
    const startY = pipelineSection.offsetTop;
    const totalHeight = pipelineSection.offsetHeight - windowHeight;
    const currentScroll = window.scrollY - startY + (windowHeight * 0.3);

    const progressFraction = Math.max(0, Math.min(1, currentScroll / totalHeight));
    threadProgress.style.height = `${progressFraction * 100}%`;

    // Determine currently active stage card
    let activeIndex = 0;
    stageCards.forEach((card, index) => {
      const cardRect = card.getBoundingClientRect();
      // If the top of the card has passed above 45% of the viewport height
      if (cardRect.top < windowHeight * 0.5) {
        activeIndex = index;
      }
    });

    threadSteps.forEach((step, index) => {
      if (index === activeIndex) {
        step.classList.add('is-active');
      } else {
        step.classList.remove('is-active');
      }
    });
  }

  window.addEventListener('scroll', updatePipelineProgress, { passive: true });
  updatePipelineProgress();

  // Allow clicking on thread step to smooth scroll
  threadSteps.forEach(step => {
    step.addEventListener('click', () => {
      const targetId = step.getAttribute('data-target');
      const targetEl = document.getElementById(targetId);
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    });
  });

  /* ==========================================================================
     6. Validation Map Interactive Layer Toggle
     ========================================================================== */
  const mapButtons = document.querySelectorAll('.map-btn');
  const layerPredicted = document.getElementById('layer-predicted');
  const layerActual = document.getElementById('layer-actual');
  const layerOverlap = document.getElementById('layer-overlap');

  mapButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      mapButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const mode = btn.getAttribute('data-layer');

      if (mode === 'all') {
        if (layerPredicted) layerPredicted.classList.remove('hidden');
        if (layerActual) layerActual.classList.remove('hidden');
        if (layerOverlap) layerOverlap.classList.remove('hidden');
      } else if (mode === 'predicted') {
        if (layerPredicted) layerPredicted.classList.remove('hidden');
        if (layerActual) layerActual.classList.add('hidden');
        if (layerOverlap) layerOverlap.classList.add('hidden');
      } else if (mode === 'actual') {
        if (layerPredicted) layerPredicted.classList.add('hidden');
        if (layerActual) layerActual.classList.remove('hidden');
        if (layerOverlap) layerOverlap.classList.add('hidden');
      }
    });
  });

  /* ==========================================================================
     7. Smooth Navigation Anchor Offset Support
     ========================================================================== */
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const href = this.getAttribute('href');
      if (href === '#') return;
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        const headerOffset = 80;
        const elementPosition = target.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth'
        });
      }
    });
  });

  /* ==========================================================================
     8. KaTeX Render Safeguard
     ========================================================================== */
  function renderAllMath() {
    if (typeof renderMathInElement === 'function') {
      renderMathInElement(document.body, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false },
          { left: '$', right: '$', display: false }
        ],
        throwOnError: false
      });
    }
  }

  renderAllMath();
  window.addEventListener('load', renderAllMath);

  /* ==========================================================================
     9. V-JEPA Before/After Latent Comparison Slider
     ========================================================================== */
  const vjepaSlider = document.getElementById('vjepa-slider');
  const vjepaClipLayer = document.getElementById('vjepa-clip-layer');
  const vjepaDivider = document.getElementById('vjepa-divider');
  const compButtons = document.querySelectorAll('.comp-btn');
  const vjepaCompareContainer = document.getElementById('vjepa-compare');

  function updateVjepaSplit(percentage) {
    const val = Math.max(0, Math.min(100, parseFloat(percentage)));
    if (vjepaClipLayer) {
      vjepaClipLayer.style.clipPath = `polygon(0 0, ${val}% 0, ${val}% 100%, 0 100%)`;
    }
    if (vjepaDivider) {
      vjepaDivider.style.left = `${val}%`;
    }
    if (vjepaSlider && vjepaSlider.value !== String(val)) {
      vjepaSlider.value = val;
    }

    compButtons.forEach(btn => {
      const btnVal = parseFloat(btn.getAttribute('data-val'));
      if (Math.abs(btnVal - val) < 5) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });
  }

  if (vjepaSlider) {
    vjepaSlider.addEventListener('input', (e) => {
      updateVjepaSplit(e.target.value);
    });
  }

  compButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const val = btn.getAttribute('data-val');
      updateVjepaSplit(val);
    });
  });

  // Direct drag/click interaction on the comparison container
  if (vjepaCompareContainer) {
    let isDraggingVjepa = false;

    function handlePointerMove(e) {
      if (!isDraggingVjepa) return;
      const rect = vjepaCompareContainer.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const x = clientX - rect.left;
      const percentage = (x / rect.width) * 100;
      updateVjepaSplit(percentage);
    }

    vjepaCompareContainer.addEventListener('mousedown', (e) => {
      isDraggingVjepa = true;
      handlePointerMove(e);
    });

    window.addEventListener('mousemove', handlePointerMove);
    window.addEventListener('mouseup', () => { isDraggingVjepa = false; });

    vjepaCompareContainer.addEventListener('touchstart', (e) => {
      isDraggingVjepa = true;
      handlePointerMove(e);
    }, { passive: true });

    window.addEventListener('touchmove', handlePointerMove, { passive: true });
    window.addEventListener('touchend', () => { isDraggingVjepa = false; });
  }

  /* ==========================================================================
     10. Julia Physics 2D Grid-Cell Flood Propagation Metaphor
     ========================================================================== */
  const simGridContainer = document.getElementById('sim-grid');
  const simStepBtn = document.getElementById('sim-step-btn');
  const simSurgeBtn = document.getElementById('sim-surge-btn');
  const simResetBtn = document.getElementById('sim-reset-btn');
  const simStepCount = document.getElementById('sim-step-count');
  const simInundatedCount = document.getElementById('sim-inundated-count');

  // 7 cols x 4 rows elevation terrain (coastal low on left, higher ridge on right)
  const elevMatrix = [
    [1.2, 2.4, 4.8, 7.2, 11.5, 16.0, 21.0], // Row 0 (North bank)
    [0.8, 1.4, 2.2, 3.8,  6.4, 12.0, 18.5], // Row 1 (Central plain)
    [0.5, 0.9, 1.6, 2.8,  5.0,  9.8, 15.2], // Row 2 (River Delta)
    [1.0, 1.8, 3.2, 5.5,  8.8, 14.5, 19.8]  // Row 3 (South plain)
  ];

  let simCellsData = [];
  let currentStep = 0;

  function initSimGrid() {
    if (!simGridContainer) return;
    simGridContainer.innerHTML = '';
    simCellsData = [];
    currentStep = 0;

    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 7; c++) {
        const elev = elevMatrix[r][c];
        const cellObj = {
          r,
          c,
          elev,
          state: 'dry', // 'dry', 'wave', 'deep', 'barrier'
          depth: 0
        };

        const cellEl = document.createElement('div');
        cellEl.className = 'sim-cell';
        cellEl.dataset.r = r;
        cellEl.dataset.c = c;
        cellEl.setAttribute('role', 'button');
        cellEl.setAttribute('tabindex', '0');
        cellEl.setAttribute('title', `Cell (${c}, ${r}) · Elevation: ${elev}m · Click to toggle barrier`);

        cellEl.innerHTML = `
          <span class="sim-cell-coord">(${c},${r})</span>
          <span class="sim-cell-val">${elev}m</span>
        `;

        cellEl.addEventListener('click', () => {
          if (cellObj.state === 'barrier') {
            cellObj.state = 'dry';
            cellObj.depth = 0;
          } else {
            cellObj.state = 'barrier';
            cellObj.depth = 0;
          }
          renderCell(cellEl, cellObj);
          updateTelemetry();
        });

        cellObj.el = cellEl;
        simCellsData.push(cellObj);
        simGridContainer.appendChild(cellEl);
      }
    }
    updateTelemetry();
  }

  function renderCell(el, cell) {
    el.className = 'sim-cell';
    const valEl = el.querySelector('.sim-cell-val');

    if (cell.state === 'barrier') {
      el.classList.add('is-barrier');
      if (valEl) valEl.textContent = 'BARRIER';
    } else if (cell.state === 'deep') {
      el.classList.add('is-wet-deep');
      if (valEl) valEl.textContent = `h:${cell.depth.toFixed(1)}m`;
    } else if (cell.state === 'wave') {
      el.classList.add('is-wet-wave');
      if (valEl) valEl.textContent = `h:${cell.depth.toFixed(1)}m`;
    } else {
      if (valEl) valEl.textContent = `${cell.elev}m`;
    }
  }

  function updateTelemetry() {
    if (simStepCount) simStepCount.textContent = currentStep;
    if (simInundatedCount) {
      const wetCount = simCellsData.filter(c => c.state === 'wave' || c.state === 'deep').length;
      simInundatedCount.textContent = wetCount;
    }
  }

  function stepPropagation() {
    currentStep++;

    simCellsData.forEach(cell => {
      if (cell.state === 'barrier') return;

      // Inundation wave expands column by column through elevation gradient
      if (cell.c === 0 && currentStep >= 1) {
        if (cell.state === 'dry') {
          cell.state = 'wave';
          cell.depth = 1.4;
        } else if (cell.state === 'wave') {
          cell.state = 'deep';
          cell.depth = 2.4;
        }
      } else if (cell.c === 1 && currentStep >= 2) {
        if (cell.state === 'dry') {
          cell.state = 'wave';
          cell.depth = 1.2;
        } else if (cell.state === 'wave') {
          cell.state = 'deep';
          cell.depth = 1.8;
        }
      } else if (cell.c === 2 && currentStep >= 3) {
        // Deltas flood earlier
        if (cell.state === 'dry') {
          cell.state = 'wave';
          cell.depth = 0.9;
        } else if (cell.state === 'wave') {
          cell.state = 'deep';
          cell.depth = 1.4;
        }
      } else if (cell.c === 3 && currentStep >= 4) {
        if (cell.elev <= 6.0) {
          if (cell.state === 'dry') {
            cell.state = 'wave';
            cell.depth = 0.6;
          }
        }
      } else if (cell.c === 4 && currentStep >= 5) {
        if (cell.elev <= 7.0 && cell.state === 'dry') {
          cell.state = 'wave';
          cell.depth = 0.4;
        }
      }

      renderCell(cell.el, cell);
    });

    updateTelemetry();
  }

  function injectSurge() {
    currentStep += 2;
    simCellsData.forEach(cell => {
      if (cell.state === 'barrier') return;
      if (cell.c <= 2) {
        cell.state = 'deep';
        cell.depth = 2.8 - (cell.c * 0.5);
      } else if (cell.c === 3 && cell.elev < 6.0) {
        cell.state = 'wave';
        cell.depth = 1.1;
      }
      renderCell(cell.el, cell);
    });
    updateTelemetry();
  }

  function resetSim() {
    currentStep = 0;
    simCellsData.forEach(cell => {
      if (cell.state !== 'barrier') {
        cell.state = 'dry';
        cell.depth = 0;
      }
      renderCell(cell.el, cell);
    });
    updateTelemetry();
  }

  if (simStepBtn) simStepBtn.addEventListener('click', stepPropagation);
  if (simSurgeBtn) simSurgeBtn.addEventListener('click', injectSurge);
  if (simResetBtn) simResetBtn.addEventListener('click', resetSim);

  initSimGrid();

  /* ==========================================================================
     11. Validation Historical Opacity Cross-Fader Slider
     ========================================================================== */
  const faniBlendSlider = document.getElementById('fani-blend-slider');
  const blendReadout = document.getElementById('blend-readout');

  function updateFaniBlend(val) {
    const fraction = parseFloat(val) / 100; // 0 (100% Sim) to 1 (100% SAR)

    if (layerPredicted) {
      layerPredicted.classList.remove('hidden');
      // Fade out prediction as we approach 100% SAR
      const predOpacity = Math.max(0, 1 - (fraction * 1.05));
      layerPredicted.style.opacity = predOpacity.toFixed(2);
    }

    if (layerActual) {
      layerActual.classList.remove('hidden');
      // Fade in ground truth SAR as we move right
      const actOpacity = Math.min(1, fraction * 1.05);
      layerActual.style.opacity = actOpacity.toFixed(2);
    }

    if (layerOverlap) {
      layerOverlap.classList.remove('hidden');
      // Overlap peaks at 50% composite view
      const overlapOpacity = Math.max(0, 1 - Math.abs(fraction - 0.5) * 2.2);
      layerOverlap.style.opacity = overlapOpacity.toFixed(2);
    }

    if (blendReadout) {
      if (val === 0 || val === '0') {
        blendReadout.textContent = 'Simulation Only · 100% Numerical Physics';
      } else if (val < 45) {
        blendReadout.textContent = `Simulation Bias (${100 - val}% Sim / ${val}% Radar)`;
      } else if (val <= 55) {
        blendReadout.textContent = 'Composite View · 85.6% Spatial IoU Overlap';
      } else if (val < 100) {
        blendReadout.textContent = `Ground Truth Bias (${100 - val}% Sim / ${val}% Radar)`;
      } else {
        blendReadout.textContent = 'EMSR357 Radar Only · 100% Ground Truth';
      }
    }

    // Keep top map-btn buttons synchronized
    mapButtons.forEach(btn => {
      const mode = btn.getAttribute('data-layer');
      if (val <= 10 && mode === 'predicted') {
        btn.classList.add('active');
      } else if (val >= 90 && mode === 'actual') {
        btn.classList.add('active');
      } else if (val > 40 && val < 60 && mode === 'all') {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });
  }

  if (faniBlendSlider) {
    faniBlendSlider.addEventListener('input', (e) => {
      updateFaniBlend(e.target.value);
    });
  }

  // Allow clicking original map-btn buttons to smoothly glide the slider
  mapButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-layer');
      if (mode === 'all') {
        if (faniBlendSlider) faniBlendSlider.value = 50;
        updateFaniBlend(50);
      } else if (mode === 'predicted') {
        if (faniBlendSlider) faniBlendSlider.value = 0;
        updateFaniBlend(0);
      } else if (mode === 'actual') {
        if (faniBlendSlider) faniBlendSlider.value = 100;
        updateFaniBlend(100);
      }
    });
  });

})();
