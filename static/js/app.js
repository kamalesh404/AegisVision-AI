/**
 * AegisVision AI — Enterprise Real-Time YOLO Video Analytics Dashboard
 * Frontend Application Module
 *
 * @fileoverview Complete client-side logic for the AegisVision AI dashboard.
 * Handles status polling, mode switching, video feed, configuration,
 * Chart.js visualizations, incident logging, keyboard shortcuts, and toast notifications.
 *
 * @version 1.0.0
 */

/* =========================================================================
 *  Toast Notification System
 * ========================================================================= */

/**
 * Lightweight toast notification manager.
 * Creates a fixed container and renders ephemeral messages with severity styling.
 */
const Toast = (() => {
  let container = null;

  /**
   * Ensure the toast container exists in the DOM.
   * @returns {HTMLElement} The toast container element.
   */
  const _ensureContainer = () => {
    if (!container) {
      container = document.createElement('div');
      container.id = 'toast-container';
      Object.assign(container.style, {
        position: 'fixed',
        bottom: '24px',
        right: '24px',
        zIndex: '10000',
        display: 'flex',
        flexDirection: 'column-reverse',
        gap: '10px',
        pointerEvents: 'none',
      });
      document.body.appendChild(container);
    }
    return container;
  };

  /**
   * Map severity to accent colour.
   * @param {'info'|'success'|'warning'|'error'} severity
   * @returns {string} CSS colour string.
   */
  const _color = (severity) => {
    const map = {
      info: '#00b4d8',
      success: '#00e676',
      warning: '#ffab00',
      error: '#ff1744',
    };
    return map[severity] || map.info;
  };

  /**
   * Display a toast notification.
   * @param {string} message - The message to display.
   * @param {'info'|'success'|'warning'|'error'} [severity='info'] - Severity level.
   * @param {number} [duration=3500] - Time in ms before auto-dismiss.
   */
  const show = (message, severity = 'info', duration = 3500) => {
    const wrap = _ensureContainer();
    const el = document.createElement('div');
    const accent = _color(severity);
    Object.assign(el.style, {
      background: 'rgba(15,20,30,0.92)',
      color: '#e0e0e0',
      padding: '12px 20px',
      borderRadius: '8px',
      borderLeft: `4px solid ${accent}`,
      boxShadow: `0 4px 24px rgba(0,0,0,0.5), 0 0 8px ${accent}44`,
      fontFamily: "'Inter', 'Segoe UI', sans-serif",
      fontSize: '13px',
      lineHeight: '1.4',
      maxWidth: '380px',
      pointerEvents: 'auto',
      opacity: '0',
      transform: 'translateY(16px)',
      transition: 'opacity 0.3s ease, transform 0.3s ease',
    });
    el.textContent = message;
    wrap.appendChild(el);

    // Trigger entrance animation
    requestAnimationFrame(() => {
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    });

    // Auto-dismiss
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(16px)';
      setTimeout(() => el.remove(), 350);
    }, duration);
  };

  return { show };
})();


/* =========================================================================
 *  AegisVisionApp — Main Application Class
 * ========================================================================= */

class AegisVisionApp {

  /* -----------------------------------------------------------------------
   *  Constructor & Instance State
   * --------------------------------------------------------------------- */

  constructor() {
    /** @type {number|null} Status polling interval ID */
    this._statusInterval = null;

    /** @type {number|null} Events polling interval ID */
    this._eventsInterval = null;

    /** @type {boolean} Whether the feed is currently paused */
    this.paused = false;

    /** @type {string} Currently active mode slug */
    this.activeMode = '';

    /** @type {Object[]} Cached incident events */
    this.events = [];

    /** @type {Chart|null} Chart.js doughnut instance */
    this.classChart = null;

    /** @type {Chart|null} Chart.js line instance */
    this.crossingChart = null;

    /** @type {number} Incrementing poll tick for crossing timeline x-axis */
    this.pollTick = 0;

    /** @type {number[]} Cumulative IN values for crossing chart */
    this.crossingInData = [];

    /** @type {number[]} Cumulative OUT values for crossing chart */
    this.crossingOutData = [];

    /** @type {string[]} Time labels for crossing chart */
    this.crossingLabels = [];

    /** @type {number} Max data points to keep in crossing chart */
    this.MAX_CROSSING_POINTS = 60;

    /** @type {number} Max rows to show in incident table */
    this.MAX_INCIDENT_ROWS = 30;

    /** @type {Set<string>} Set of already-rendered event IDs to detect new rows */
    this.renderedEventIds = new Set();

    /** @type {boolean} Guard to avoid firing toast bursts during initial load */
    this._hasLoadedEvents = false;

    /** Professional colour palette for doughnut chart segments */
    this.CLASS_COLORS = [
      '#00b4d8', '#00e676', '#ff9100', '#e040fb', '#ffea00',
      '#ff1744', '#00e5ff', '#76ff03', '#d500f9', '#ffc400',
      '#ff3d00', '#1de9b6', '#651fff', '#f50057', '#00b0ff',
      '#c6ff00', '#ff6e40', '#6200ea', '#64ffda', '#aa00ff',
    ];

    /** @type {Object} Local config mirror */
    this.config = {
      show_trails: false,
      show_heatmap: false,
      show_ids: true,
      show_confidence: true,
      paused: false,
    };
  }


  /* -----------------------------------------------------------------------
   *  Initialisation
   * --------------------------------------------------------------------- */

  /**
   * Bootstrap the entire application.
   * Called once on DOMContentLoaded.
   */
  async init() {
    this._cacheDOM();
    this._loadPreferences();
    this._initCharts();
    this._setVideoFeed();
    this._bindConfigToggles();
    this._bindActionButtons();
    this._bindSourceInput();
    this._bindKeyboardShortcuts();

    await this._fetchAndRenderModes();
    this._startStatusPolling();
    this._startEventsPolling();

    Toast.show('AegisVision AI initialised', 'success');
  }

  /**
   * Cache frequently accessed DOM elements for performance.
   */
  _cacheDOM() {
    // Stat cards
    this.$fps = document.getElementById('stat-fps') || document.getElementById('statFps');
    this.$tracks = document.getElementById('stat-tracks') || document.getElementById('statTracks');
    this.$inference = document.getElementById('stat-inference') || document.getElementById('statInference');
    this.$crossings = document.getElementById('stat-crossings') || document.getElementById('statCrossings');

    // Model info
    this.$modelInfo = document.getElementById('model-info') || document.getElementById('modelBadge');

    // Video
    this.$videoFeed = document.getElementById('video-feed') || document.getElementById('videoFeed');
    this.$liveBadge = document.getElementById('live-badge') || document.querySelector('.live-badge');
    this.$fpsBadge = document.getElementById('fps-badge') || document.getElementById('fpsBadge');

    // Mode bar
    this.$modeBar = document.getElementById('mode-bar') || document.getElementById('modeSelector');

    // Config toggles
    this.$toggleTrails = document.getElementById('toggle-trails') || document.getElementById('toggleTrails');
    this.$toggleHeatmap = document.getElementById('toggle-heatmap') || document.getElementById('toggleHeatmap');
    this.$toggleIds = document.getElementById('toggle-ids') || document.getElementById('toggleIds');
    this.$toggleConfidence = document.getElementById('toggle-confidence') || document.getElementById('toggleConfidence');

    // Model selector
    this.$modelSelect = document.getElementById('model-select') || document.getElementById('modelSelect');

    // Pause button
    this.$pauseBtn = document.getElementById('btn-pause') || document.getElementById('btnPause');

    // Source input
    this.$sourceInput = document.getElementById('source-input') || document.getElementById('sourceInput');
    this.$sourceBtn = document.getElementById('btn-connect') || document.getElementById('btnConnect');
    this.$sourceStatus = document.getElementById('source-status');

    // Action buttons
    this.$resetBtn = document.getElementById('btn-reset') || document.getElementById('btnReset');
    this.$snapshotBtn = document.getElementById('btn-snapshot') || document.getElementById('btnSnapshot');
    this.$exportBtn = document.getElementById('btn-export') || document.getElementById('btnExportCSV');

    // Charts
    this.$classChartCanvas = document.getElementById('chart-class-distribution') || document.getElementById('chartDistribution');
    this.$crossingChartCanvas = document.getElementById('chart-crossing-timeline') || document.getElementById('chartTimeline');

    // Incident table
    this.$incidentTbody = document.getElementById('incident-tbody') || document.getElementById('incidentTableBody');

    // Spatial & Capacity Telemetry Container
    this.$zoneTelemetryContainer = document.getElementById('zoneTelemetryContainer');
  }


  /* -----------------------------------------------------------------------
   *  LocalStorage Preferences
   * --------------------------------------------------------------------- */

  /**
   * Load saved preferences from localStorage and apply them.
   */
  _loadPreferences() {
    try {
      const saved = localStorage.getItem('aegis_prefs');
      if (saved) {
        const prefs = JSON.parse(saved);
        this.config = { ...this.config, ...prefs };
      }
    } catch (_) {
      // Ignore corrupt storage
    }
  }

  /**
   * Persist current config to localStorage.
   */
  _savePreferences() {
    try {
      localStorage.setItem('aegis_prefs', JSON.stringify(this.config));
    } catch (_) {
      // Quota exceeded — non-critical
    }
  }


  /* -----------------------------------------------------------------------
   *  Video Feed
   * --------------------------------------------------------------------- */

  /**
   * Point the video <img> element at the MJPEG stream endpoint.
   */
  _setVideoFeed() {
    if (this.$videoFeed) {
      this.$videoFeed.src = '/video_feed';
      this.$videoFeed.addEventListener('load', () => {
        if (this.$liveBadge) this.$liveBadge.style.display = 'flex';
      });
      this.$videoFeed.addEventListener('error', () => {
        if (this.$liveBadge) this.$liveBadge.style.display = 'none';
      });
    }
  }


  /* -----------------------------------------------------------------------
   *  Chart.js Initialisation
   * --------------------------------------------------------------------- */

  /**
   * Create both Chart.js chart instances.
   */
  _initCharts() {
    this._initClassChart();
    this._initCrossingChart();
  }

  /**
   * Initialise the Object Distribution doughnut chart.
   */
  _initClassChart() {
    if (!this.$classChartCanvas) return;
    const ctx = this.$classChartCanvas.getContext('2d');
    this.classChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: [],
        datasets: [{
          data: [],
          backgroundColor: this.CLASS_COLORS,
          borderColor: 'rgba(10,14,20,0.8)',
          borderWidth: 2,
          hoverBorderColor: '#ffffff',
          hoverBorderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        animation: {
          animateRotate: true,
          duration: 600,
          easing: 'easeInOutQuart',
        },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: '#a0aec0',
              font: { family: "'Inter', sans-serif", size: 11 },
              padding: 12,
              usePointStyle: true,
              pointStyleWidth: 10,
            },
          },
          tooltip: {
            backgroundColor: 'rgba(15,20,30,0.92)',
            titleColor: '#e0e0e0',
            bodyColor: '#a0aec0',
            borderColor: '#00b4d8',
            borderWidth: 1,
            cornerRadius: 8,
            padding: 10,
          },
        },
      },
    });
  }

  /**
   * Initialise the Crossing Timeline line chart.
   */
  _initCrossingChart() {
    if (!this.$crossingChartCanvas) return;
    const ctx = this.$crossingChartCanvas.getContext('2d');
    this.crossingChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'IN',
            data: [],
            borderColor: '#00e676',
            backgroundColor: 'rgba(0,230,118,0.08)',
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 2,
          },
          {
            label: 'OUT',
            data: [],
            borderColor: '#ff6d00',
            backgroundColor: 'rgba(255,109,0,0.08)',
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        animation: {
          duration: 400,
          easing: 'easeOutQuart',
        },
        scales: {
          x: {
            display: true,
            ticks: {
              color: '#596575',
              font: { size: 10 },
              maxTicksLimit: 12,
              maxRotation: 0,
            },
            grid: {
              color: 'rgba(255,255,255,0.04)',
            },
          },
          y: {
            display: true,
            beginAtZero: true,
            ticks: {
              color: '#596575',
              font: { size: 10 },
              precision: 0,
            },
            grid: {
              color: 'rgba(255,255,255,0.06)',
            },
          },
        },
        plugins: {
          legend: {
            labels: {
              color: '#a0aec0',
              font: { family: "'Inter', sans-serif", size: 11 },
              usePointStyle: true,
              pointStyleWidth: 10,
              padding: 16,
            },
          },
          tooltip: {
            backgroundColor: 'rgba(15,20,30,0.92)',
            titleColor: '#e0e0e0',
            bodyColor: '#a0aec0',
            borderColor: '#00b4d8',
            borderWidth: 1,
            cornerRadius: 8,
            padding: 10,
          },
        },
      },
    });
  }


  /* -----------------------------------------------------------------------
   *  API Helpers
   * --------------------------------------------------------------------- */

  /**
   * Perform a JSON GET request.
   * @param {string} url - Endpoint URL.
   * @returns {Promise<any>} Parsed JSON response.
   */
  async _get(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GET ${url} → ${res.status}`);
    return res.json();
  }

  /**
   * Perform a JSON POST request.
   * @param {string} url - Endpoint URL.
   * @param {Object} body - JSON-serialisable payload.
   * @returns {Promise<any>} Parsed JSON response.
   */
  async _post(url, body) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`POST ${url} → ${res.status}`);
    return res.json();
  }


  /* -----------------------------------------------------------------------
   *  Status Polling
   * --------------------------------------------------------------------- */

  /**
   * Begin polling the status endpoint.
   */
  _startStatusPolling() {
    // Immediate first fetch
    this._pollStatus();
    this._statusInterval = setInterval(() => this._pollStatus(), 800);
  }

  /**
   * Single status poll iteration.
   */
  async _pollStatus() {
    try {
      const s = await this._get('/api/status');
      this._updateStatCards(s);
      this._updateModelInfo(s);
      this._updateFPSBadge(s);
      this._syncConfigState(s);
      this._updateClassChart(s);
      this._updateCrossingChart(s);
      this._pollSpatialAnalytics();
    } catch (err) {
      console.warn('[AegisVision] Status poll error:', err);
    }
  }

  /**
   * Update the four top stat cards.
   * @param {Object} s - Status response.
   */
  _updateStatCards(s) {
    if (this.$fps) {
      this.$fps.textContent = s.fps != null ? parseFloat(s.fps).toFixed(1) : '--';
    }
    if (this.$tracks) {
      this.$tracks.textContent = s.track_count != null ? s.track_count : '--';
    }
    if (this.$inference) {
      const ms = s.inference_ms ?? s.inference_time ?? null;
      this.$inference.textContent = ms != null ? `${parseFloat(ms).toFixed(1)}ms` : '--';
    }
    if (this.$crossings) {
      let total = 0;
      if (Array.isArray(s.tripwire_stats)) {
        for (const tw of s.tripwire_stats) {
          total += (tw.in_count || 0) + (tw.out_count || 0);
        }
      }
      this.$crossings.textContent = total;
    }
  }

  /**
   * Update the model info display.
   * @param {Object} s - Status response.
   */
  _updateModelInfo(s) {
    if (this.$modelInfo && s.model_info) {
      const info = typeof s.model_info === 'string' ? s.model_info : (s.model_info.name || JSON.stringify(s.model_info));
      this.$modelInfo.textContent = info;
    }
  }

  /**
   * Update the FPS overlay badge on the video feed.
   * @param {Object} s - Status response.
   */
  _updateFPSBadge(s) {
    if (this.$fpsBadge && s.fps != null) {
      this.$fpsBadge.textContent = `${parseFloat(s.fps).toFixed(1)} FPS`;
      this.$fpsBadge.style.display = 'flex';
    }
  }

  /**
   * Sync local config state with the server's reported config.
   * @param {Object} s - Status response.
   */
  _syncConfigState(s) {
    if (s.config) {
      if (s.config.paused !== undefined) {
        this.paused = s.config.paused;
        this._updatePauseButton();
      }
      // Sync toggle states from server truth
      if (s.config.show_trails !== undefined) this.config.show_trails = s.config.show_trails;
      if (s.config.show_heatmap !== undefined) this.config.show_heatmap = s.config.show_heatmap;
      if (s.config.show_ids !== undefined) this.config.show_ids = s.config.show_ids;
      if (s.config.show_confidence !== undefined) this.config.show_confidence = s.config.show_confidence;

      this._syncToggles();
    }
    if (s.mode) {
      this.activeMode = s.mode;
      this._highlightActiveMode();
    }
  }

  /**
   * Synchronise toggle switch DOM states to local config.
   */
  _syncToggles() {
    if (this.$toggleTrails) this.$toggleTrails.checked = this.config.show_trails;
    if (this.$toggleHeatmap) this.$toggleHeatmap.checked = this.config.show_heatmap;
    if (this.$toggleIds) this.$toggleIds.checked = this.config.show_ids;
    if (this.$toggleConfidence) this.$toggleConfidence.checked = this.config.show_confidence;
  }


  /* -----------------------------------------------------------------------
   *  Chart Updates
   * --------------------------------------------------------------------- */

  /**
   * Update the Object Distribution doughnut chart from status data.
   * @param {Object} s - Status response.
   */
  _updateClassChart(s) {
    if (!this.classChart) return;

    const byClass = s.track_count_by_class || s.class_counts || {};
    const labels = Object.keys(byClass);
    const data = Object.values(byClass);

    this.classChart.data.labels = labels;
    this.classChart.data.datasets[0].data = data;
    this.classChart.data.datasets[0].backgroundColor = labels.map((_, i) => this.CLASS_COLORS[i % this.CLASS_COLORS.length]);
    this.classChart.update('none'); // Skip animation for polling updates
  }

  /**
   * Update the Crossing Timeline line chart with cumulative IN/OUT data.
   * @param {Object} s - Status response.
   */
  _updateCrossingChart(s) {
    if (!this.crossingChart) return;

    this.pollTick++;

    let totalIn = 0;
    let totalOut = 0;
    if (Array.isArray(s.tripwire_stats)) {
      for (const tw of s.tripwire_stats) {
        totalIn += tw.in_count || 0;
        totalOut += tw.out_count || 0;
      }
    }

    this.crossingLabels.push(String(this.pollTick));
    this.crossingInData.push(totalIn);
    this.crossingOutData.push(totalOut);

    // Trim to max points
    while (this.crossingLabels.length > this.MAX_CROSSING_POINTS) {
      this.crossingLabels.shift();
      this.crossingInData.shift();
      this.crossingOutData.shift();
    }

    this.crossingChart.data.labels = [...this.crossingLabels];
    this.crossingChart.data.datasets[0].data = [...this.crossingInData];
    this.crossingChart.data.datasets[1].data = [...this.crossingOutData];
    this.crossingChart.update('none');
  }

  /**
   * Reset local chart data — called on mode switch.
   */
  _resetChartData() {
    this.pollTick = 0;
    this.crossingLabels = [];
    this.crossingInData = [];
    this.crossingOutData = [];

    if (this.classChart) {
      this.classChart.data.labels = [];
      this.classChart.data.datasets[0].data = [];
      this.classChart.update();
    }
    if (this.crossingChart) {
      this.crossingChart.data.labels = [];
      this.crossingChart.data.datasets[0].data = [];
      this.crossingChart.data.datasets[1].data = [];
      this.crossingChart.update();
    }
  }


  /* -----------------------------------------------------------------------
   *  Spatial Analytics & Capacity Telemetry
   * --------------------------------------------------------------------- */

  /**
   * Poll live zone capacity and dwell analytics endpoints and update telemetry widget.
   */
  async _pollSpatialAnalytics() {
    if (!this.$zoneTelemetryContainer) return;
    try {
      const [occResp, dwellResp] = await Promise.all([
        this._get('/api/analytics/occupancy'),
        this._get('/api/analytics/dwell')
      ]);

      const occList = occResp?.occupancy_analytics || [];
      const dwellList = dwellResp?.dwell_analytics || [];
      const dwellMap = new Map(dwellList.map((d) => [d.zone_id, d]));

      if (occList.length === 0) {
        this.$zoneTelemetryContainer.innerHTML = '<div class="zone-empty-text">No active zones monitored in current mode</div>';
        return;
      }

      let html = '';
      for (const zone of occList) {
        const dwell = dwellMap.get(zone.zone_id) || {};
        const isBreached = zone.is_breached;
        const currentOcc = zone.current_occupancy || 0;
        const maxCap = zone.max_capacity || 0;
        const areaPx = zone.area_px || 0;
        const dwellThresh = dwell.dwell_threshold_sec || 0;
        const loiterCount = dwell.loitering_incidents || 0;

        const pct = maxCap > 0 ? Math.min(100, Math.round((currentOcc / maxCap) * 100)) : (currentOcc > 0 ? 100 : 0);

        let maxDwellSec = 0;
        if (dwell.current_dwell_times) {
          const vals = Object.values(dwell.current_dwell_times);
          if (vals.length > 0) maxDwellSec = Math.max(...vals);
        }

        html += `
          <div class="zone-card-item ${isBreached ? 'breached' : ''}">
            <div class="zone-card-header">
              <span>${zone.label}</span>
              <div class="zone-card-badges">
                ${isBreached ? '<span class="zone-badge-pill danger">BREACH</span>' : ''}
                ${loiterCount > 0 ? `<span class="zone-badge-pill warning">${loiterCount} LOITER</span>` : ''}
                <span class="zone-badge-pill ${currentOcc > 0 ? 'success' : ''}">${currentOcc} active</span>
              </div>
            </div>

            ${maxCap > 0 ? `
              <div class="progress-bar ${isBreached ? 'danger' : ''}">
                <div class="progress-fill" style="width: ${pct}%"></div>
              </div>
            ` : ''}

            <div class="zone-meta-row">
              <span>${maxCap > 0 ? `Cap: ${currentOcc}/${maxCap} (${pct}%)` : 'Uncapped zone'} · ${Number(areaPx).toLocaleString()} px²</span>
              ${dwellThresh > 0 ? `<span class="zone-dwell-badge">${maxDwellSec > 0 ? `${maxDwellSec.toFixed(1)}s` : '0s'} / ${dwellThresh}s limit</span>` : ''}
            </div>
          </div>
        `;
      }

      this.$zoneTelemetryContainer.innerHTML = html;
    } catch (err) {
      console.warn('[AegisVision] Spatial analytics poll error:', err);
    }
  }


  /* -----------------------------------------------------------------------
   *  Mode Switching
   * --------------------------------------------------------------------- */

  /**
   * Fetch available modes from the API and render selector buttons.
   */
  async _fetchAndRenderModes() {
    try {
      const resp = await this._get('/api/modes');
      const modesList = Array.isArray(resp) ? resp : (resp.modes || []);
      this._renderModeButtons(modesList);
    } catch (err) {
      console.warn('[AegisVision] Failed to fetch modes:', err);
      // Fallback: render default modes
      this._renderModeButtons([
        { key: 'safety', label: 'Safety', icon: '🦺' },
        { key: 'traffic', label: 'Traffic', icon: '🚗' },
        { key: 'retail', label: 'Retail', icon: '🛒' },
        { key: 'security', label: 'Security', icon: '🔒' },
      ]);
    }
  }

  /**
   * Render mode buttons into the mode bar.
   * @param {Object[]} modes - Array of mode descriptors.
   */
  _renderModeButtons(modes) {
    if (!this.$modeBar) return;
    this.$modeBar.innerHTML = '';

    for (const mode of modes) {
      const btn = document.createElement('button');
      btn.className = 'mode-btn';
      btn.dataset.mode = mode.key || mode.slug || mode.id || mode.name || '';
      btn.innerHTML = `<span class="mode-icon">${mode.icon || '⚙️'}</span><span class="mode-label">${mode.label || mode.name || mode.key || ''}</span>`;
      btn.addEventListener('click', () => this._switchMode(btn.dataset.mode));
      this.$modeBar.appendChild(btn);
    }

    this._highlightActiveMode();
  }

  /**
   * Switch to a new operational mode.
   * @param {string} mode - Mode slug.
   */
  async _switchMode(mode) {
    try {
      await this._post('/api/mode', { mode });
      this.activeMode = mode;
      this._highlightActiveMode();
      this._resetChartData();
      this.renderedEventIds.clear();
      if (this.$incidentTbody) this.$incidentTbody.innerHTML = '';
      Toast.show(`Mode switched to ${mode.toUpperCase()}`, 'info');
    } catch (err) {
      Toast.show(`Failed to switch mode: ${err.message}`, 'error');
    }
  }

  /**
   * Apply the `.active` class to the button matching the current mode.
   */
  _highlightActiveMode() {
    if (!this.$modeBar) return;
    const buttons = this.$modeBar.querySelectorAll('.mode-btn');
    buttons.forEach((btn) => {
      const isActive = btn.dataset.mode === this.activeMode;
      btn.classList.toggle('active', isActive);
    });
  }


  /* -----------------------------------------------------------------------
   *  Configuration Panel
   * --------------------------------------------------------------------- */

  /**
   * Bind change events to configuration toggle switches and dropdowns.
   */
  _bindConfigToggles() {
    // Toggle switches
    const toggles = [
      { el: this.$toggleTrails, key: 'show_trails' },
      { el: this.$toggleHeatmap, key: 'show_heatmap' },
      { el: this.$toggleIds, key: 'show_ids' },
      { el: this.$toggleConfidence, key: 'show_confidence' },
    ];

    for (const { el, key } of toggles) {
      if (!el) continue;
      el.addEventListener('change', async () => {
        this.config[key] = el.checked;
        this._savePreferences();
        try {
          await this._post('/api/config', { [key]: el.checked });
        } catch (err) {
          Toast.show(`Config update failed: ${err.message}`, 'error');
        }
      });
    }

    // Model dropdown
    if (this.$modelSelect) {
      this.$modelSelect.addEventListener('change', async () => {
        const model = this.$modelSelect.value;
        try {
          await this._post('/api/model', { model });
          Toast.show(`Model changed to ${model}`, 'success');
        } catch (err) {
          Toast.show(`Model switch failed: ${err.message}`, 'error');
        }
      });
    }

    // Pause / Resume button
    if (this.$pauseBtn) {
      this.$pauseBtn.addEventListener('click', () => this._togglePause());
    }
  }

  /**
   * Toggle the paused state and notify the server.
   */
  async _togglePause() {
    this.paused = !this.paused;
    this._updatePauseButton();
    try {
      await this._post('/api/config', { paused: this.paused });
      Toast.show(this.paused ? 'Stream paused' : 'Stream resumed', 'info');
    } catch (err) {
      Toast.show(`Pause toggle failed: ${err.message}`, 'error');
    }
  }

  /**
   * Update the pause button label and visual state.
   */
  _updatePauseButton() {
    if (!this.$pauseBtn) return;
    if (this.paused) {
      this.$pauseBtn.textContent = '▶ Resume';
      this.$pauseBtn.classList.add('paused');
    } else {
      this.$pauseBtn.textContent = '⏸ Pause';
      this.$pauseBtn.classList.remove('paused');
    }
  }


  /* -----------------------------------------------------------------------
   *  Video Source Input
   * --------------------------------------------------------------------- */

  /**
   * Bind the source input + connect button.
   */
  _bindSourceInput() {
    if (!this.$sourceBtn || !this.$sourceInput) return;

    this.$sourceBtn.addEventListener('click', () => this._connectSource());
    this.$sourceInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this._connectSource();
    });
  }

  /**
   * Send the user-provided source URL/path to the backend.
   */
  async _connectSource() {
    const source = this.$sourceInput?.value?.trim();
    if (!source) {
      Toast.show('Enter a source URL or camera index', 'warning');
      return;
    }

    this._setSourceStatus('connecting');
    try {
      await this._post('/api/source', { source });
      this._setSourceStatus('connected');
      Toast.show(`Connected to source: ${source}`, 'success');

      // Refresh the video feed
      if (this.$videoFeed) {
        this.$videoFeed.src = '';
        requestAnimationFrame(() => {
          this.$videoFeed.src = '/video_feed';
        });
      }
    } catch (err) {
      this._setSourceStatus('error');
      Toast.show(`Source connection failed: ${err.message}`, 'error');
    }
  }

  /**
   * Update the source status indicator.
   * @param {'connecting'|'connected'|'error'|'idle'} state
   */
  _setSourceStatus(state) {
    if (!this.$sourceStatus) return;
    const map = {
      connecting: { text: 'Connecting…', cls: 'connecting' },
      connected: { text: 'Connected', cls: 'connected' },
      error: { text: 'Error', cls: 'error' },
      idle: { text: '', cls: '' },
    };
    const info = map[state] || map.idle;
    this.$sourceStatus.textContent = info.text;
    this.$sourceStatus.className = `source-status ${info.cls}`;
  }


  /* -----------------------------------------------------------------------
   *  Action Buttons
   * --------------------------------------------------------------------- */

  /**
   * Bind click handlers for reset, snapshot, and export buttons.
   */
  _bindActionButtons() {
    if (this.$resetBtn) {
      this.$resetBtn.addEventListener('click', () => this._resetCounters());
    }
    if (this.$snapshotBtn) {
      this.$snapshotBtn.addEventListener('click', () => this._downloadSnapshot());
    }
    if (this.$exportBtn) {
      this.$exportBtn.addEventListener('click', () => this._exportCSV());
    }
  }

  /**
   * Reset all backend counters and refresh status.
   */
  async _resetCounters() {
    try {
      await this._post('/api/reset_counters', {});
      this._resetChartData();
      Toast.show('Counters reset', 'success');
      await this._pollStatus();
    } catch (err) {
      Toast.show(`Reset failed: ${err.message}`, 'error');
    }
  }

  /**
   * Open the snapshot endpoint in a new tab to download a JPEG.
   */
  _downloadSnapshot() {
    window.open('/api/snapshot', '_blank');
    Toast.show('Snapshot captured', 'info');
  }

  /**
   * Export the current incident events as a CSV file.
   * Attempts to download from the server first, falls back to client-side generation.
   */
  _exportCSV() {
    // Try server-side CSV first
    const serverCsvUrl = '/alerts/incident_audit_log.csv';
    const testLink = document.createElement('a');
    testLink.href = serverCsvUrl;

    // Generate client-side CSV from cached events
    if (this.events.length === 0) {
      Toast.show('No events to export', 'warning');
      return;
    }

    const headers = ['Timestamp', 'Event ID', 'Type', 'Severity', 'Object Class', 'Confidence', 'Zone/Line', 'Dwell (s)', 'Track ID'];
    const rows = this.events.map((ev) => [
      ev.timestamp || '',
      ev.id || '',
      ev.type || '',
      ev.severity || '',
      ev.cls_name || '',
      ev.confidence != null ? parseFloat(ev.confidence).toFixed(2) : '',
      ev.zone_or_line || '',
      ev.dwell_sec != null ? ev.dwell_sec : '',
      ev.track_id || '',
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `incident_audit_log_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}.csv`;
    a.click();
    URL.revokeObjectURL(url);

    Toast.show(`Exported ${this.events.length} events to CSV`, 'success');
  }


  /* -----------------------------------------------------------------------
   *  Incident Log
   * --------------------------------------------------------------------- */

  /**
   * Begin polling the events endpoint.
   */
  _startEventsPolling() {
    this._pollEvents();
    this._eventsInterval = setInterval(() => this._pollEvents(), 2000);
  }

  /**
   * Single events poll iteration.
   */
  async _pollEvents() {
    try {
      const resp = await this._get('/api/events');
      const eventsList = Array.isArray(resp) ? resp : (resp.events || []);
      this.events = eventsList;
      this._renderIncidentTable(eventsList);
    } catch (err) {
      console.warn('[AegisVision] Events poll error:', err);
    }
  }

  /**
   * Render incident events into the table body.
   * New rows slide in with animation.
   * @param {Object[]} events - Array of incident event objects.
   */
  _renderIncidentTable(events) {
    if (!this.$incidentTbody) return;

    // Determine which events are new
    const newEventIds = new Set(events.map((e) => e.id));
    const prevIds = new Set(this.renderedEventIds);

    // Rebuild the table
    this.$incidentTbody.innerHTML = '';

    const display = events.slice(0, this.MAX_INCIDENT_ROWS);

    for (const ev of display) {
      const row = document.createElement('tr');
      const isNew = !prevIds.has(ev.id);

      if (isNew) {
        row.classList.add('incident-row-enter');
        if (this._hasLoadedEvents) {
          if (ev.type === 'CAPACITY_BREACH') {
            Toast.show(`⚠️ CAPACITY BREACH: ${ev.zone_or_line || 'Zone capacity exceeded'}`, 'error', 4500);
          } else if (ev.type === 'LOITERING_DETECTED') {
            Toast.show(`⏱️ LOITERING ALERT: Track #${ev.track_id} in ${ev.zone_or_line || 'Zone'}`, 'warning', 4000);
          } else if (ev.severity === 'CRITICAL') {
            Toast.show(`🚨 ALERT: ${ev.type} in ${ev.zone_or_line || 'Monitored area'}`, 'error', 4000);
          }
        }
      }

      row.innerHTML = `
        <td class="cell-timestamp">${this._formatTimestamp(ev.timestamp)}</td>
        <td class="cell-id" title="${ev.id || ''}">${this._truncateId(ev.id)}</td>
        <td class="cell-type">${ev.type || '—'}</td>
        <td class="cell-severity">${this._severityBadge(ev.severity)}</td>
        <td class="cell-class">${ev.cls_name || '—'}</td>
        <td class="cell-zone">${ev.zone_or_line || '—'}</td>
        <td class="cell-dwell">${ev.dwell_sec != null ? `${parseFloat(ev.dwell_sec).toFixed(1)}s` : '—'}</td>
      `;

      this.$incidentTbody.appendChild(row);
    }

    // Trigger entrance animation for new rows
    if (display.some((e) => !prevIds.has(e.id))) {
      requestAnimationFrame(() => {
        const enterRows = this.$incidentTbody.querySelectorAll('.incident-row-enter');
        enterRows.forEach((r) => {
          r.classList.add('incident-row-active');
        });
      });
    }

    // Update rendered IDs set
    this.renderedEventIds = newEventIds;
    this._hasLoadedEvents = true;
  }

  /**
   * Format a timestamp for table display.
   * @param {string} ts - ISO timestamp string.
   * @returns {string} Formatted time string.
   */
  _formatTimestamp(ts) {
    if (!ts) return '—';
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return ts;
    }
  }

  /**
   * Truncate a long event ID for display.
   * @param {string} id - Full event ID.
   * @returns {string} Truncated ID.
   */
  _truncateId(id) {
    if (!id) return '—';
    return id.length > 10 ? `${id.slice(0, 4)}…${id.slice(-4)}` : id;
  }

  /**
   * Generate a severity badge HTML string.
   * @param {string} severity - 'CRITICAL', 'WARNING', or 'INFO'.
   * @returns {string} Badge HTML.
   */
  _severityBadge(severity) {
    const sev = (severity || 'INFO').toUpperCase();
    const classMap = {
      CRITICAL: 'severity-critical',
      WARNING: 'severity-warning',
      INFO: 'severity-info',
    };
    const cls = classMap[sev] || 'severity-info';
    return `<span class="severity-badge ${cls}">${sev}</span>`;
  }


  /* -----------------------------------------------------------------------
   *  Keyboard Shortcuts
   * --------------------------------------------------------------------- */

  /**
   * Bind global keyboard shortcuts.
   */
  _bindKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Skip if user is typing in an input or textarea
      const tag = e.target.tagName.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

      switch (e.key) {
        case ' ':
          e.preventDefault();
          this._togglePause();
          break;
        case 'r':
        case 'R':
          this._resetCounters();
          break;
        case 's':
        case 'S':
          this._downloadSnapshot();
          break;
        case 'h':
        case 'H':
          this._toggleConfigKey('show_heatmap');
          break;
        case 't':
        case 'T':
          this._toggleConfigKey('show_trails');
          break;
        case '1':
          this._switchModeByIndex(0);
          break;
        case '2':
          this._switchModeByIndex(1);
          break;
        case '3':
          this._switchModeByIndex(2);
          break;
        case '4':
          this._switchModeByIndex(3);
          break;
        default:
          break;
      }
    });
  }

  /**
   * Toggle a boolean config key via keyboard shortcut.
   * @param {string} key - Config key to toggle.
   */
  async _toggleConfigKey(key) {
    this.config[key] = !this.config[key];
    this._savePreferences();
    this._syncToggles();
    try {
      await this._post('/api/config', { [key]: this.config[key] });
      Toast.show(`${key.replace(/_/g, ' ')} ${this.config[key] ? 'ON' : 'OFF'}`, 'info');
    } catch (err) {
      Toast.show(`Config update failed: ${err.message}`, 'error');
    }
  }

  /**
   * Switch mode by numeric index (for 1–4 keyboard shortcuts).
   * @param {number} index - Zero-based index into mode buttons.
   */
  _switchModeByIndex(index) {
    if (!this.$modeBar) return;
    const buttons = this.$modeBar.querySelectorAll('.mode-btn');
    if (buttons[index]) {
      const mode = buttons[index].dataset.mode;
      this._switchMode(mode);
    }
  }
}


/* =========================================================================
 *  Bootstrap
 * ========================================================================= */

document.addEventListener('DOMContentLoaded', () => {
  const app = new AegisVisionApp();
  app.init();

  // Expose for console debugging
  window.__aegis = app;
});
