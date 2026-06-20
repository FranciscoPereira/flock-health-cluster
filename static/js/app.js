/* ============================================================
   NHS Federated Disease Detection — Google Maps + API integration
   ============================================================ */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let map, heatmap;
const markers   = {};   // { node_key: google.maps.Marker }
const infoWins  = {};   // { node_key: google.maps.InfoWindow }
const charts    = {};   // { node_key: Chart }
let pipelineRunning = false;
let pollTimer       = null;

// NHS region polygons (simplified lat/lng paths)
const REGION_PATHS = {
  node_a: [                          // Wales
    [53.4, -3.35], [53.4, -2.65], [52.35, -2.65], [51.35, -2.65],
    [51.35, -5.35], [52.5, -5.35], [53.4, -3.35],
  ],
  node_b: [                          // South-East England (simplified)
    [53.5, -1.5], [53.5, 1.8], [50.9, 1.8], [50.9, -1.5], [53.5, -1.5],
  ],
  node_c: [                          // Scotland
    [60.9, -1.7], [60.9, -6.3], [54.6, -6.3], [54.6, -1.7], [60.9, -1.7],
  ],
};

const REGION_COLORS = {
  normal: { fill: '#00e676', stroke: '#00e676' },
  alert:  { fill: '#ff1744', stroke: '#ff1744' },
};

// ── Google Maps callback ───────────────────────────────────────────────────
window.initMap = function () {
  map = new google.maps.Map(document.getElementById('map'), {
    center: { lat: 54.5, lng: -3.5 },
    zoom: 5,
    mapTypeId: 'roadmap',
    styles: DARK_MAP_STYLE,
    disableDefaultUI: false,
    zoomControl: true,
    mapTypeControl: false,
    streetViewControl: false,
  });

  // Draw region polygons + markers with default (normal) colours
  const defaultNodes = [
    { key: 'node_a', label: 'Node A – Wales',    lat: 51.4816, lng: -3.1791 },
    { key: 'node_b', label: 'Node B – England',  lat: 51.5074, lng: -0.1278 },
    { key: 'node_c', label: 'Node C – Scotland', lat: 55.9533, lng: -3.1883 },
  ];

  defaultNodes.forEach(n => {
    drawRegion(n.key, false);
    addMarker(n);
  });

  // Poll every 4 s unconditionally so state stays in sync
  fetchStatus();
  pollTimer = setInterval(fetchStatus, 4000);
};

// ── Google Maps helpers ────────────────────────────────────────────────────
function drawRegion(key, isAlert) {
  // Remove previous polygon if any
  if (window[`__poly_${key}`]) window[`__poly_${key}`].setMap(null);

  const col = isAlert ? REGION_COLORS.alert : REGION_COLORS.normal;
  const path = REGION_PATHS[key].map(([lat, lng]) => ({ lat, lng }));

  const poly = new google.maps.Polygon({
    paths: path,
    strokeColor:   col.stroke,
    strokeOpacity: 0.6,
    strokeWeight:  1.5,
    fillColor:     col.fill,
    fillOpacity:   isAlert ? 0.18 : 0.07,
    map,
  });

  window[`__poly_${key}`] = poly;
}

function addMarker({ key, label, lat, lng }) {
  const isAlert = false;
  const icon = makeIcon(isAlert);

  const marker = new google.maps.Marker({
    position: { lat, lng },
    map,
    title: label,
    icon,
    animation: google.maps.Animation.DROP,
  });

  const iw = new google.maps.InfoWindow({ content: buildInfoContent(key, {}) });
  marker.addListener('click', () => {
    Object.values(infoWins).forEach(w => w.close());
    iw.open(map, marker);
  });

  markers[key]  = marker;
  infoWins[key] = iw;
}

function makeIcon(isAlert) {
  return {
    path: google.maps.SymbolPath.CIRCLE,
    scale:       isAlert ? 16 : 12,
    fillColor:   isAlert ? '#ff1744' : '#00e676',
    fillOpacity: 1,
    strokeColor: '#0b0f1a',
    strokeWeight: 2,
  };
}

function updateMarker(key, node) {
  const marker = markers[key];
  if (!marker) return;
  marker.setIcon(makeIcon(node.is_alert));
  if (node.is_alert) marker.setAnimation(google.maps.Animation.BOUNCE);
  else               marker.setAnimation(null);

  infoWins[key].setContent(buildInfoContent(key, node));
  drawRegion(key, node.is_alert);
}

function buildInfoContent(key, node) {
  const status    = node.is_alert ? '🔴 OUTBREAK ALERT' : '🟢 NORMAL';
  const count     = node.anomaly_count ?? '—';
  const threshold = node.threshold    ?? '—';

  return `
    <div style="font-family:'Courier New',monospace;background:#111827;color:#c9d6e8;
                padding:12px 16px;border-radius:6px;min-width:200px;font-size:12px">
      <div style="font-size:14px;font-weight:bold;color:#00d4ff;margin-bottom:8px">
        ${node.label ?? key}
      </div>
      <div style="margin-bottom:4px">Status: <strong>${status}</strong></div>
      <div style="margin-bottom:4px">Anomalies: <strong>${count}</strong></div>
      <div style="margin-bottom:4px">Threshold: <strong>${threshold}</strong></div>
      <div style="font-size:10px;color:#4a5568;margin-top:8px">
        Click elsewhere to close
      </div>
    </div>`;
}

// ── Chart helpers ──────────────────────────────────────────────────────────
function initChart(key) {
  const canvas = document.getElementById(`chart-${key}`);
  if (!canvas || charts[key]) return;

  charts[key] = new Chart(canvas, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Reconstruction Error',
          data:  [],
          borderColor: '#00d4ff',
          backgroundColor: 'rgba(0,212,255,0.08)',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          fill: true,
        },
        {
          label: 'Threshold',
          data:  [],
          borderColor: '#ff1744',
          borderWidth: 1,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 600 },
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: '#4a5568', maxTicksLimit: 8, font: { size: 9 } },
          grid:  { color: '#1e2d45' },
        },
        y: {
          ticks: { color: '#4a5568', font: { size: 9 } },
          grid:  { color: '#1e2d45' },
        },
      },
    },
  });
}

function updateChart(key, node) {
  initChart(key);
  const chart = charts[key];
  if (!node.time_series?.length) return;

  chart.data.labels = node.time_series.map(p => `D${p.day}`);
  chart.data.datasets[0].data = node.time_series.map(p => p.error);
  chart.data.datasets[1].data = node.time_series.map(() => node.threshold);
  chart.update();
}

// ── API calls ──────────────────────────────────────────────────────────────
async function fetchStatus() {
  try {
    const [statusRes, alertsRes] = await Promise.all([
      fetch('/api/status'),
      fetch('/api/alerts'),
    ]);
    const data   = await statusRes.json();
    const alerts = await alertsRes.json();
    applyStatus(data);
    renderAlertLog(alerts);
    document.getElementById('alert-count').textContent = alerts.length;
  } catch (e) {
    console.error('Status poll failed:', e);
  }
}

async function runPipeline() {
  if (pipelineRunning) return;
  pipelineRunning = true;

  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = '⏳ RUNNING…';

  showProgress(0, 'Generating synthetic data…');
  setStep(0, 'active');

  try {
    const res  = await fetch('/api/run-pipeline', { method: 'POST' });
    const data = await res.json();

    if (data.error) {
      showProgress(100, `Error: ${data.error}`);
    } else {
      showProgress(100, '✓ Pipeline complete');
      setStep(0, 'done'); setStep(1, 'done'); setStep(2, 'done');
      await fetchStatus();
      await fetchAlerts();
    }
  } catch (e) {
    showProgress(100, 'Network error — see console');
    console.error(e);
  }

  pipelineRunning = false;
  btn.disabled = false;
  btn.textContent = '▶  RUN FEDERATED PIPELINE';
}

async function fetchAlerts() {
  const res    = await fetch('/api/alerts');
  const alerts = await res.json();
  renderAlertLog(alerts);
}

// ── DOM helpers ────────────────────────────────────────────────────────────
function applyStatus(data) {
  document.getElementById('sys-status').textContent  = data.pipeline_status;
  document.getElementById('last-run').textContent    =
    data.trained_at ? new Date(data.trained_at).toLocaleTimeString() : '—';
  document.getElementById('alert-count').textContent = data.alert_count ?? 0;

  const pulse = document.getElementById('pulse-dot');
  pulse.style.background = data.alert_count > 0 ? '#ff1744' : '#00e676';

  (data.nodes ?? []).forEach(node => {
    const key = node.key;

    // Sidebar status row
    const row  = document.getElementById(`status-${key}`);
    if (row) {
      row.querySelector('.status-dot').className =
        `status-dot ${node.is_alert ? 'red' : 'green'}`;

      const badge = row.querySelector('.badge');
      if (node.anomaly_count !== undefined) {
        badge.textContent = node.is_alert
          ? `⚠ ${node.anomaly_count} ALERTS`
          : '✓ NORMAL';
        badge.className = `badge ${node.is_alert ? 'alert' : 'normal'}`;
      }
    }

    // Chart sub-label
    const sub = document.getElementById(`chart-sub-${key}`);
    if (sub && node.threshold) {
      sub.textContent = node.is_alert
        ? `🔴 ALERT — ${node.anomaly_count} anomalies | threshold: ${node.threshold}`
        : `🟢 Normal — threshold: ${node.threshold}`;
      sub.style.color = node.is_alert ? '#ff1744' : '#00e676';
    }

    // Map + chart (only if data available)
    if (node.time_series?.length) {
      if (window.GOOGLE_MAPS_KEY) updateMarker(key, node);
      else                        updateLeafletMarker(key, node);
      updateChart(key, node);
    }
  });

  // Progress simulation while running
  if (data.pipeline_status === 'RUNNING') {
    const pct = Math.min(80, (Date.now() % 80000) / 1000);
    showProgress(pct, stepsFor(Math.floor(pct / 27)));
  }
}

function stepsFor(idx) {
  return ['Generating data…', 'Federated training…', 'Detecting anomalies…'][idx] ?? 'Finishing…';
}

function renderAlertLog(alerts) {
  const log = document.getElementById('alert-log');
  if (!alerts.length) { log.innerHTML = '<p class="dim">No alerts detected.</p>'; return; }

  log.innerHTML = alerts.map(a => `
    <div class="alert-entry">
      <div class="a-head">⚠ ${a.node?.toUpperCase()} — Day ${a.day} — ${a.severity}</div>
      <div class="a-body">Error: ${a.error?.toFixed(4)} | Z-score: ${a.z_score?.toFixed(2)}</div>
      <div class="a-time">${new Date(a.timestamp).toLocaleTimeString()}</div>
    </div>`).join('');
}

const STEPS = ['1. Generate data', '2. Federated training', '3. Anomaly detection'];

function ensureSteps() {
  const list = document.getElementById('step-list');
  if (list.children.length) return;
  STEPS.forEach((s, i) => {
    const el = document.createElement('div');
    el.className = 'step-item'; el.id = `step-${i}`; el.textContent = s;
    list.appendChild(el);
  });
}

function setStep(i, state) {
  ensureSteps();
  const el = document.getElementById(`step-${i}`);
  if (el) el.className = `step-item ${state}`;
}

function showProgress(pct, label) {
  ensureSteps();
  const wrap = document.getElementById('progress-wrap');
  wrap.style.display = 'block';
  document.getElementById('progress-bar').style.width   = `${pct}%`;
  document.getElementById('progress-label').textContent = label;

  const stepIdx = pct < 33 ? 0 : pct < 66 ? 1 : 2;
  STEPS.forEach((_, i) => {
    if (i < stepIdx) setStep(i, 'done');
    else if (i === stepIdx) setStep(i, 'active');
  });
}

// ── Leaflet fallback map (no Google Maps API key) ─────────────────────────
let leafletMap = null;
const leafletMarkers = {};
const leafletLayers  = {};

const NODE_LEAFLET = [
  { key: 'node_a', label: 'Node A – Wales',    lat: 51.4816, lng: -3.1791 },
  { key: 'node_b', label: 'Node B – England',  lat: 51.5074, lng: -0.1278 },
  { key: 'node_c', label: 'Node C – Scotland', lat: 55.9533, lng: -3.1883 },
];

// Approximate GeoJSON-style bounding boxes per region (as L.Rectangles)
const NODE_BOUNDS = {
  node_a: [[51.35, -5.35], [53.40, -2.65]],
  node_b: [[50.90, -1.50], [53.50,  1.80]],
  node_c: [[54.60, -6.30], [60.90, -1.70]],
};

function leafletIcon(isAlert) {
  const color = isAlert ? '#ff1744' : '#00e676';
  const glow  = isAlert ? '#ff1744' : '#00e676';
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28">
    <circle cx="14" cy="14" r="11" fill="${color}" fill-opacity="0.9" stroke="#0b0f1a" stroke-width="2"/>
    <circle cx="14" cy="14" r="5"  fill="#0b0f1a"/>
  </svg>`;
  return L.divIcon({
    html: svg,
    className: '',
    iconSize: [28, 28],
    iconAnchor: [14, 14],
    popupAnchor: [0, -16],
  });
}

function leafletPopupContent(key, node) {
  const status = node.is_alert ? '🔴 OUTBREAK ALERT' : '🟢 NORMAL';
  const count  = node.anomaly_count ?? '—';
  const thr    = node.threshold    ?? '—';
  return `<div style="font-family:'Courier New',monospace;font-size:12px;min-width:180px">
    <b style="color:#00d4ff">${node.label ?? key}</b><br/>
    Status: <b>${status}</b><br/>
    Anomalies: <b>${count}</b><br/>
    Threshold: <b>${thr}</b>
  </div>`;
}

window.initLeafletMap = function () {
  leafletMap = L.map('map', { zoomControl: true }).setView([54.5, -3.5], 5);

  // Dark tile layer (CartoDB Dark Matter — no key needed)
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap contributors © CARTO',
    subdomains: 'abcd', maxZoom: 19,
  }).addTo(leafletMap);

  NODE_LEAFLET.forEach(n => {
    const rect = L.rectangle(NODE_BOUNDS[n.key], {
      color: '#00e676', weight: 1.5, fillColor: '#00e676', fillOpacity: 0.07,
    }).addTo(leafletMap);
    leafletLayers[n.key] = rect;

    const m = L.marker([n.lat, n.lng], { icon: leafletIcon(false) })
      .addTo(leafletMap)
      .bindPopup(leafletPopupContent(n.key, {}));
    leafletMarkers[n.key] = m;
  });

  // Always poll so the page reflects backend state immediately
  fetchStatus();
  setInterval(fetchStatus, 4000);
};

// Called from applyStatus when Leaflet is active
function updateLeafletMarker(key, node) {
  const m = leafletMarkers[key];
  if (!m) return;
  m.setIcon(leafletIcon(node.is_alert));
  m.setPopupContent(leafletPopupContent(key, node));

  const rect = leafletLayers[key];
  if (rect) {
    const col = node.is_alert ? '#ff1744' : '#00e676';
    rect.setStyle({ color: col, fillColor: col,
      fillOpacity: node.is_alert ? 0.18 : 0.07 });
  }
}

// ── Dark map style ─────────────────────────────────────────────────────────
const DARK_MAP_STYLE = [
  { elementType: 'geometry',          stylers: [{ color: '#0b0f1a' }] },
  { elementType: 'labels.text.fill',  stylers: [{ color: '#4a5568' }] },
  { elementType: 'labels.text.stroke',stylers: [{ color: '#0b0f1a' }] },
  { featureType: 'administrative',    elementType: 'geometry',
    stylers: [{ color: '#1e2d45' }] },
  { featureType: 'administrative.country', elementType: 'geometry.stroke',
    stylers: [{ color: '#2d4a6b' }] },
  { featureType: 'road',              elementType: 'geometry',
    stylers: [{ color: '#1a2744' }] },
  { featureType: 'road.highway',      elementType: 'geometry',
    stylers: [{ color: '#1e3a5f' }] },
  { featureType: 'water',             elementType: 'geometry',
    stylers: [{ color: '#060d1a' }] },
  { featureType: 'water',             elementType: 'labels.text.fill',
    stylers: [{ color: '#1e2d45' }] },
  { featureType: 'landscape',         elementType: 'geometry',
    stylers: [{ color: '#0d1420' }] },
  { featureType: 'poi',               elementType: 'geometry',
    stylers: [{ color: '#111827' }] },
  { featureType: 'poi',               elementType: 'labels.text.fill',
    stylers: [{ color: '#4a5568' }] },
  { featureType: 'transit',           elementType: 'geometry',
    stylers: [{ color: '#111827' }] },
];
