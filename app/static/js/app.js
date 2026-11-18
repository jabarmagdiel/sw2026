// ============================================================================
// Arquitectura Frontend Modular — Módulo 3: Reconocimiento Facial Multirrostro y Video
// ============================================================================

const AppState = {
  currentView: 'monitor',
  stream: null,
  videoObjectUrl: null,
  sourceMode: null, // 'webcam' | 'video'
  liveInterval: null,
  isScanning: false,
  capturedBlobForBlacklist: null,
  soundEnabled: true,
  lastBeepTime: 0,
  vetados: [],
  alertas: [],
};

// ----------------------------------------------------------------------------
// 1. NAVEGACIÓN ENTRE MÓDULOS (SPA ROUTER)
// ----------------------------------------------------------------------------
const VIEW_META = {
  monitor: {
    title: '🎥 Monitor de Acceso y Video Multirrostro',
    subtitle: 'Detección e identificación simultánea de múltiples rostros en cámara en vivo o videos pregrabados',
  },
  blacklist: {
    title: '🚫 Gestión de Personas Vetadas (Lista Negra)',
    subtitle: 'Administración de antecedentes, enrolamiento biométrico ArcFace (512-D) e indexación en FAISS',
  },
  alerts: {
    title: '🚨 Centro de Alertas e Incidentes de Seguridad',
    subtitle: 'Historial de coincidencias biométricas detectadas en puertas de acceso y auditoría de operadores',
  },
  verify: {
    title: '🎬 Laboratorio de Análisis Multirrostro (Fotos y Videos)',
    subtitle: 'Inspección simultánea de múltiples rostros en fotografías grupales y procesamiento forense de archivos de video',
  },
};

function switchView(viewId) {
  AppState.currentView = viewId;
  document.querySelectorAll('.module-view').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

  const targetSection = document.getElementById(`view-${viewId}`);
  const targetBtn = document.querySelector(`.nav-btn[data-view="${viewId}"]`);
  if (targetSection) targetSection.classList.add('active');
  if (targetBtn) targetBtn.classList.add('active');

  const meta = VIEW_META[viewId];
  if (meta) {
    document.getElementById('topbar-title').textContent = meta.title;
    document.getElementById('topbar-subtitle').textContent = meta.subtitle;
  }
  refreshAllData();
}

// ----------------------------------------------------------------------------
// 2. SONIDO DE ALERTA Y MODAL DE EVIDENCIA
// ----------------------------------------------------------------------------
function playAlertBeep() {
  if (!AppState.soundEnabled) return;
  const now = Date.now();
  if (now - AppState.lastBeepTime < 2500) return;
  AppState.lastBeepTime = now;
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, audioCtx.currentTime);
    osc.frequency.setValueAtTime(640, audioCtx.currentTime + 0.12);
    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.28);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.28);
  } catch (_) {}
}

function toggleSound() {
  AppState.soundEnabled = !AppState.soundEnabled;
  const btn = document.getElementById('btn-sound');
  btn.textContent = AppState.soundEnabled ? '🔊 Alarma: ON' : '🔇 Alarma: OFF';
}

function openImageModal(imgUrl, title, subtitle) {
  document.getElementById('modal-img').src = imgUrl;
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-sub').textContent = subtitle || '';
  document.getElementById('modal-evidence').style.display = 'flex';
}

function closeImageModal() {
  document.getElementById('modal-evidence').style.display = 'none';
}

// ----------------------------------------------------------------------------
// 3. MÓDULO 1: CÁMARA EN VIVO, VIDEO EN MONITOR Y DETECCIÓN MULTIRROSTRO
// ----------------------------------------------------------------------------
function detenerFuenteActual() {
  const video = document.getElementById('webcam');
  if (AppState.stream) {
    AppState.stream.getTracks().forEach(t => t.stop());
    AppState.stream = null;
  }
  if (AppState.videoObjectUrl) {
    URL.revokeObjectURL(AppState.videoObjectUrl);
    AppState.videoObjectUrl = null;
  }
  video.srcObject = null;
  video.removeAttribute('src');
}

async function iniciarCamara() {
  const video = document.getElementById('webcam');
  const overlay = document.getElementById('overlay');
  try {
    detenerFuenteActual();
    AppState.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: 'user' },
      audio: false,
    });
    AppState.sourceMode = 'webcam';
    video.loop = false;
    video.srcObject = AppState.stream;
    await video.play();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;

    document.getElementById('video-player-controls').style.display = 'none';
    document.getElementById('cam-status').textContent = 'CÁMARA WEB ACTIVA';
    document.getElementById('rec-dot').style.background = '#10b981';
    document.getElementById('btn-auto-scan').disabled = false;
    document.getElementById('btn-snap-vetar').disabled = false;
    document.getElementById('btn-start-cam').textContent = '🔄 Reiniciar Cámara';
  } catch (err) {
    alert('No se pudo acceder a la cámara web: ' + err.message);
  }
}

async function cargarVideoEnMonitor(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  const video = document.getElementById('webcam');
  const overlay = document.getElementById('overlay');

  detenerFuenteActual();
  AppState.sourceMode = 'video';
  AppState.videoObjectUrl = URL.createObjectURL(file);

  video.src = AppState.videoObjectUrl;
  video.loop = true;
  video.muted = true;
  video.playsInline = true;

  video.onloadedmetadata = async () => {
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    try {
      await video.play();
    } catch (_) {}

    document.getElementById('video-player-controls').style.display = 'flex';
    document.getElementById('monitor-video-name').textContent = `${file.name} (${video.videoWidth}x${video.videoHeight})`;
    document.getElementById('btn-video-playpause').textContent = '⏸️ Pausar Video';
    document.getElementById('cam-status').textContent = 'VIDEO CCTV EN REPRODUCCIÓN';
    document.getElementById('rec-dot').style.background = '#3b82f6';
    document.getElementById('btn-auto-scan').disabled = false;
    document.getElementById('btn-snap-vetar').disabled = false;

    // Iniciar detección multirrostro automáticamente al cargar el video
    if (!AppState.liveInterval) {
      toggleEscaneoEnVivo();
    }
  };
}

function togglePlayPauseVideoMonitor() {
  const video = document.getElementById('webcam');
  const btn = document.getElementById('btn-video-playpause');
  if (!video || AppState.sourceMode !== 'video') return;
  if (video.paused) {
    video.play();
    btn.textContent = '⏸️ Pausar Video';
  } else {
    video.pause();
    btn.textContent = '▶️ Reanudar Video';
    // Escanear el fotograma pausado actual
    escanearFrameActual();
  }
}

function reiniciarVideoMonitor() {
  const video = document.getElementById('webcam');
  if (!video || AppState.sourceMode !== 'video') return;
  video.currentTime = 0;
  video.play();
  document.getElementById('btn-video-playpause').textContent = '⏸️ Pausar Video';
}

function tieneFuenteActiva() {
  return Boolean(AppState.stream || AppState.videoObjectUrl);
}

function capturarFrameCanvas() {
  const video = document.getElementById('webcam');
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = video.videoWidth || 640;
  tempCanvas.height = video.videoHeight || 480;
  const tctx = tempCanvas.getContext('2d');
  tctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
  return tempCanvas;
}

function capturarYEnviarAModuloVetados() {
  if (!tieneFuenteActiva()) return;
  const tempCanvas = capturarFrameCanvas();
  tempCanvas.toBlob((blob) => {
    enviarBlobAModuloVetados(blob);
  }, 'image/jpeg', 0.92);
}

function enviarBlobAModuloVetados(blob) {
  AppState.capturedBlobForBlacklist = blob;
  const url = URL.createObjectURL(blob);
  document.getElementById('img-captura-vetar').src = url;
  document.getElementById('preview-captura').style.display = 'flex';
  document.getElementById('input-foto-vetar').value = '';
  switchView('blacklist');
  document.getElementById('input-cedula-vetar').focus();
}

async function vetarRecorteBase64(dataUrl) {
  if (!dataUrl) return;
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  enviarBlobAModuloVetados(blob);
}

function limpiarCaptura() {
  AppState.capturedBlobForBlacklist = null;
  document.getElementById('preview-captura').style.display = 'none';
}

function toggleEscaneoEnVivo() {
  const btn = document.getElementById('btn-auto-scan');
  const overlay = document.getElementById('overlay');
  const ctx = overlay.getContext('2d');
  const camBox = document.getElementById('cam-box');

  if (AppState.liveInterval) {
    clearInterval(AppState.liveInterval);
    AppState.liveInterval = null;
    btn.textContent = '🔍 Iniciar Detección Multirrostro';
    btn.className = 'btn btn-emerald btn-sm';
    document.getElementById('cam-status').textContent = 'ESCANEADO PAUSADO';
    document.getElementById('rec-dot').className = 'rec-dot';
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    camBox.className = 'cam-stage';
  } else {
    btn.textContent = '⏹️ Detener Detección';
    btn.className = 'btn btn-danger btn-sm';
    document.getElementById('cam-status').textContent =
      AppState.sourceMode === 'video' ? 'ANALIZANDO VIDEO EN VIVO (IA)' : 'MONITOREANDO MULTIRROSTRO';
    document.getElementById('rec-dot').className = 'rec-dot live';
    escanearFrameActual();
    AppState.liveInterval = setInterval(escanearFrameActual, 950);
  }
}

function dibujarHUD(ctx, x1, y1, x2, y2, color, title, subtitle) {
  const w = x2 - x1;
  const h = y2 - y1;
  const corner = Math.min(22, w * 0.25, h * 0.25);

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, w, h);

  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(x1, y1 + corner); ctx.lineTo(x1, y1); ctx.lineTo(x1 + corner, y1);
  ctx.moveTo(x2 - corner, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + corner);
  ctx.moveTo(x1, y2 - corner); ctx.lineTo(x1, y2); ctx.lineTo(x1 + corner, y2);
  ctx.moveTo(x2 - corner, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - corner);
  ctx.stroke();

  ctx.font = '700 13px Inter, sans-serif';
  const tWidth = ctx.measureText(title).width;
  ctx.font = '500 11px JetBrains Mono, monospace';
  const sWidth = subtitle ? ctx.measureText(subtitle).width : 0;
  const textW = Math.max(tWidth, sWidth) + 18;
  const boxH = subtitle ? 40 : 24;
  const boxY = Math.max(y1 - boxH - 4, 4);

  ctx.fillStyle = color;
  ctx.fillRect(x1, boxY, textW, boxH);
  ctx.fillStyle = '#ffffff';
  ctx.font = '700 12px Inter, sans-serif';
  ctx.fillText(title, x1 + 8, boxY + 16);
  if (subtitle) {
    ctx.font = '500 11px JetBrains Mono, monospace';
    ctx.fillText(subtitle, x1 + 8, boxY + 32);
  }
}

function renderMultifaceCards(matches, noMatches) {
  const cards = [];

  matches.forEach((item) => {
    const m = item.match;
    const num = item.face_index + 1;
    const thumb = item.face_crop_base64 || (m.foto_path ? '/' + m.foto_path.replace(/^\.\//, '') : '');
    cards.push(`
      <div class="face-card is-vetado">
        ${thumb ? `<img src="${thumb}" alt="Rostro #${num}" onclick="openImageModal('${thumb}', 'Rostro #${num} — VETADO: ${m.nombre} ${m.apellido}', 'CI: ${m.cedula} • Similitud: ${(m.similitud * 100).toFixed(1)}%')" />` : ''}
        <div class="face-card-info">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:4px;">
            <span class="face-card-title" style="color:#fca5a5;">🚨 #${num} ${m.nombre} ${m.apellido}</span>
            <span class="risk-badge risk-${m.nivel_riesgo}" style="font-size:9px;padding:1px 6px;">${m.nivel_riesgo}</span>
          </div>
          <div class="face-card-sub">CI: <b>${m.cedula}</b> • Similitud: <b>${(m.similitud * 100).toFixed(1)}%</b></div>
          <div class="face-card-sub" style="color:#fda4af;">Motivo: ${m.motivo_veto}</div>
        </div>
      </div>
    `);
  });

  noMatches.forEach((item) => {
    const num = item.face_index + 1;
    const thumb = item.face_crop_base64 || '';
    const bestSim = item.best_score !== null && item.best_score !== undefined
      ? `Máx sim: ${(item.best_score * 100).toFixed(1)}%`
      : `Conf: ${(item.det_score * 100).toFixed(0)}%`;
    cards.push(`
      <div class="face-card">
        ${thumb ? `<img src="${thumb}" alt="Rostro #${num}" onclick="openImageModal('${thumb}', 'Rostro #${num} — Acceso Autorizado', 'Sin coincidencias en FAISS (${bestSim})')" />` : ''}
        <div class="face-card-info">
          <div class="face-card-title" style="color:#6ee7b7;">✅ #${num} Autorizado</div>
          <div class="face-card-sub">Sin antecedentes (${bestSim})</div>
          ${thumb ? `<button type="button" class="btn btn-slate btn-sm" style="margin-top:5px;padding:3px 8px;font-size:10px;" onclick="vetarRecorteBase64('${thumb}')">🚫 Vetar este rostro</button>` : ''}
        </div>
      </div>
    `);
  });

  return cards.join('');
}

async function escanearFrameActual() {
  if (!tieneFuenteActiva() || AppState.isScanning) return;
  const video = document.getElementById('webcam');
  if (video.readyState < 2) return;

  AppState.isScanning = true;
  const overlay = document.getElementById('overlay');
  const ctx = overlay.getContext('2d');
  const camBox = document.getElementById('cam-box');

  try {
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    const tempCanvas = capturarFrameCanvas();
    const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.84);

    const res = await fetch('/api/v1/recognize/base64', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: dataUrl,
        camara_id: document.getElementById('live-camara-id').value || 'CAM-ACCESO-01',
        zona: document.getElementById('live-zona').value || 'Puerta Principal',
      }),
    });

    const data = await res.json();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!res.ok) return;

    document.getElementById('kpi-latency').textContent = `${Math.round(data.processing_time_ms)} ms`;
    document.getElementById('kpi-faces-now').textContent = data.faces_detected;
    document.getElementById('hud-faces-count').textContent = `👥 ${data.faces_detected} ROSTRO(S)`;
    document.getElementById('live-faces-badge').textContent = `${data.faces_detected} en escena`;

    // Dibujar HUD para TODOS los rostros vetados detectados simultáneamente
    data.matches.forEach(item => {
      const [x1, y1, x2, y2] = item.bbox;
      const m = item.match;
      const num = item.face_index + 1;
      dibujarHUD(
        ctx, x1, y1, x2, y2,
        'rgba(239, 68, 68, 0.92)',
        `🚨 #${num} VETADO: ${m.nombre.toUpperCase()} ${m.apellido.toUpperCase()}`,
        `CI: ${m.cedula} | SIM: ${(m.similitud * 100).toFixed(1)}%`
      );
    });

    // Dibujar HUD para TODOS los rostros autorizados detectados simultáneamente
    data.no_matches.forEach(item => {
      const [x1, y1, x2, y2] = item.bbox;
      const num = item.face_index + 1;
      dibujarHUD(
        ctx, x1, y1, x2, y2,
        'rgba(16, 185, 129, 0.88)',
        `✅ #${num} ACCESO AUTORIZADO`,
        `CONF: ${(item.det_score * 100).toFixed(0)}% • SIN VETO`
      );
    });

    const liveBox = document.getElementById('live-detection-card');
    const facesGrid = document.getElementById('live-faces-grid');

    if (data.alert_triggered) {
      playAlertBeep();
      camBox.className = 'cam-stage state-danger';
      liveBox.className = 'banner danger';

      const vetadosListHtml = data.matches.map(item => {
        const m = item.match;
        return `
          <div style="padding:8px 10px;margin-top:6px;background:rgba(0,0,0,0.28);border-radius:8px;border-left:3px solid #ef4444;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <strong>🚨 Rostro #${item.face_index + 1}: ${m.nombre} ${m.apellido} (CI: <code>${m.cedula}</code>)</strong>
              <span class="risk-badge risk-${m.nivel_riesgo}">${m.nivel_riesgo}</span>
            </div>
            <div style="font-size:12px;margin-top:3px;">
              Motivo: <b>${m.motivo_veto}</b> — Similitud ArcFace: <b>${(m.similitud * 100).toFixed(1)}%</b>
            </div>
          </div>`;
      }).join('');

      liveBox.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <strong>🚨 ¡ALERTA MULTIRROSTRO! ${data.matches.length} VETADO(S) EN ESCENA</strong>
          <span style="font-size:11px;font-family:'JetBrains Mono',monospace;">Total rostros: ${data.faces_detected} (${data.processing_time_ms} ms)</span>
        </div>
        ${vetadosListHtml}`;

      facesGrid.innerHTML = renderMultifaceCards(data.matches, data.no_matches);
      refreshAllData();
    } else if (data.faces_detected > 0) {
      camBox.className = 'cam-stage state-safe';
      liveBox.className = 'banner safe';
      liveBox.innerHTML = `✅ <strong>Análisis Multirrostro Activo:</strong> Se verificaron simultáneamente <b>${data.faces_detected}</b> rostro(s) en <b>${data.processing_time_ms} ms</b>. Ninguno registra prohibición de ingreso.`;
      facesGrid.innerHTML = renderMultifaceCards([], data.no_matches);
    } else {
      camBox.className = 'cam-stage';
      liveBox.className = 'banner';
      liveBox.innerHTML = `👁️ Monitoreando escena... Sin rostros frente al lente (${data.processing_time_ms} ms).`;
      facesGrid.innerHTML = '';
    }
  } catch (err) {
    console.error(err);
  } finally {
    AppState.isScanning = false;
  }
}

// ----------------------------------------------------------------------------
// 4. SINCRONIZACIÓN DE DATOS Y RENDERIZADO DE TABLAS POR MÓDULO
// ----------------------------------------------------------------------------
async function refreshAllData() {
  try {
    const [st, vetados, alertas] = await Promise.all([
      fetch('/api/v1/stats').then(r => r.json()),
      fetch('/api/v1/vetados').then(r => r.json()),
      fetch('/api/v1/alertas?limit=30').then(r => r.json()),
    ]);

    AppState.vetados = vetados;
    AppState.alertas = alertas;

    document.getElementById('kpi-vetados').textContent = st.vetados_activos;
    document.getElementById('kpi-faiss').textContent = st.vectores_en_faiss;
    document.getElementById('kpi-alertas').textContent = st.alertas_pendientes;
    document.getElementById('nav-count-vetados').textContent = st.vetados_activos;
    document.getElementById('nav-count-alertas').textContent = st.alertas_pendientes;
    document.getElementById('engine-name').textContent = st.motor_inferencia;

    renderRecentAlertsFeed();
    renderBlacklistTable();
    renderAlertsTable();
  } catch (e) {
    console.error('Error cargando datos:', e);
  }
}

function renderRecentAlertsFeed() {
  const container = document.getElementById('monitor-recent-alerts');
  const recent = AppState.alertas.slice(0, 5);
  if (!recent.length) {
    container.innerHTML = `<p style="color:var(--text-dim);font-size:13px;">No se han registrado alertas recientes en este acceso.</p>`;
    return;
  }
  container.innerHTML = recent.map(a => {
    const p = a.persona_vetada || {};
    const capUrl = a.screenshot_path ? '/' + a.screenshot_path.replace(/^\.\//, '') : '';
    const hora = new Date(a.created_at).toLocaleTimeString();
    return `
      <div style="display:flex;align-items:center;justify-content:space-between;padding:10px;border-radius:10px;background:var(--bg-elevated);margin-bottom:8px;border-left:3px solid ${a.revisada ? '#10b981' : '#ef4444'};">
        <div style="display:flex;align-items:center;gap:10px;">
          ${capUrl ? `<img class="avatar-thumb" src="${capUrl}" onclick="openImageModal('${capUrl}', 'Alerta #${a.id} — ${p.nombre || ''} ${p.apellido || ''}', '${a.camara_id} • ${hora}')" />` : ''}
          <div>
            <div style="font-size:13px;font-weight:700;">${p.nombre || 'ID ' + a.persona_vetada_id} ${p.apellido || ''} <span style="color:#f87171;">(${(a.similitud * 100).toFixed(0)}%)</span></div>
            <div style="font-size:11px;color:var(--text-muted);">${a.camara_id} • ${hora}</div>
          </div>
        </div>
        <button class="btn btn-slate btn-sm" onclick="switchView('alerts')">Ver</button>
      </div>`;
  }).join('');
}

function renderBlacklistTable() {
  const q = (document.getElementById('search-blacklist')?.value || '').toLowerCase().trim();
  const tbody = document.getElementById('tbody-blacklist');
  const filtered = AppState.vetados.filter(v =>
    !q || `${v.nombre} ${v.apellido} ${v.cedula} ${v.motivo_veto}`.toLowerCase().includes(q)
  );

  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-dim);padding:28px;">No hay personas vetadas registradas en el padrón.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(v => {
    const imgUrl = v.foto_path ? '/' + v.foto_path.replace(/^\.\//, '') : '';
    return `
      <tr>
        <td>${imgUrl ? `<img class="avatar-thumb" src="${imgUrl}" onclick="openImageModal('${imgUrl}', '${v.nombre} ${v.apellido}', 'CI: ${v.cedula} — ${v.motivo_veto}')" />` : '-'}</td>
        <td><strong style="font-family:JetBrains Mono,monospace;">${v.cedula}</strong></td>
        <td><strong>${v.nombre} ${v.apellido}</strong></td>
        <td><span class="risk-badge risk-${v.nivel_riesgo}">${v.nivel_riesgo}</span></td>
        <td style="color:var(--text-muted);max-width:260px;">${v.motivo_veto}</td>
        <td>
          <button class="btn btn-danger btn-sm" onclick="eliminarVetado(${v.id})">Levantar Veto</button>
        </td>
      </tr>`;
  }).join('');
}

function renderAlertsTable() {
  const tbody = document.getElementById('tbody-alerts');
  if (!AppState.alertas.length) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--text-dim);padding:28px;">Sin alertas registradas en el historial.</td></tr>`;
    return;
  }

  tbody.innerHTML = AppState.alertas.map(a => {
    const p = a.persona_vetada || {};
    const capUrl = a.screenshot_path ? '/' + a.screenshot_path.replace(/^\.\//, '') : '';
    const fechaHora = new Date(a.created_at).toLocaleString();
    return `
      <tr>
        <td>${capUrl ? `<img class="avatar-thumb" src="${capUrl}" onclick="openImageModal('${capUrl}', 'Evidencia de Alerta #${a.id}', '${a.camara_id} (${a.zona}) — Similitud: ${(a.similitud * 100).toFixed(1)}%')" />` : '-'}</td>
        <td><strong>${p.nombre || 'ID ' + a.persona_vetada_id} ${p.apellido || ''}</strong><br/><span style="font-size:11px;color:var(--text-muted);">CI: ${p.cedula || '-'}</span></td>
        <td><span class="risk-badge risk-${p.nivel_riesgo || 'ALTO'}">${p.nivel_riesgo || 'ALTO'}</span></td>
        <td><strong>${(a.similitud * 100).toFixed(1)}%</strong></td>
        <td>${a.camara_id}<br/><span style="font-size:11px;color:var(--text-muted);">${a.zona}</span></td>
        <td style="font-size:12px;color:var(--text-muted);">${fechaHora}</td>
        <td>
          ${a.revisada
            ? `<span style="color:#34d399;font-weight:700;font-size:12px;">✔ Atendida</span>`
            : `<button class="btn btn-emerald btn-sm" onclick="marcarAlertaRevisada(${a.id})">Marcar Revisada</button>`
          }
        </td>
      </tr>`;
  }).join('');
}

async function eliminarVetado(id) {
  await fetch(`/api/v1/vetados/${id}`, { method: 'DELETE' });
  refreshAllData();
}

async function marcarAlertaRevisada(id) {
  await fetch(`/api/v1/alertas/${id}/revisar`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ revisada: true, notas: 'Revisada por operador en consola' }),
  });
  refreshAllData();
}

// ----------------------------------------------------------------------------
// 5. EVENTOS DE FORMULARIOS (VETADOS, FOTO MULTIRROSTRO Y VIDEO COMPLETO)
// ----------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  // Formulario de Registro de Persona Vetada (Módulo 2)
  document.getElementById('form-vetar').addEventListener('submit', async (e) => {
    e.preventDefault();
    const msg = document.getElementById('msg-vetar');
    const fd = new FormData(e.target);
    const fileInput = document.getElementById('input-foto-vetar');

    if ((!fileInput.files || fileInput.files.length === 0) && AppState.capturedBlobForBlacklist) {
      fd.set('foto', AppState.capturedBlobForBlacklist, 'camara_captura.jpg');
    } else if (!fileInput.files || fileInput.files.length === 0) {
      msg.innerHTML = `<div class="banner danger" style="margin-top:12px;">❌ Selecciona una fotografía o captura un rostro desde el Monitor / Video.</div>`;
      return;
    }

    msg.innerHTML = `<div class="banner" style="margin-top:12px;">⏳ Extrayendo vector ArcFace de 512 dimensiones e indexando en FAISS...</div>`;
    const res = await fetch('/api/v1/vetados', { method: 'POST', body: fd });
    const data = await res.json();
    if (res.ok) {
      msg.innerHTML = `<div class="banner safe" style="margin-top:12px;">✅ <strong>${data.nombre} ${data.apellido}</strong> (CI: ${data.cedula}) fue registrado e indexado en FAISS exitosamente.</div>`;
      e.target.reset();
      limpiarCaptura();
      refreshAllData();
    } else {
      msg.innerHTML = `<div class="banner danger" style="margin-top:12px;">❌ Error: ${data.detail}</div>`;
    }
  });

  // Formulario de Verificación Multirrostro por Fotografía (Módulo 4 - Foto)
  document.getElementById('form-file-scan').addEventListener('submit', async (e) => {
    e.preventDefault();
    const msg = document.getElementById('msg-file-scan');
    msg.innerHTML = `<div class="banner" style="margin-top:14px;">⏳ Detectando múltiples rostros en la fotografía y consultando FAISS en lote...</div>`;
    const fd = new FormData(e.target);
    const res = await fetch('/api/v1/recognize', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) {
      msg.innerHTML = `<div class="banner danger" style="margin-top:14px;">❌ ${data.detail}</div>`;
      return;
    }

    const cardsHtml = renderMultifaceCards(data.matches, data.no_matches);
    if (data.alert_triggered) {
      playAlertBeep();
      msg.innerHTML = `
        <div class="banner danger" style="margin-top:14px;">
          <h4>🚨 ${data.matches.length} PERSONA(S) VETADA(S) DETECTADA(S) EN LA FOTOGRAFÍA</h4>
          <p style="margin-top:4px;font-size:12px;">
            Se analizaron <b>${data.faces_detected}</b> rostro(s) en simultáneo (${data.matches.length} vetado(s) y ${data.no_matches.length} autorizado(s)) en <b>${data.processing_time_ms} ms</b>.
          </p>
        </div>
        <div class="multiface-grid" style="margin-top:12px;">${cardsHtml}</div>`;
    } else if (data.faces_detected > 0) {
      msg.innerHTML = `
        <div class="banner safe" style="margin-top:14px;">
          ✅ <strong>Sin antecedentes:</strong> Se analizaron simultáneamente <b>${data.faces_detected}</b> rostro(s) en <b>${data.processing_time_ms} ms</b> y ninguno coincide con la lista negra.
        </div>
        <div class="multiface-grid" style="margin-top:12px;">${cardsHtml}</div>`;
    } else {
      msg.innerHTML = `
        <div class="banner" style="margin-top:14px;">
          ⚠️ No se detectaron rostros visibles en la imagen enviada (${data.processing_time_ms} ms).
        </div>`;
    }
    refreshAllData();
  });

  // Formulario de Análisis Completo de Archivo de Video (Módulo 4 - Video)
  document.getElementById('form-video-scan').addEventListener('submit', async (e) => {
    e.preventDefault();
    const msg = document.getElementById('msg-video-scan');
    msg.innerHTML = `<div class="banner" style="margin-top:14px;">⏳ Procesando fotogramas del video con IA Multirrostro (RetinaFace + ArcFace + FAISS)... Esto puede tomar unos segundos según la duración del video.</div>`;

    const fd = new FormData(e.target);
    try {
      const res = await fetch('/api/v1/recognize/video', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) {
        msg.innerHTML = `<div class="banner danger" style="margin-top:14px;">❌ Error al analizar video: ${data.detail}</div>`;
        return;
      }

      if (data.alert_triggered) {
        playAlertBeep();
      }

      const kpiHtml = `
        <div class="video-kpi-grid">
          <div class="video-kpi-box">
            <span class="val">${data.frames_analyzed} / ${data.total_frames}</span>
            <span class="lbl">Frames Analizados</span>
          </div>
          <div class="video-kpi-box">
            <span class="val">${data.duration_seconds}s (${data.fps} fps)</span>
            <span class="lbl">Duración Video</span>
          </div>
          <div class="video-kpi-box">
            <span class="val" style="color:#60a5fa;">${data.total_faces_detected}</span>
            <span class="lbl">Rostros Analizados</span>
          </div>
          <div class="video-kpi-box">
            <span class="val" style="color:#a78bfa;">${data.max_simultaneous_faces}</span>
            <span class="lbl">Máx. Rostros Simultáneos</span>
          </div>
          <div class="video-kpi-box">
            <span class="val" style="color:${data.unique_vetados_detected.length ? '#f87171' : '#34d399'};">${data.unique_vetados_detected.length}</span>
            <span class="lbl">Vetados Identificados</span>
          </div>
        </div>
      `;

      let vetadosSectionHtml = '';
      if (data.unique_vetados_detected.length > 0) {
        const itemsHtml = data.unique_vetados_detected.map(v => {
          const capUrl = v.screenshot_path ? '/' + v.screenshot_path.replace(/^\.\//, '') : '';
          const thumb = v.face_crop_base64 || capUrl || (v.foto_path ? '/' + v.foto_path.replace(/^\.\//, '') : '');
          const tsBadges = v.timestamps.slice(0, 8).map(t => `<span class="ts-pill">⏱️ ${t}</span>`).join('');
          return `
            <div class="face-card is-vetado" style="margin-top:8px;">
              ${thumb ? `<img src="${thumb}" onclick="openImageModal('${capUrl || thumb}', 'Evidencia en Video: ${v.nombre} ${v.apellido}', 'CI: ${v.cedula} • Máx Similitud: ${(v.max_similitud * 100).toFixed(1)}% • Apariciones: ${v.apariciones}')" />` : ''}
              <div class="face-card-info">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                  <span class="face-card-title" style="color:#fca5a5;">🚨 ${v.nombre} ${v.apellido} (CI: ${v.cedula})</span>
                  <span class="risk-badge risk-${v.nivel_riesgo}">${v.nivel_riesgo}</span>
                </div>
                <div class="face-card-sub">
                  Similitud Máx: <b>${(v.max_similitud * 100).toFixed(1)}%</b> • Apariciones: <b>${v.apariciones} frame(s)</b> • Motivo: ${v.motivo_veto}
                </div>
                <div style="margin-top:4px;">${tsBadges}</div>
              </div>
            </div>`;
        }).join('');

        vetadosSectionHtml = `
          <div style="margin-top:14px;">
            <div style="font-size:12px;font-weight:700;text-transform:uppercase;color:#fca5a5;margin-bottom:6px;">
              🚨 Personas Vetadas Identificadas en el Video (${data.unique_vetados_detected.length})
            </div>
            ${itemsHtml}
          </div>`;
      }

      let timelineHtml = '';
      if (data.timeline && data.timeline.length > 0) {
        const framesHtml = data.timeline.map(fr => {
          const preview = fr.frame_preview_base64 || '';
          const names = fr.matches.map(m => `🚨 ${m.match.nombre}`).join(', ');
          return `
            <div class="timeline-card ${fr.alert_triggered ? 'alert-frame' : ''}">
              ${preview ? `<img src="${preview}" onclick="openImageModal('${preview}', 'Fotograma #${fr.frame_number} (t=${fr.timestamp_label})', '${fr.faces_detected} rostro(s) simultáneo(s) en escena')" />` : ''}
              <div class="timeline-card-body">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:#93c5fd;">⏱️ ${fr.timestamp_label}</span>
                  <span style="font-size:11px;color:var(--text-muted);">👥 ${fr.faces_detected} rostro(s)</span>
                </div>
                <div style="margin-top:4px;font-size:11px;color:${fr.alert_triggered ? '#fca5a5' : '#6ee7b7'};font-weight:600;">
                  ${fr.alert_triggered ? `${names} (${fr.matches.length} vetado / ${fr.no_matches.length} ok)` : `✅ ${fr.no_matches.length} autorizado(s)`}
                </div>
              </div>
            </div>`;
        }).join('');

        timelineHtml = `
          <div style="margin-top:16px;">
            <div style="font-size:12px;font-weight:700;text-transform:uppercase;color:var(--text-muted);">
              🎞️ Línea de Tiempo de Fotogramas con Detecciones (${data.timeline.length} eventos — Clic en imagen para ampliar)
            </div>
            <div class="timeline-grid">${framesHtml}</div>
          </div>`;
      }

      const headerBannerClass = data.alert_triggered ? 'banner danger' : 'banner safe';
      const headerBannerTitle = data.alert_triggered
        ? `🚨 ¡ALERTA EN VIDEO! Se detectaron ${data.unique_vetados_detected.length} persona(s) vetada(s) en "${data.video_filename}" (${data.processing_time_ms} ms)`
        : `✅ Video "${data.video_filename}" analizado (${data.processing_time_ms} ms): ${data.total_faces_detected} rostro(s) verificados sin coincidencias en lista negra.`;

      msg.innerHTML = `
        <div class="${headerBannerClass}" style="margin-top:14px;">
          <strong>${headerBannerTitle}</strong>
        </div>
        ${kpiHtml}
        ${vetadosSectionHtml}
        ${timelineHtml}
      `;
      refreshAllData();
    } catch (err) {
      msg.innerHTML = `<div class="banner danger" style="margin-top:14px;">❌ Error de red procesando el video: ${err.message}</div>`;
    }
  });

  refreshAllData();
});
