const $ = (selector, root=document) => root.querySelector(selector);
const $$ = (selector, root=document) => [...root.querySelectorAll(selector)];
let options = null;
let jobs = [];
let videoObjectUrl = null;
let currentEngine = null;
let draggedJobId = null;
let shots = [];
let shotSequence = 0;
let enhancedSoundscape = '';
let enhancedReferenceProtocol = null;
let sectionUndoSnapshots = {references:null, visuals:null, sound:null};
let referenceRoles = {image: [], video: [], audio: []};
let activeMemoryProfile = null;
let activePage = new URLSearchParams(location.search).get('page') === 'create' ? 'create' : 'tasks';
let activeShotIndex = 0;
let engineReconcilePromise = null;
let workspaceBrowseParent = null;
let promptEditorMode = 'structured';

function invalidatePromptEnhancement(scope='visuals') {
  const scopes = scope === 'all' ? ['references','visuals','sound'] : [scope];
  for (const item of scopes) {
    sectionUndoSnapshots[item] = null;
    const suffix = item === 'references' ? 'Reference' : item === 'visuals' ? 'Visual' : 'Sound';
    const undo = $(`#undo${suffix}Enhancement`); if (undo) undo.hidden = true;
    const message = $(`#${item === 'references' ? 'reference' : item === 'visuals' ? 'visual' : 'sound'}EnhancementMessage`);
    if (message) message.hidden = true;
  }
  const message = $('#enhancementMessage'); if (message) message.hidden = true;
}

const engineDescriptions = {
  first_last: '文生视频与首帧、尾帧、首尾帧约束；任务内可热切换原始权重或 LoRA。',
  reference: '图片、视频、音频多模态参考；任务内可热切换原始权重或 LoRA。',
};

function currentVariant() { return $('[name="model_variant"]')?.value || 'base'; }
function currentEngineKey() {
  if (currentEngine === 'reference') return currentVariant() === 'lora' ? 'reference_lora' : 'reference';
  return currentVariant() === 'lora' ? 'lora' : 'original';
}

function renderEngineLobby() {
  const unified = options?.deployment_mode === 'unified_console';
  const active = options?.current_engine || null;
  $('#engineLobby').hidden = !unified || Boolean(active);
  $('#createPage').hidden = unified && !active;
  if (unified && !active) $('#tasksPage').hidden = true;
  $('.workspace-tabs').hidden = unified && !active;
  $('#exitEngine').hidden = !unified || !active;
  if (active) {
    $('#engineLobby').classList.remove('loading');
    $('#engineLobbyMessage').hidden = true;
  }
  if (!unified) return;
  renderWorkspace();
  const families = options.service_families || {};
  $('#engineChoices').innerHTML = Object.entries(families).map(([key, info]) => `
    <section class="engine-choice-group"><h2>${escapeHtml(info.label)}</h2><div>
      <button type="button" class="engine-choice" data-enter-engine="${escapeHtml(key)}">
        <strong>${escapeHtml(info.label)}</strong><small>${escapeHtml(engineDescriptions[key] || info.description || '')}</small>
      </button>
    </div></section>`).join('');
  $$('[data-enter-engine]').forEach(button => button.addEventListener('click', () => enterEngine(button.dataset.enterEngine)));
}

function renderWorkspace() {
  const workspace = options?.workspace?.current;
  if (!workspace) return;
  $('#workspaceName').textContent = workspace.is_default ? '默认工作空间' : (workspace.name || '工作空间');
  $('#workspacePath').textContent = workspace.path;
  $('#workspacePath').title = workspace.path;
  $('#chooseWorkspace').disabled = !options.workspace.switchable;
}

async function browseWorkspace(path) {
  const suffix = path ? `?path=${encodeURIComponent(path)}` : '';
  const document = await (await api(`/api/v1/workspace/browse${suffix}`)).json();
  $('#workspacePathInput').value = document.path;
  workspaceBrowseParent = document.parent;
  $('#workspaceParent').disabled = !document.parent;
  $('#workspaceDirectories').innerHTML = document.directories.length
    ? document.directories.map(item => `<button type="button" class="workspace-directory" data-workspace-path="${escapeHtml(item.path)}"><i>▸</i><span>${escapeHtml(item.name)}</span></button>`).join('')
    : '<div class="manager-empty">这个文件夹中没有子文件夹</div>';
  $$('.workspace-directory').forEach(button => button.addEventListener('click', () => browseWorkspace(button.dataset.workspacePath).catch(showWorkspaceError)));
}

function showWorkspaceError(error) {
  $('#workspaceMessage').textContent = error.message;
  $('#workspaceMessage').hidden = false;
}

async function openWorkspaceDialog() {
  $('#workspaceMessage').hidden = true;
  $('#workspaceDialog').showModal();
  try { await browseWorkspace(options.workspace.current.path); }
  catch (error) { showWorkspaceError(error); }
}

async function activateWorkspace() {
  const button = $('#selectWorkspace');
  button.disabled = true; button.textContent = '正在切换…';
  try {
    await api('/api/v1/workspace', {
      method:'PUT', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path:$('#workspacePathInput').value.trim()}),
    });
    $('#workspaceDialog').close();
    jobs = [];
    await reloadOptions();
    await refreshJobs();
  } catch (error) { showWorkspaceError(error); }
  finally { button.disabled = false; button.textContent = '使用这个文件夹'; }
}

async function enterEngine(engine) {
  const lobby = $('#engineLobby'), message = $('#engineLobbyMessage');
  lobby.classList.add('loading');
  message.textContent = `正在加载${options.service_families[engine].label}（含原始权重与LoRA开关），首次进入可能需要几十秒…`;
  message.hidden = false;
  $$('[data-enter-engine]').forEach(button => button.disabled = true);
  try {
    await api('/api/v1/engine', {
      method:'PUT', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({service_family:engine}),
    });
    await reloadOptions();
    switchPage('tasks');
  } catch (error) {
    // The model load may have completed even if the browser lost the long PUT
    // response. Reconcile with server truth before reporting a false failure.
    const recovered = await reconcileEngineState(true).catch(() => false);
    if (!recovered) {
      message.textContent = `进入失败：${error.message}`;
      $$('[data-enter-engine]').forEach(button => button.disabled = false);
    }
  } finally { lobby.classList.remove('loading'); }
}

async function exitEngine() {
  if (!confirm('退出当前模式会释放模型热态。确认退出？')) return;
  const button = $('#exitEngine'); button.disabled = true; button.textContent = '正在释放…';
  try {
    await api('/api/v1/engine', {method:'DELETE'});
    await reloadOptions();
  } catch (error) { alert(`退出失败：${error.message}`); }
  finally { button.disabled = false; button.textContent = '切换模型'; }
}

function applyOptions(document) {
  const previousEngine = currentEngine;
  options = document;
  currentEngine = options.current_engine;
  if (previousEngine !== currentEngine) invalidatePromptEnhancement('all');
  renderEngineLobby();
  renderReferenceMediaPolicy();
  renderGenerationLimitPolicy();
  if (currentEngine) {
    applyEngineIdentity();
    switchPage(activePage);
  }
  renderMemoryProfiles(); updateContract();
}

async function reloadOptions() {
  applyOptions(await (await api('/api/v1/options')).json());
  await checkHealth();
}

async function reconcileEngineState(force=false) {
  if (engineReconcilePromise) return engineReconcilePromise;
  engineReconcilePromise = (async () => {
    const document = await (await api('/api/v1/options')).json();
    const lobbyStuck = $('#engineLobby').classList.contains('loading');
    const changed = document.current_engine !== currentEngine
      || Boolean(document.engine_control?.switching) !== Boolean(options?.engine_control?.switching);
    if (force || changed || (lobbyStuck && document.current_engine)) {
      applyOptions(document);
    }
    return Boolean(document.current_engine) && !document.engine_control?.switching;
  })();
  try { return await engineReconcilePromise; }
  finally { engineReconcilePromise = null; }
}

function apiHeaders() {
  const headers = {};
  const key = localStorage.getItem('h3serve_api_key');
  if (key) headers['X-API-Key'] = key;
  return headers;
}

async function api(path, init={}) {
  init.headers = {...apiHeaders(), ...(init.headers || {})};
  const response = await fetch(path, init);
  if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
  return response;
}

async function configureServerMimoKey(value) {
  return api('/api/v1/settings/mimo-key', {
    method:'PUT',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({api_key:String(value || '').trim()}),
  });
}

async function serverMimoKeyStatus() {
  const response = await api('/api/v1/settings/mimo-key');
  return response.json();
}

async function serverReferenceMediaSettings() {
  const response = await api('/api/v1/settings/reference-media');
  return response.json();
}

async function configureServerReferenceMedia(imageResolution, videoResolution) {
  const response = await api('/api/v1/settings/reference-media', {
    method:'PUT',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      image_resolution:String(imageResolution || '').trim(),
      video_resolution:String(videoResolution || '').trim(),
    }),
  });
  return response.json();
}

async function serverGenerationLimitSettings() {
  const response = await api('/api/v1/settings/generation-limits');
  return response.json();
}

async function configureServerGenerationLimits(presetLimits) {
  const response = await api('/api/v1/settings/generation-limits', {
    method:'PUT',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({preset_limits:presetLimits}),
  });
  return response.json();
}

function resolutionPolicyLabel(value) {
  return value === 'original' ? '原分辨率' : String(value || '').toUpperCase();
}

function renderReferenceMediaPolicy(document=null) {
  const policy = document || options?.reference_media_processing || {};
  const image = policy.image_resolution || policy.image_default || '720p';
  const video = policy.video_resolution || policy.video_default || '360p';
  const imageControl = $('#globalReferenceImageResolution');
  const videoControl = $('#globalReferenceVideoResolution');
  if (imageControl) imageControl.value = image;
  if (videoControl) videoControl.value = video;
  const hint = $('#referenceMediaPolicyHint');
  if (hint) {
    hint.textContent = `自动等比降分辨率：图片${resolutionPolicyLabel(image)} · 视频${resolutionPolicyLabel(video)}`;
  }
}

function renderGenerationLimitPolicy(document=null) {
  const policy = document || options?.generation_limits;
  if (!policy) return;
  const detected = Number(policy.detected_vram_gib);
  const status = $('#generationLimitStatus');
  if (status) status.textContent = Number.isFinite(detected)
    ? `当前GPU ${detected.toFixed(1)} GB`
    : '显存未检测到';
  const hint = $('#generationLimitHint');
  if (hint) hint.textContent = '逐项设置每种分辨率和画面比例允许提交的最长时长；只限制任务提交，不改变输出画面。';
  const matrix = policy.preset_limits || policy.max_by_preset || options?.duration?.max_by_preset;
  const editor = $('#generationLimitEditor');
  if (!editor || !matrix) return;
  const ratios = policy.aspect_ratios || options?.aspect_ratios || ['1:1','4:3','3:4','16:9','9:16'];
  const resolutions = policy.resolutions || options?.resolutions || Object.keys(matrix);
  editor.innerHTML = `<div class="generation-limit-head"><b>规格</b>${ratios.map(ratio => `<b>${escapeHtml(ratio)}</b>`).join('')}</div>${resolutions.map(resolution => `<section><b>${escapeHtml(String(resolution).toUpperCase())}</b>${ratios.map(ratio => { const seconds=Number(matrix?.[resolution]?.[ratio] ?? 15); return `<label title="${escapeHtml(resolution)} ${escapeHtml(ratio)} 最长时长"><input type="range" min="1" max="15" step="0.5" value="${seconds}" data-limit-resolution="${escapeHtml(resolution)}" data-limit-ratio="${escapeHtml(ratio)}"><output>${seconds.toFixed(1)}秒</output></label>`; }).join('')}</section>`).join('')}`;
  $$('[data-limit-resolution]', editor).forEach(input => input.addEventListener('input', () => { input.nextElementSibling.textContent = `${Number(input.value).toFixed(1)}秒`; }));
}

function readGenerationLimitEditor() {
  const limits = {};
  $$('[data-limit-resolution]', $('#generationLimitEditor')).forEach(input => {
    const resolution = input.dataset.limitResolution;
    const ratio = input.dataset.limitRatio;
    if (!limits[resolution]) limits[resolution] = {};
    limits[resolution][ratio] = Number(input.value);
  });
  return limits;
}

function selected(name) { return $(`[name="${name}"]`).value; }
function escapeHtml(value='') { return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char])); }
function formatSeconds(value) {
  if (value == null || !Number.isFinite(Number(value))) return '估算中';
  const seconds = Math.max(0, Math.round(Number(value)));
  if (seconds < 60) return `约 ${seconds} 秒`;
  return `约 ${Math.floor(seconds / 60)}分${String(seconds % 60).padStart(2, '0')}秒`;
}

function formatElapsed(value) {
  if (value == null || !Number.isFinite(Number(value))) return '未记录';
  const seconds = Math.max(0, Number(value));
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} 秒`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${minutes}分${remainder.toFixed(1).padStart(4, '0')}秒`;
}

function renderMemoryProfiles() {
  const memory = options?.host_memory;
  if (!memory) return;
  activeMemoryProfile = memory.active_profile;
  const effective = Number(memory.detected?.effective_limit_gib || 0);
  const available = Number(memory.detected?.available_gib || 0);
  $('#detectedMemory').textContent = `有效 ${effective.toFixed(1)} GiB · 当前可用 ${available.toFixed(1)} GiB`;
  $('#memoryProfileList').innerHTML = memory.profiles.map(profile => {
    // 128GB Windows hosts commonly expose about 110GiB to WSL. The backend
    // remains authoritative and also checks current free memory.
    const measuredCapacityFloors = {
      fullspeed: 96.835,
      generation_hot: 81.141,
      compact: 57.931,
    };
    const capacityFloor = measuredCapacityFloors[profile.key]
      ?? Number(profile.minimum_ram_gib) * .95;
    const enough = effective >= capacityFloor;
    const selectable = ['validated','review'].includes(profile.evidence);
    const enabled = enough && selectable;
    const badge = profile.evidence === 'validated' ? `最低 ${profile.minimum_ram_gib}GB` : profile.evidence === 'review' ? `最低 ${profile.minimum_ram_gib}GB · 终审中` : '实验 · 尚未开放';
    return `<label class="memory-profile-option ${enabled ? '' : 'unavailable'}">
      <input type="radio" name="host_memory_profile" value="${escapeHtml(profile.key)}" ${profile.key === memory.active_profile ? 'checked' : ''} ${enabled ? '' : 'disabled'}>
      <span><strong>${escapeHtml(profile.label)}</strong><small>${escapeHtml(profile.description)}</small></span>
      <em class="${profile.evidence === 'validated' ? '' : 'experimental'}">${badge}</em>
    </label>`;
  }).join('');
  $('#memoryProfileHint').textContent = currentEngine ? '模式改变CPU权重驻留和阶段切换，不改变模型权重、采样步数或画质。96GB/64GB超分时会自动释放H3，完成后恢复热态。' : '当前没有加载H3引擎；可先选择内存模式，再进入生成模式。';
}

async function applyMemoryProfile() {
  const selectedProfile = $('[name="host_memory_profile"]:checked')?.value;
  if (!selectedProfile || selectedProfile === activeMemoryProfile) return;
  $('#memoryProfileHint').textContent = '正在释放旧驻留并重新预加载H3，请勿提交任务…';
  try {
    await api('/api/v1/memory-profile', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({profile: selectedProfile}),
    });
    options = await (await api('/api/v1/options')).json();
    renderMemoryProfiles();
    $('#memoryProfileHint').textContent = '切换完成。模型权重、采样配置和生成质量均未改变。';
  } catch (error) {
    $('#memoryProfileHint').textContent = `切换失败：${error.message}`;
    const current = $(`[name="host_memory_profile"][value="${activeMemoryProfile}"]`);
    if (current) current.checked = true;
    throw error;
  }
}

function timecode(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const remainder = value - minutes * 60;
  return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(1).padStart(4, '0')}`;
}

function h3Timecode(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const remainder = value - minutes * 60;
  return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(3).padStart(6, '0')}`;
}

function conditionMode() {
  if (currentEngine === 'reference') return 'Ref2VA';
  const first = Boolean($('[name="first_frame"]').files[0]);
  const last = Boolean($('[name="last_frame"]').files[0]);
  if (first && last) return 'FL2VA';
  if (first) return 'I2VA';
  if (last) return 'L2VA';
  return 'T2VA';
}

function pictureAlignment(mode, duration) {
  if (mode === 'I2VA') return 'For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.';
  if (mode === 'FL2VA') return `How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot ${shots.length}) aligns with the ${duration.toFixed(2)}-second mark of the target video.`;
  if (mode === 'L2VA') return `How the reference pictures align with the target video — <Picture 1> (from [Shot ${shots.length}]) aligns with the ${duration.toFixed(2)}-second mark of the target video.`;
  return '';
}

function addShot({duration_seconds=5, prompt=''}={}) {
  invalidatePromptEnhancement();
  shots.push({id:`shot-${++shotSequence}`, duration_seconds:Number(duration_seconds), prompt:String(prompt)});
  renderShots();
}

function shotTotal() { return shots.reduce((total, shot) => total + (Number(shot.duration_seconds) || 0), 0); }

function currentMaxDuration() {
  const fallback = 15;
  if (!options) return fallback;
  if (selected('size_mode') !== 'custom') {
    const configured = Number(
      options.duration?.max_by_preset?.[selected('resolution')]?.[selected('aspect_ratio')]
    );
    if (Number.isFinite(configured)) return configured;
  }
  const geometry = currentGeometry();
  const budget = Number(options.duration?.max_native_pixel_frames);
  if (!budget || !geometry.width || !geometry.height) return fallback;
  const rawFrameLimit = Math.min(362, Math.floor(budget / (geometry.width * geometry.height)));
  if (rawFrameLimit < 5) return 0;
  const legalFrames = 5 + 17 * Math.floor((rawFrameLimit - 5) / 17);
  const exactMaximum = Math.min(Number(options.duration?.max) || fallback, legalFrames / 24);
  // Storyboard sliders use half-second increments. Keep their visible ceiling
  // on the same grid while the HTTP API retains the exact H3 frame-grid limit.
  return Math.floor(exactMaximum * 2 + 1e-9) / 2;
}

function framesForDuration(seconds) {
  return Math.min(362, 5 + 17 * Math.max(0, Math.round((Number(seconds) * 24 - 5) / 17)));
}

function synchronizeDurationInputs() {
  if (promptEditorMode === 'freeform') return;
  const total = shotTotal();
  if (total >= 1 && total <= currentMaxDuration()) {
    $('[name="duration_seconds"]').value = total.toFixed(1);
  }
}

function referenceBindingError() {
  if (currentEngine !== 'reference') return '';
  const counts = {
    picture: referenceFiles('image').length,
    video: referenceFiles('video').length,
    audio: referenceFiles('audio').length,
  };
  if (!counts.picture && !counts.video && !counts.audio) {
    return '当前没有实际上传的参考素材。刷新页面后浏览器不会保留本地文件，请重新添加图片、视频或音频。';
  }
  const labels = [
    shots.map(shot => shot.prompt).join('\n'),
    $('#referenceDefinitions')?.value || '',
    $('#referenceRetention')?.value || '',
    $('#referenceSummary')?.value || '',
    $('#overallSoundscape')?.value || '',
    $('#bgmStyle')?.value || '',
  ].join('\n').matchAll(/<(Picture|Video|Audio)\s+(\d+)>/gi);
  for (const match of labels) {
    const kind = match[1].toLowerCase();
    const index = Number(match[2]);
    if (index < 1 || index > counts[kind]) {
      return `<${match[1]} ${index}> 没有对应的已上传文件，请重新添加素材或删除这个引用。`;
    }
  }
  return '';
}

function referenceFiles(kind) {
  if (kind === 'image') return Array.from($('#referenceImages')?.files || []).slice(0, 9);
  if (kind === 'video') return Array.from($('#referenceVideos')?.files || []).slice(0, 3);
  return Array.from($('#referenceAudios')?.files || []).slice(0, 3);
}

function referenceMediaPayload() {
  return ['image','video','audio'].flatMap(kind => referenceFiles(kind).map((file, index) => ({
    kind, name:file.name, mime_type:file.type, role:String(referenceRoles[kind]?.[index] || '').trim(),
  })));
}

function defaultReferenceRole(kind, index) {
  if (kind === 'image') return `说明 <Picture ${index + 1}> 是哪个人物、物体、场景或画面风格，以及需要保留什么`;
  if (kind === 'video') return `说明 <Video ${index + 1}> 用于动作、镜头结构还是视频延续`;
  return `例如：<Audio ${index + 1}> 只作为女孩(S1)的音色参考，不复用原音频台词`;
}

function renderReferencePreviews(kind) {
  const files = referenceFiles(kind);
  referenceRoles[kind] = files.map((_, index) => referenceRoles[kind]?.[index] || '');
  const preview = kind === 'image' ? $('#referencePreview') : kind === 'video' ? $('#referenceVideoPreview') : $('#referenceAudioPreview');
  preview.innerHTML = files.map((file, index) => {
    const label = kind === 'image' ? `Picture ${index + 1}` : kind === 'video' ? `Video ${index + 1}` : `Audio ${index + 1}`;
    const visual = kind === 'image'
      ? `<img src="${URL.createObjectURL(file)}" alt="参考图片${index + 1}">`
      : kind === 'video'
        ? `<video src="${URL.createObjectURL(file)}" muted preload="metadata"></video>`
        : `<span class="reference-audio-icon">♫</span>`;
    return `<article class="reference-chip" data-reference-kind="${kind}" data-reference-index="${index}"><button type="button" class="reference-chip-remove" data-reference-remove aria-label="删除素材">×</button>
      <div class="reference-media-visual">${visual}</div>
      <div class="reference-chip-copy"><button type="button" class="reference-token" data-insert-reference title="插入当前提示词">&lt;${label}&gt;</button><small title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</small><input maxlength="1000" data-reference-role aria-label="素材用途" title="${escapeHtml(defaultReferenceRole(kind, index))}" placeholder="用途（可选）" value="${escapeHtml(referenceRoles[kind][index])}"></div>
    </article>`;
  }).join('');
  $$('[data-reference-role]', preview).forEach((input, index) => input.addEventListener('input', event => {
    referenceRoles[kind][index] = event.target.value;
    invalidatePromptEnhancement('references');
    compileStoryboard();
  }));
  $$('[data-reference-remove]', preview).forEach((button, index) => button.addEventListener('click', () => removeReferenceFile(kind, index)));
  $$('[data-insert-reference]', preview).forEach((button, index) => button.addEventListener('click', () => insertReferenceToken(kind, index)));
}

function insertReferenceToken(kind, index) {
  const prefix = kind === 'image' ? 'Picture' : kind === 'video' ? 'Video' : 'Audio';
  const token = `<${prefix} ${index + 1}>`;
  if (promptEditorMode === 'freeform') {
    insertTokenIntoTextarea($('#freeformPrompt'), token, false);
    return;
  }
  const textareas = $$('#shotList .shot-prompt textarea');
  const target = textareas[Math.min(activeShotIndex, textareas.length - 1)] || textareas[0];
  if (!target) return;
  insertTokenIntoTextarea(target, token, false);
}

function insertTokenIntoTextarea(target, token, replaceMention=false) {
  const start = target.selectionStart ?? target.value.length;
  const end = target.selectionEnd ?? start;
  let before = target.value.slice(0, start);
  if (replaceMention && before.endsWith('@')) before = before.slice(0, -1);
  const spacer = before && !/\s$/.test(before) ? ' ' : '';
  target.value = `${before}${spacer}${token} ${target.value.slice(end)}`;
  target.dispatchEvent(new Event('input', {bubbles:true}));
  const cursor = before.length + spacer.length + token.length + 1;
  target.focus(); target.setSelectionRange(cursor, cursor);
}

function referenceMentionItems() {
  return ['image','video','audio'].flatMap(kind => referenceFiles(kind).map((file, index) => ({
    kind, index, file, token:`<${kind === 'image' ? 'Picture' : kind === 'video' ? 'Video' : 'Audio'} ${index + 1}>`,
  })));
}

function updateReferenceMentionMenu(card, textarea) {
  const menu = $('.reference-mention-menu', card);
  if (currentEngine !== 'reference') { menu.hidden = true; return; }
  const before = textarea.value.slice(0, textarea.selectionStart ?? 0);
  if (!before.endsWith('@')) { menu.hidden = true; return; }
  const items = referenceMentionItems();
  menu.innerHTML = items.length ? items.map(item => `<button type="button" data-mention-kind="${item.kind}" data-mention-index="${item.index}"><b>${escapeHtml(item.token)}</b><small>${escapeHtml(item.file.name)}</small></button>`).join('') : '<span>请先添加参考文件</span>';
  menu.hidden = false;
  $$('button', menu).forEach(button => button.addEventListener('mousedown', event => {
    event.preventDefault();
    const kind = button.dataset.mentionKind;
    const index = Number(button.dataset.mentionIndex);
    const token = `<${kind === 'image' ? 'Picture' : kind === 'video' ? 'Video' : 'Audio'} ${index + 1}>`;
    insertTokenIntoTextarea(textarea, token, true);
    menu.hidden = true;
  }));
}

function removeReferenceFile(kind, removeIndex) {
  const input = kind === 'image' ? $('#referenceImages') : kind === 'video' ? $('#referenceVideos') : $('#referenceAudios');
  const transfer = new DataTransfer();
  Array.from(input.files || []).forEach((file, index) => { if (index !== removeIndex) transfer.items.add(file); });
  input.files = transfer.files;
  referenceRoles[kind] = (referenceRoles[kind] || []).filter((_, index) => index !== removeIndex);
  renderReferencePreviews(kind);
  resetReferenceEditors();
  updateContract();
}

function fallbackReferenceProtocol(media) {
  const pictures = media.filter(item => item.kind === 'image');
  const videos = media.filter(item => item.kind === 'video');
  const audios = media.filter(item => item.kind === 'audio');
  // Without multimodal enhancement we do not know whether a picture is a person,
  // an object or a scene. Do not invent Subject numbering from upload order.
  const subjects = pictures.map((item, index) => `<Picture ${index + 1}> is a supplied visual reference. ${item.role || 'Use only its observable attributes when explicitly referenced in detailed_description.'}`);
  const audioDefinitions = audios.map((item, index) => `<Audio ${index + 1}> is the requested audio reference for the speaker or audible layer explicitly associated with <Audio ${index + 1}> in detailed_description. ${item.role || 'For spoken dialogue, transfer voice timbre and delivery without copying the source words.'}`);
  const retention = [
    ...pictures.map((item, index) => `<Picture ${index + 1}>: weak_reference - ${item.role || 'use only the explicitly requested visible identity, object, composition or style characteristics.'}`),
    ...videos.map((item, index) => `<Video ${index + 1}>: weak_reference - ${item.role || 'use only the explicitly requested motion, camera or temporal-structure characteristics.'}`),
    ...audios.map((item, index) => `<Audio ${index + 1}>: reference — ${item.role || 'use as an audio/voice reference only; do not copy source dialogue.'}`),
  ];
  return {
    subject_definitions: [...subjects, ...audioDefinitions],
    summary: `[reference generation${audios.length ? ' + audio reference' : ''}] Generate the requested ${resolvedVideoDuration().toFixed(2)}-second audio-video scene using the supplied visual and audio references according to their stated roles.`,
    retention_analysis: retention,
    style_opening: '',
  };
}

function protocolLines(value) {
  return String(value || '').split(/\r?\n/).map(item => item.trim()).filter(Boolean);
}

function currentReferenceProtocol() {
  if (conditionMode() !== 'Ref2VA') return fallbackReferenceProtocol(referenceMediaPayload());
  return {
    subject_definitions: protocolLines($('#referenceDefinitions').value),
    summary: $('#referenceSummary').value.trim(),
    retention_analysis: protocolLines($('#referenceRetention').value),
    style_opening: '',
  };
}

function populateReferenceEditors(protocol, {overwrite=false}={}) {
  if (!protocol) return;
  const values = {
    referenceDefinitions: (protocol.subject_definitions || []).join('\n'),
    referenceSummary: protocol.summary || '',
    referenceRetention: (protocol.retention_analysis || []).join('\n'),
  };
  for (const [id, value] of Object.entries(values)) {
    const input = $(`#${id}`);
    if (input && (overwrite || !input.value.trim())) input.value = value;
  }
  enhancedReferenceProtocol = currentReferenceProtocol();
}

function resetReferenceEditors() {
  if (conditionMode() !== 'Ref2VA') return;
  enhancedReferenceProtocol = currentReferenceProtocol();
  invalidatePromptEnhancement('references');
  compileStoryboard();
}

function compileStoryboard() {
  if (promptEditorMode === 'freeform') {
    $('#compiledPrompt').value = $('#freeformPrompt').value;
    return $('#compiledPrompt').value;
  }
  let cursor = 0;
  const sections = shots.map((shot, index) => {
    // H3's first shot must not carry a timestamp. Official-style timestamps
    // begin at the second shot and describe that shot's start only.
    const prefix = index === 0 ? '[Shot 1]' : `[Shot ${index + 1}] At ${h3Timecode(cursor)},`;
    const end = cursor + Number(shot.duration_seconds);
    const text = `${prefix}\n${shot.prompt.trim()}`;
    cursor = end;
    return text;
  });
  const bgmEnabled = $('#bgmEnabled').checked;
  const style = $('#bgmStyle').value.trim();
  const soundscapeValue = $('#overallSoundscape').value.trim()
    || enhancedSoundscape
    || 'Preserve dialogue and scene-grounded ambience/action sounds described in each shot.';
  const soundscape = `overall_soundscape: ${soundscapeValue}`;
  const music = `non_diegetic_music: ${bgmEnabled ? style : 'N/A'}`;
  if (conditionMode() === 'Ref2VA') {
    const protocol = currentReferenceProtocol();
    enhancedReferenceProtocol = protocol;
    const subjectRows = protocol.subject_definitions?.length ? protocol.subject_definitions.join('\n') : 'N/A';
    const retentionRows = protocol.retention_analysis?.length ? protocol.retention_analysis.join('\n') : 'N/A';
    const detailed = sections.join('\n\n');
    $('#compiledPrompt').value = [
      `subject_definitions:\n${subjectRows}`,
      `summary:\n${protocol.summary}`,
      `retention_analysis:\n${retentionRows}`,
      `detailed_description:\n${detailed}`,
      soundscape,
      music,
    ].join('\n\n');
  } else {
    const alignment = pictureAlignment(conditionMode(), resolvedVideoDuration());
    const body = `integrated_multimodal_description: ${sections.join('\n\n')}`;
    $('#compiledPrompt').value = [alignment, body, soundscape, music].filter(Boolean).join('\n\n');
  }
  if ($('#compiledPromptPreview')) $('#compiledPromptPreview').textContent = $('#compiledPrompt').value;
  return $('#compiledPrompt').value;
}

function storyboardPayload() {
  return {
    shots: shots.map(shot => ({id:shot.id, duration_seconds:Number(shot.duration_seconds), prompt:shot.prompt.trim()})),
    bgm_enabled: $('#bgmEnabled').checked,
    bgm_style: $('#bgmStyle').value.trim(),
    condition_mode: conditionMode(),
    effective_duration_seconds: resolvedVideoDuration(),
    reference_media: referenceMediaPayload(),
    reference_protocol: conditionMode() === 'Ref2VA' ? currentReferenceProtocol() : null,
    soundtrack: {
      overall_soundscape: $('#overallSoundscape').value.trim(),
      non_diegetic_music: $('#bgmEnabled').checked ? $('#bgmStyle').value.trim() : 'N/A',
    },
  };
}

function enhancementSnapshot() {
  return {
    shots,
    bgm_enabled:$('#bgmEnabled').checked,
    bgm_style:$('#bgmStyle').value,
    soundscape:$('#overallSoundscape').value,
    reference_protocol:conditionMode() === 'Ref2VA' ? currentReferenceProtocol() : null,
    reference_roles:referenceRoles,
  };
}

function resolvedVideoDuration() {
  const requested = Number(selected('duration_seconds')) || 5;
  const frames = framesForDuration(requested);
  return frames / 24;
}

function syncStoryboardTiming() {
  const maximum = currentMaxDuration();
  const durationInput = $('[name="duration_seconds"]');
  durationInput.max = String(maximum);
  if (promptEditorMode === 'freeform') {
    durationInput.value = String(Math.max(1, Math.min(maximum, Number(durationInput.value) || 5)));
    $('#freeformDurationValue').textContent = `${Number(durationInput.value).toFixed(1)} 秒`;
    compileStoryboard();
    updateContract();
    return;
  }
  const total = shotTotal();
  $('#storyboardTotal').textContent = `${total.toFixed(1)} 秒`;
  const valid = total >= 1 && total <= maximum;
  $('#storyboardTimingHint').textContent = valid
    ? (maximum < 15 ? `目标时长 · 当前画布最多${maximum}秒` : '目标时长')
    : `需为1–${maximum}秒`;
  $('#storyboardTimingHint').classList.toggle('invalid', !valid);
  $$('.shot-duration input').forEach(input => { input.max = String(maximum); });
  if (valid) synchronizeDurationInputs();
  compileStoryboard();
  updateContract();
}

function renderShots() {
  let cursor = 0;
  const promptPlaceholder = currentEngine === 'reference'
    ? '写什么：描述这个连续镜头中实际发生的画面、动作、台词、运镜和同步声音。\n怎么写：建议使用英文；按发生顺序写，人物与素材使用 @ 引用，台词保留原语言并写成 <d>[Language] ...</d>。\n示例：A medium shot follows <Subject 1> walking toward the bench. She stops, looks up, and says: <d>[Chinese] 你终于来了。</d>'
    : '写什么：描述这个连续镜头中实际发生的画面、动作、台词、运镜和同步声音。\n怎么写：按发生顺序写清主体、动作、环境、镜头运动和现场声音；台词保留原语言。\n示例：镜头从桌上的书包特写缓慢拉远，女孩走近并拉开拉链，清晰听见脚步声和拉链声。';
  $('#shotList').innerHTML = shots.map((shot, index) => {
    const start = cursor; const end = cursor + Number(shot.duration_seconds || 0); cursor = end;
    return `<article class="shot-card" data-shot-id="${escapeHtml(shot.id)}">
      <header><div><span class="shot-number">SHOT ${index + 1}</span><strong>${timecode(start)} → ${timecode(end)}</strong></div><div class="shot-header-actions"><label class="shot-duration"><span>镜头时长</span><input aria-label="持续时间（秒）" type="range" min="0.5" max="${currentMaxDuration()}" step="0.5" value="${Number(shot.duration_seconds)}"><output>${Number(shot.duration_seconds).toFixed(1)} 秒</output></label><div class="shot-actions"><button type="button" data-shot-up title="上移" ${index === 0 ? 'disabled' : ''}>↑</button><button type="button" data-shot-down title="下移" ${index === shots.length - 1 ? 'disabled' : ''}>↓</button><button type="button" data-shot-remove title="删除" ${shots.length === 1 ? 'disabled' : ''}>×</button></div></div></header>
      <div class="shot-prompt"><textarea rows="5" maxlength="6000" required aria-label="SHOT ${index + 1} 镜头内容" placeholder="${escapeHtml(promptPlaceholder)}">${escapeHtml(shot.prompt)}</textarea><div class="reference-mention-menu" hidden></div></div>
    </article>`;
  }).join('');
  $$('#shotList .shot-card').forEach((card, index) => {
    $('input', card).addEventListener('input', event => { invalidatePromptEnhancement(); shots[index].duration_seconds = Number(event.target.value); renderShotTimesOnly(); });
    $('textarea', card).addEventListener('focus', () => { activeShotIndex = index; });
    $('textarea', card).addEventListener('input', event => { activeShotIndex = index; invalidatePromptEnhancement(); shots[index].prompt = event.target.value; updateReferenceMentionMenu(card, event.target); compileStoryboard(); });
    $('textarea', card).addEventListener('keydown', event => { if (event.key === 'Escape') $('.reference-mention-menu', card).hidden = true; });
    $('textarea', card).addEventListener('blur', () => setTimeout(() => { $('.reference-mention-menu', card).hidden = true; }, 120));
    $('[data-shot-up]', card).addEventListener('click', () => { invalidatePromptEnhancement(); [shots[index - 1], shots[index]] = [shots[index], shots[index - 1]]; renderShots(); });
    $('[data-shot-down]', card).addEventListener('click', () => { invalidatePromptEnhancement(); [shots[index], shots[index + 1]] = [shots[index + 1], shots[index]]; renderShots(); });
    $('[data-shot-remove]', card).addEventListener('click', () => { if (shots.length > 1) { invalidatePromptEnhancement(); shots.splice(index, 1); renderShots(); } });
  });
  syncStoryboardTiming();
}

function renderShotTimesOnly() {
  let cursor = 0;
  $$('#shotList .shot-card').forEach((card, index) => {
    const end = cursor + Number(shots[index].duration_seconds || 0);
    $('header strong', card).textContent = `${timecode(cursor)} → ${timecode(end)}`;
    $('.shot-duration output', card).textContent = `${Number(shots[index].duration_seconds).toFixed(1)} 秒`;
    cursor = end;
  });
  syncStoryboardTiming();
}

const enhancementScopeUi = {
  references: {button:'enhanceReferences', message:'referenceEnhancementMessage', undo:'undoReferenceEnhancement', label:'参考对象'},
  visuals: {button:'enhanceVisuals', message:'visualEnhancementMessage', undo:'undoVisualEnhancement', label:'画面内容'},
  sound: {button:'enhanceSound', message:'soundEnhancementMessage', undo:'undoSoundEnhancement', label:'声音设计'},
};

async function enhanceSection(scope) {
  const ui = enhancementScopeUi[scope], button = $(`#${ui.button}`), message = $(`#${ui.message}`);
  const keyStatus = await serverMimoKeyStatus().catch(() => ({configured:false}));
  if (!keyStatus.configured) { message.textContent = '请先点击右上角设置，填写小米 MiMo API Key。'; message.hidden = false; $('#settingsDialog').showModal(); return; }
  if (scope === 'references' && conditionMode() !== 'Ref2VA') return;
  if (scope === 'references' && (!$('#referenceDefinitions').value.trim() || !$('#referenceRetention').value.trim())) { message.textContent = '请先填写对象定义和保留规则，再让 MiMo 润色。'; message.hidden = false; return; }
  if (scope === 'visuals' && shots.some(shot => !shot.prompt.trim())) { message.textContent = '请先填写每个分镜的镜头内容，再润色画面。'; message.hidden = false; return; }
  if (scope === 'visuals' && conditionMode() === 'Ref2VA' && !$('#referenceSummary').value.trim()) { message.textContent = '请先填写总体摘要，再让 MiMo 润色画面。'; message.hidden = false; return; }
  const bindingError = referenceBindingError();
  if (bindingError) { message.textContent = bindingError; message.hidden = false; return; }
  if (scope === 'sound' && !$('#overallSoundscape').value.trim() && !$('#bgmStyle').value.trim()) { message.textContent = '请先填写画面内声音或配乐描述，再润色声音。'; message.hidden = false; return; }
  if (scope === 'sound' && $('#bgmEnabled').checked && !$('#bgmStyle').value.trim()) { message.textContent = '已开启BGM，请先填写配乐描述或引用音频。'; message.hidden = false; return; }
  synchronizeDurationInputs();
  sectionUndoSnapshots[scope] = enhancementSnapshot();
  button.disabled = true;
  const originalLabel = button.innerHTML;
  button.innerHTML = '<span>✦</span> 正在润色…';
  message.hidden = true;
  try {
    const payload = {...storyboardPayload(), enhancement_scope:scope};
    const form = new FormData(); form.set('storyboard', JSON.stringify(payload));
    appendEnhancementMedia(form);
    const response = await api('/studio/prompt-enhancements', {method:'POST', body:form});
    const result = await response.json();
    applyEnhancementResult(scope, result);
    message.textContent = `${ui.label}润色完成，其他板块未修改。`;
    message.className = 'enhancement-message success'; message.hidden = false;
  } catch (error) { message.textContent = error.message; message.className = 'enhancement-message'; message.hidden = false; }
  finally { button.disabled = false; button.innerHTML = originalLabel; }
}

function appendEnhancementMedia(form) {
  for (const role of ['first_frame','last_frame']) { const file = $(`[name="${role}"]`).files[0]; if (file) form.set(role, file); }
  Array.from($('#referenceImages')?.files || []).forEach((file, index) => form.set(`reference_image_${index + 1}`, file));
  Array.from($('#referenceVideos')?.files || []).forEach((file, index) => form.set(`reference_video_${index + 1}`, file));
  Array.from($('#referenceAudios')?.files || []).forEach((file, index) => form.set(`reference_audio_${index + 1}`, file));
}

function applyEnhancementResult(scope, result) {
  if (scope === 'references' && result.reference_protocol) {
    const current = currentReferenceProtocol();
    populateReferenceEditors({...current, ...result.reference_protocol}, {overwrite:true});
  } else if (scope === 'visuals') {
    if (result.shots) shots = result.shots.map(shot => ({id:shot.id, duration_seconds:Number(shot.duration_seconds), prompt:shot.prompt}));
    if (result.reference_protocol) {
      const current = currentReferenceProtocol();
      populateReferenceEditors({...current, ...result.reference_protocol}, {overwrite:true});
    }
    renderShots();
  } else if (scope === 'sound' && result.soundtrack) {
    enhancedSoundscape = result.soundtrack.overall_soundscape || '';
    $('#overallSoundscape').value = enhancedSoundscape;
    const music = result.soundtrack.non_diegetic_music || 'N/A';
    if ($('#bgmEnabled').checked && music !== 'N/A') $('#bgmStyle').value = music;
  }
  $(`#${enhancementScopeUi[scope].undo}`).hidden = false;
  compileStoryboard();
}

function undoEnhancement(scope) {
  const snapshot = sectionUndoSnapshots[scope];
  if (!snapshot) return;
  if (scope === 'visuals') shots = snapshot.shots;
  if (scope === 'references' && snapshot.reference_protocol) populateReferenceEditors(snapshot.reference_protocol, {overwrite:true});
  if (scope === 'visuals' && snapshot.reference_protocol) populateReferenceEditors(snapshot.reference_protocol, {overwrite:true});
  if (scope === 'sound') {
  $('#bgmEnabled').checked = snapshot.bgm_enabled; $('#bgmStyle').value = snapshot.bgm_style; enhancedSoundscape = snapshot.soundscape || '';
    $('#overallSoundscape').value = snapshot.soundscape || '';
    $('#bgmStyleField').hidden = !snapshot.bgm_enabled;
  }
  sectionUndoSnapshots[scope] = null;
  $(`#${enhancementScopeUi[scope].undo}`).hidden = true;
  $(`#${enhancementScopeUi[scope].message}`).hidden = true;
  if (scope === 'visuals') renderShots(); else compileStoryboard();
}

function setPromptEditorMode(mode) {
  promptEditorMode = mode === 'freeform' ? 'freeform' : 'structured';
  $$('[data-prompt-mode]').forEach(button => {
    button.classList.toggle('active', button.dataset.promptMode === promptEditorMode);
    button.setAttribute('aria-selected', String(button.dataset.promptMode === promptEditorMode));
  });
  $('#structuredPromptEditor').hidden = promptEditorMode !== 'structured';
  $('#freeformPromptEditor').hidden = promptEditorMode !== 'freeform';
  $('#freeformDurationField').hidden = promptEditorMode !== 'freeform';
  if (promptEditorMode === 'structured') synchronizeDurationInputs();
  syncStoryboardTiming();
}

function switchPage(page) {
  if (!currentEngine) return;
  activePage = page;
  $$('.workspace-tabs button').forEach(button => button.classList.toggle('active', button.dataset.page === page));
  $('#createPage').hidden = page !== 'create';
  $('#tasksPage').hidden = page !== 'tasks';
  $('#createPage').classList.toggle('active', page === 'create');
  $('#tasksPage').classList.toggle('active', page === 'tasks');
  if (page === 'tasks') refreshJobs();
}

function currentGeometry() {
  if (selected('size_mode') === 'custom') {
    return {
      width: Number(selected('width')) || 864,
      height: Number(selected('height')) || 480,
    };
  }
  return options.geometry[selected('resolution')][selected('aspect_ratio')];
}

function even(value) { return Math.max(2, Math.round(Number(value) / 2) * 2); }

function resolvedUpscaleGeometry() {
  const source = currentGeometry();
  if (!$('#upscaleEnabled').checked) return source;
  const shortEdge = ({'720p':720, '1080p':1080, '2k':1440})[selected('upscale_resolution')] || 720;
  return source.width >= source.height
    ? {width:even(shortEdge * source.width / source.height), height:shortEdge}
    : {width:shortEdge, height:even(shortEdge * source.height / source.width)};
}

function updateUpscaleControls() {
  if (!options) return;
  const enabled = $('#upscaleEnabled').checked;
  const available = Boolean(options.advanced_limits?.upscaler?.available);
  $('#upscaleCard').classList.toggle('unavailable', !available);
  $('#upscaleEnabled').disabled = !available;
  $('#upscaleControls').hidden = !enabled;
  $('[name="upscale_mode"]').disabled = !enabled;
  $('[name="upscale_resolution"]').disabled = !enabled;
  const source = currentGeometry();
  const sourceShortEdge = Math.min(source.width, source.height);
  const levels = {'720p':720, '1080p':1080, '2k':1440};
  const buttons = $$('.upscale-resolution-tabs button');
  buttons.forEach(button => { button.disabled = levels[button.dataset.upscaleResolution] <= sourceShortEdge; });
  if ((levels[selected('upscale_resolution')] || 0) <= sourceShortEdge) {
    const next = buttons.find(button => !button.disabled)?.dataset.upscaleResolution;
    if (next) {
      $('[name="upscale_resolution"]').value = next;
      buttons.forEach(button => button.classList.toggle('active', button.dataset.upscaleResolution === next));
    }
  }
  $('#upscaleStateText').textContent = !available ? '不可用' : !enabled ? '关闭' : `开启 · ${selected('upscale_resolution')}`;
  const target = resolvedUpscaleGeometry();
  const deliveryText = $('#deliveryText');
  if (deliveryText) deliveryText.textContent = `${target.width} × ${target.height}`;
  const hint = $('#upscaleHint');
  if (hint) hint.textContent = !available
    ? 'FlashVSR 源码或权重未安装完整，请运行项目安装脚本。'
    : `输出 ${target.width} × ${target.height}；H3仍按 ${source.width} × ${source.height} 生成，超分耗时单独记录。`;
  updateSettingsSummaries();
}

function updateSettingsSummaries() {
  const inference = $('#inferenceSummary');
  if (inference) {
    const variant = currentVariant() === 'lora' ? 'LoRA' : 'INT8';
    const steps = Number(selected('sampling_steps')) || (currentVariant() === 'lora' ? 8 : 20);
    const acceleration = Number(selected('acceleration')) || 0;
    const checkpoint = selected('execution_mode') === 'checkpoint'
      ? ` · 断点@${Number(selected('checkpoint_step')) || 1}` : '';
    inference.textContent = `${variant} · ${steps}步 · ${acceleration ? `加速${acceleration}` : 'Dense'}${checkpoint}`;
  }
  const video = $('#videoSettingsSummary');
  if (video && options) {
    const geometry = currentGeometry();
    const source = selected('size_mode') === 'custom'
      ? `自定义 ${geometry.width}×${geometry.height}`
      : `${String(selected('resolution')).toUpperCase()} · ${selected('aspect_ratio')}`;
    const upscale = $('#upscaleEnabled').checked ? ` · 超分${String(selected('upscale_resolution')).toUpperCase()}` : ' · 不超分';
    const duration = promptEditorMode === 'freeform' ? ` · ${Number(selected('duration_seconds')).toFixed(1)}秒` : '';
    video.textContent = source + duration + upscale;
  }
}

function updateSizeModeControls({initialize=false}={}) {
  if (!options) return;
  const custom = selected('size_mode') === 'custom';
  const fields = $('#customSizeFields');
  fields.hidden = !custom;
  $$('.preset-size-field').forEach(field => { field.hidden = custom; });
  $$('[name="width"],[name="height"]').forEach(input => { input.disabled = !custom; });
  if (custom && initialize) {
    const preset = options.geometry[selected('resolution')][selected('aspect_ratio')];
    $('[name="width"]').value = preset.width;
    $('[name="height"]').value = preset.height;
  }
  $('#customWidthValue').textContent = `${selected('width')} px`;
  $('#customHeightValue').textContent = `${selected('height')} px`;
  syncStoryboardTiming();
  updateSettingsSummaries();
}

function updateJointAccelerationControls() {
  if (!options) return;
  const variant = currentVariant();
  const limits = options.advanced_limits?.sampling_steps?.[variant] || (variant === 'lora'
    ? {min:4, max:10, default:8}
    : {min:5, max:30, default:20});
  const steps = $('[name="sampling_steps"]');
  const acceleration = $('[name="acceleration"]');
  const available = Boolean(options.advanced_limits?.sparse_attention_available);
  const accelerationLimits = options.advanced_limits?.acceleration || {min:0, max:100, step:1};
  steps.min = String(limits.min);
  steps.max = String(limits.max);
  if (steps.dataset.variant !== variant) {
    steps.value = String(limits.default);
    steps.dataset.variant = variant;
  }
  steps.value = String(Math.max(limits.min, Math.min(limits.max, Number(steps.value) || limits.default)));
  steps.disabled = false;
  $('#samplingStepsValue').textContent = `${steps.value} 步`;
  $('#samplingStepsHint').textContent = variant === 'lora'
    ? Number(steps.value) > 8
      ? '超过8步未经LoRA质量校准；允许运行，但不保证质量随步数单调增加。'
      : 'LoRA 使用完整 Turbo 步；内部加速只分配逐步逐层注意力，不擅自加入预测步。'
    : '决定完整 σ 去噪轨迹长度；系统在这条轨迹内联合安排真实步和预测步。';
  if (!available) {
    acceleration.value = '0';
    acceleration.min = '0';
    acceleration.max = '0';
    acceleration.disabled = true;
    $('#accelerationAdvanced').classList.add('unavailable');
    $('#accelerationSafety').textContent = '当前服务未安装 SM89 稀疏运行时，只能使用 0（Dense）；请运行项目安装脚本。';
  } else {
    acceleration.min = String(accelerationLimits.min);
    acceleration.max = String(accelerationLimits.max);
    acceleration.step = String(accelerationLimits.step);
    acceleration.disabled = false;
    $('#accelerationAdvanced').classList.remove('unavailable');
    const scheduler = accelerationLimits.scheduler_by_variant?.[variant]
      || accelerationLimits.scheduler;
    const certified = scheduler === 'v19_certified_frontier';
    $('#accelerationSafety').textContent = variant === 'lora'
      ? 'LoRA 无预测调度：全部 Turbo 步保持真实计算，档位只改变逐步逐层 Attention 配额。'
      : certified
        ? 'V19认证前沿：仅命中已封存工作负载时加速；其他输入自动Dense回退。'
        : '冻结的 Round229 调度：构图锚点、因果层、预测后恢复步和末端细节保护始终开启。';
  }
  const level = Math.max(0, Math.min(100, Number(acceleration.value) || 0));
  $('#accelerationValue').textContent = level === 0 ? '0 · Dense' : `${level} / 100`;
  updateSettingsSummaries();
}

function updateContract() {
  if (!options || !currentEngine) return;
  const geometry = currentGeometry();
  const width = geometry.width;
  const height = geometry.height;
  const requested = Number(selected('duration_seconds')) || 5;
  const frames = framesForDuration(requested);
  const geometryText = $('#geometryText');
  const durationText = $('#durationText');
  if (geometryText) geometryText.textContent = `${width} × ${height}`;
  if (durationText) durationText.textContent = `${(frames / 24).toFixed(2)}秒 · ${frames}帧`;
  const first = $('[name="first_frame"]').files.length > 0;
  const last = $('[name="last_frame"]').files.length > 0;
  const references = $('#referenceImages')?.files?.length || 0;
  const referenceVideos = $('#referenceVideos')?.files?.length || 0;
  const referenceAudios = $('#referenceAudios')?.files?.length || 0;
  const conditionText = $('#conditionText');
  if (conditionText) conditionText.textContent = currentEngine === 'reference' ? `多参考生视频 · ${references}图 / ${referenceVideos}视频 / ${referenceAudios}音频` : first && last ? '首尾帧生视频' : first ? '首帧生视频' : last ? '尾帧生视频' : '文生视频';
  updateUpscaleControls();
  updateSettingsSummaries();
}

function solverStepCount() {
  return Number(selected('sampling_steps')) || (currentVariant() === 'lora' ? 8 : 20);
}

function updateCheckpointControls() {
  const panel = $('#checkpointSettings');
  if (!panel) return;
  const enabled = selected('execution_mode') === 'checkpoint';
  const preview = $('#checkpointPreview').checked;
  panel.hidden = !enabled;
  const total = solverStepCount();
  const step = $('#checkpointStep');
  step.max = String(Math.max(1, total - 1));
  step.value = String(Math.max(1, Math.min(Number(step.value) || Math.ceil(total * .4), total - 1)));
  $('#checkpointStepValue').textContent = `第 ${step.value} 步`;
  step.disabled = !enabled;
  $('#checkpointRetain').disabled = !enabled;
  $('#checkpointPreview').disabled = !enabled;
  $$('.checkpoint-preview-option input, .checkpoint-preview-option select').forEach(input => {
    input.disabled = !enabled || !preview;
  });
  const previewSteps = $('[name="checkpoint_preview_steps"]');
  $('#checkpointPreviewStepsValue').textContent = `${previewSteps.value} 步`;
  updateSettingsSummaries();
}

function setUpscaleResolution(value) {
  const resolution = ['720p', '1080p', '2k'].includes(value) ? value : '720p';
  $('[name="upscale_resolution"]').value = resolution;
  $$('.upscale-resolution-tabs button').forEach(button => button.classList.toggle('active', button.dataset.upscaleResolution === resolution));
  updateUpscaleControls();
}

function applyEngineIdentity() {
  currentEngine = options.current_engine;
  const info = options.current_engine_options;
  if (!currentEngine || !info) return;
  const reference = currentEngine === 'reference';
  const lora = currentVariant() === 'lora';
  $('#engineBanner').dataset.engine = currentEngine;
  $('#engineIcon').textContent = reference ? 'R' : 'F';
  $('#engineIcon').className = `engine-icon ${lora ? 'turbo' : 'original'}`;
  $('#engineName').textContent = info.label;
  $('#engineBadge').textContent = lora ? 'LoRA 极速' : '原始权重';
  $('#engineDescription').textContent = reference ? `Ref2VA 多参考 · ${lora ? 'LoRA 六步极速路线' : '原始权重高保真路线'}` : `FL2VA / 文生视频 · ${lora ? 'LoRA 六步极速路线' : '原始权重高保真路线'}`;
  $('#brandSubtitle').textContent = reference ? 'Native reference generation' : 'Native first/last generation';
  $('#keyframeFieldset').hidden = reference;
  $('#referenceFieldset').hidden = !reference;
  $('#referencePromptBlock').hidden = !reference;
  $('#referenceVisualOverview').hidden = !reference;
  if (reference && enhancedReferenceProtocol) populateReferenceEditors(enhancedReferenceProtocol);
  renderReferenceMediaPolicy();
  renderShots();
  if (!reference) $$('.reference-mention-menu').forEach(menu => { menu.hidden = true; });
  updateJointAccelerationControls();
  updateCheckpointControls();
}

function bindDropzone(id) {
  const zone = $(id), input = $('input', zone), image = $('img', zone);
  input.addEventListener('change', () => {
    if (!input.files[0]) return;
    invalidatePromptEnhancement();
    image.src = URL.createObjectURL(input.files[0]); zone.classList.add('has-image'); compileStoryboard(); updateContract();
  });
  $('.remove-frame', zone).addEventListener('click', event => {
    event.preventDefault(); event.stopPropagation(); invalidatePromptEnhancement(); input.value = ''; image.removeAttribute('src'); zone.classList.remove('has-image'); compileStoryboard(); updateContract();
  });
  bindFileDrop(zone, input, {multiple:false, accept:file => file.type.startsWith('image/')});
}

function bindFileDrop(zone, input, {multiple=true, maxFiles=Infinity, accept=()=>true}={}) {
  const activate = event => { event.preventDefault(); event.stopPropagation(); zone.classList.add('drag-active'); };
  const deactivate = event => { event.preventDefault(); event.stopPropagation(); zone.classList.remove('drag-active'); };
  zone.addEventListener('dragenter', activate);
  zone.addEventListener('dragover', activate);
  zone.addEventListener('dragleave', deactivate);
  zone.addEventListener('drop', event => {
    deactivate(event);
    const files = Array.from(event.dataTransfer?.files || []).filter(accept);
    if (!files.length) return;
    const transfer = new DataTransfer();
    const existing = multiple ? Array.from(input.files || []) : [];
    const identities = new Set(existing.map(file => `${file.name}:${file.size}:${file.lastModified}`));
    const appended = files.filter(file => {
      const identity = `${file.name}:${file.size}:${file.lastModified}`;
      if (identities.has(identity)) return false;
      identities.add(identity); return true;
    });
    (multiple ? [...existing, ...appended].slice(0, maxFiles) : files.slice(0, 1)).forEach(file => transfer.items.add(file));
    input.files = transfer.files;
    input.dispatchEvent(new Event('change', {bubbles:true}));
  });
}

function appendFiles(input, files, maxFiles) {
  const transfer = new DataTransfer();
  const existing = Array.from(input.files || []);
  const identities = new Set(existing.map(file => `${file.name}:${file.size}:${file.lastModified}`));
  const appended = Array.from(files || []).filter(file => {
    const identity = `${file.name}:${file.size}:${file.lastModified}`;
    if (identities.has(identity)) return false;
    identities.add(identity); return true;
  });
  [...existing, ...appended].slice(0, maxFiles).forEach(file => transfer.items.add(file));
  input.files = transfer.files;
  input.dispatchEvent(new Event('change', {bubbles:true}));
}

function distributeReferenceFiles(files) {
  const all = Array.from(files || []);
  appendFiles($('#referenceImages'), all.filter(file => file.type.startsWith('image/')), 9);
  appendFiles($('#referenceVideos'), all.filter(file => file.type.startsWith('video/') || /\.(mp4|mov|mkv|webm|avi)$/i.test(file.name)), 3);
  appendFiles($('#referenceAudios'), all.filter(file => file.type.startsWith('audio/') || /\.(wav|mp3|flac|m4a|ogg|opus)$/i.test(file.name)), 3);
  $('#referenceFiles').value = '';
}

function statusName(job) {
  return {queued:`等待 ${job.queue_position || ''}`,starting_backend:'准备模型',running:'生成中',checkpointed:'断点已保存',awaiting_preview:'等待抽卡决定',succeeded:'已完成',failed:'失败',cancelled:'已取消'}[job.status] || job.status;
}

function progressMarkup(job) {
  const progress = job.progress || {};
  const percent = job.status === 'succeeded' ? 100 : Math.max(0, Math.min(100, Number(progress.percent) || 0));
  if (!['starting_backend','running','queued','awaiting_preview'].includes(job.status)) return '';
  const etaValue = job.status === 'queued' ? progress.estimated_completion_seconds : progress.estimated_remaining_seconds;
  const eta = formatSeconds(etaValue);
  const etaLabel = job.status === 'queued' ? '预计完成（含排队）' : job.status === 'starting_backend' ? '模型就绪后预计生成' : '预计剩余';
  return `<div class="job-progress"><div class="progress-copy"><span>${escapeHtml(progress.detail || statusName(job))}</span><b>${percent.toFixed(0)}%</b></div><div class="progress-track"><i style="width:${percent}%"></i></div><small>${etaLabel} ${eta}</small></div>`;
}

function advancedSummary(req) {
  if (req.sampling_steps != null && req.acceleration != null) {
    const acceleration = Number(req.acceleration);
    return `${req.sampling_steps}总步 · ${acceleration === 0 ? 'Dense' : `加速 ${acceleration}`}`;
  }
  if (!req.advanced) return req.quality || '默认计算';
  const compute = req.model_variant === 'base'
    ? `${req.actual_steps}实际/${req.forecast_steps}预测`
    : `${req.lora_steps}步`;
  const attention = Number(req.attention_keep_ratio) >= 1
    ? '完整注意力'
    : `${Math.round(Number(req.attention_keep_ratio) * 100)}%注意力 · ${{full:'全程固定',guarded:'动态保护',middle_only:'仅中段'}[req.sparse_scope] || req.sparse_scope}`;
  return `${compute} · ${attention}`;
}

function jobCard(job, {draggable=false}={}) {
  const req = job.request;
  const canCancel = ['queued','starting_backend','running','awaiting_preview'].includes(job.status);
  const canDelete = !['starting_backend','running','awaiting_preview'].includes(job.status);
  const actions = [
    job.preview?.ready ? `<button data-view-preview="${job.id}">查看断点预览</button>` : '',
    job.checkpoint?.resume_available ? `<button data-resume="${job.id}">继续正式生成</button>` : '',
    job.status === 'awaiting_preview' ? `<button data-preview-continue="${job.id}">继续正式生成</button><button class="danger-action" data-preview-discard="${job.id}">放弃本次抽卡</button>` : '',
    job.status === 'succeeded' ? `<button data-view="${job.id}">预览与下载</button>` : '',
    canCancel ? `<button data-cancel="${job.id}">取消</button>` : '',
    canDelete ? `<button class="danger-action" data-delete="${job.id}">删除</button>` : '',
  ].join('');
  const elapsed = job.status === 'succeeded'
    ? `<div class="job-elapsed"><span>实际总耗时</span><strong>${formatElapsed(job.elapsed_seconds)}</strong><small>${job.upscale_elapsed_seconds != null ? `H3 ${formatElapsed(job.generation_elapsed_seconds)} · FlashVSR ${formatElapsed(job.upscale_elapsed_seconds)}${job.upscale_peak_allocated_mib != null ? ` · 超分峰值 ${(Number(job.upscale_peak_allocated_mib) / 1024).toFixed(2)}GiB` : ''}` : '不含服务启动与模型预加载'}</small></div>`
    : '';
  const delivery = req.upscale_enabled ? ` → ${req.upscale_target_width}×${req.upscale_target_height} FlashVSR` : '';
  return `<article class="job manager-job ${draggable ? 'draggable' : ''}" data-job-id="${job.id}" ${draggable ? 'draggable="true"' : ''}>
    <div class="job-top"><div class="job-main">${draggable ? '<span class="drag-handle" title="拖动排序">⠿</span>' : ''}<div><div class="job-title">${escapeHtml(req.prompt)}</div><div class="job-meta">${req.width}×${req.height}${delivery} · ${req.actual_duration_seconds.toFixed(2)}秒 · ${escapeHtml(advancedSummary(req))} · Seed ${req.seed}</div></div></div><span class="status ${job.status}">${statusName(job)}</span></div>
    ${progressMarkup(job)}${elapsed}${job.error ? `<div class="job-error">${escapeHtml(job.error)}</div>` : ''}${actions ? `<div class="job-actions">${actions}</div>` : ''}
  </article>`;
}

function promptPreview(prompt) {
  const body = String(prompt || '')
    .replace(/^(integrated_multimodal_description|overall_soundscape|non_diegetic_music):\s*/gmi, '')
    .replace(/\[Shot \d+\](?: At [^,]+,)?/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  return body.length > 420 ? `${body.slice(0, 420)}…` : body;
}

function conversationItem(job) {
  const req = job.request;
  const canCancel = ['queued','starting_backend','running','awaiting_preview'].includes(job.status);
  const canDelete = !['starting_backend','running','awaiting_preview'].includes(job.status);
  const completed = job.status === 'succeeded';
  const checkpointed = job.status === 'checkpointed';
  const output = completed
    ? `<div class="conversation-result"><button class="conversation-video-placeholder" data-view="${job.id}" aria-label="打开成片预览"><span>▶</span><small>点击预览成片</small></button><div><strong>视频生成完成</strong><span>${req.width}×${req.height} · ${req.actual_duration_seconds.toFixed(2)}秒 · ${formatElapsed(job.elapsed_seconds)}</span><div class="conversation-actions"><button data-view="${job.id}">打开预览与下载</button></div></div></div>`
    : checkpointed
      ? `<div class="conversation-response-state"><span class="conversation-spinner stopped"></span><div><strong>已在第 ${job.checkpoint?.completed_steps || '?'} / ${job.checkpoint?.total_steps || '?'} 步停止</strong><small>正式状态已落盘，当前任务不占用 GPU；恢复时重新进入队列。</small></div></div>`
    : `<div class="conversation-response-state"><span class="conversation-spinner ${['failed','cancelled'].includes(job.status) ? 'stopped' : ''}"></span><div><strong>${escapeHtml(statusName(job))}</strong><small>${escapeHtml(job.progress?.detail || '')}</small></div></div>`;
  return `<article class="conversation-turn" data-conversation-job="${job.id}">
    <div class="conversation-user"><div class="bubble-label">你提交的视频任务</div><p>${escapeHtml(promptPreview(req.prompt))}</p><small>${req.width}×${req.height} · ${req.actual_duration_seconds.toFixed(2)}秒 · Seed ${req.seed}</small></div>
    <div class="conversation-assistant"><div class="bubble-label">H3 · ${req.model_variant === 'lora' ? 'LoRA 极速' : '原始权重'}</div>${output}${job.preview?.ready || job.checkpoint?.resume_available ? `<div class="conversation-actions">${job.preview?.ready ? `<button data-view-preview="${job.id}">查看断点预览</button>` : ''}${job.checkpoint?.resume_available ? `<button data-resume="${job.id}">继续正式生成</button>` : ''}${job.status === 'awaiting_preview' ? `<button data-preview-continue="${job.id}">继续正式生成</button><button data-preview-discard="${job.id}">放弃抽卡</button>` : ''}</div>` : ''}${progressMarkup(job)}${job.error ? `<div class="job-error">${escapeHtml(job.error)}</div>` : ''}<div class="conversation-record-actions">${canCancel ? `<button data-cancel="${job.id}">取消任务</button>` : ''}${canDelete ? `<button class="danger-action" data-delete="${job.id}">删除这条创作记录</button>` : ''}</div></div>
  </article>`;
}

function renderConversation() {
  const feed = $('#conversationFeed');
  if (!feed) return;
  const ordered = [...jobs].sort((a, b) => Number(a.created_at) - Number(b.created_at));
  feed.innerHTML = ordered.length
    ? ordered.map(conversationItem).join('')
    : '<div class="conversation-welcome"><b>从底部开始创建第一条视频</b><span>提交后，排队、生成进度、预计完成时间和成片都会显示在这里。</span></div>';
}

function empty(message) { return `<div class="manager-empty">${message}</div>`; }

function renderJobs() {
  const running = jobs.filter(job => ['starting_backend','running','awaiting_preview'].includes(job.status));
  const queued = jobs.filter(job => job.status === 'queued').sort((a,b) => (a.queue_position || 999) - (b.queue_position || 999));
  const history = jobs.filter(job => !['queued','starting_backend','running','awaiting_preview'].includes(job.status));
  $('#runningCount').textContent = running.length;
  $('#queuedCount').textContent = queued.length;
  $('#completedCount').textContent = history.filter(job => job.status === 'succeeded').length;
  $('#navTaskCount').textContent = running.length + queued.length;
  $('#runningJobs').innerHTML = running.length ? running.map(job => jobCard(job)).join('') : empty('当前没有正在执行的任务');
  $('#queuedJobs').innerHTML = queued.length ? queued.map(job => jobCard(job, {draggable:true})).join('') : empty('等待队列为空');
  $('#historyJobs').innerHTML = history.length ? history.map(job => jobCard(job)).join('') : empty('还没有历史任务');
  renderConversation();
  bindJobActions(); bindDragAndDrop();
}

function bindJobActions() {
  $$('[data-cancel]').forEach(button => button.addEventListener('click', () => cancelJob(button.dataset.cancel)));
  $$('[data-delete]').forEach(button => button.addEventListener('click', () => deleteJob(button.dataset.delete)));
  $$('[data-view]').forEach(button => button.addEventListener('click', () => showVideo(button.dataset.view)));
  $$('[data-view-preview]').forEach(button => button.addEventListener('click', () => showPreview(button.dataset.viewPreview)));
  $$('[data-resume]').forEach(button => button.addEventListener('click', () => resumeJob(button.dataset.resume)));
  $$('[data-preview-continue]').forEach(button => button.addEventListener('click', () => decidePreview(button.dataset.previewContinue, 'continue')));
  $$('[data-preview-discard]').forEach(button => button.addEventListener('click', () => decidePreview(button.dataset.previewDiscard, 'discard')));
}

function bindDragAndDrop() {
  $$('#queuedJobs [draggable="true"]').forEach(card => {
    card.addEventListener('dragstart', event => { draggedJobId = card.dataset.jobId; card.classList.add('dragging'); event.dataTransfer.effectAllowed = 'move'; });
    card.addEventListener('dragend', () => { card.classList.remove('dragging'); draggedJobId = null; });
    card.addEventListener('dragover', event => { event.preventDefault(); const dragged = $(`#queuedJobs [data-job-id="${draggedJobId}"]`); if (!dragged || dragged === card) return; const rect = card.getBoundingClientRect(); card.parentElement.insertBefore(dragged, event.clientY < rect.top + rect.height / 2 ? card : card.nextSibling); });
    card.addEventListener('drop', async event => { event.preventDefault(); await saveQueueOrder(); });
  });
}

async function saveQueueOrder() {
  const jobIds = $$('#queuedJobs [data-job-id]').map(card => card.dataset.jobId);
  try { await api('/api/v1/jobs/order', {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify({job_ids:jobIds})}); await refreshJobs(); }
  catch (error) { alert(`调整顺序失败：${error.message}`); await refreshJobs(); }
}

async function refreshJobs() {
  try { jobs = (await (await api('/api/v1/jobs?limit=100')).json()).jobs; renderJobs(); }
  catch (error) { $('#healthText').textContent = error.message; }
}

async function checkHealth() {
  try {
    const response = await fetch('/healthz'); if (!response.ok) throw new Error('offline'); const health = await response.json();
    $('.server-state').className = 'server-state online';
    const warm = health.warm_state?.status || 'unknown';
    const warmName = {cold:'未加载',loading:'加载中',ready:'已热身',failed:'加载失败',unsupported:'可用'}[warm] || warm;
    $('#warmStateText').textContent = warmName;
    const engineLabel = health.active_engine === 'reference' || health.active_engine === 'reference_lora'
      ? 'Ref2VA' : health.active_engine === 'first_last' || health.active_engine === 'original' || health.active_engine === 'lora'
        ? 'FL2VA' : '待选择模式';
    $('#healthText').textContent = health.engine_control?.switching ? '在线 · 正在切换引擎' : warm === 'loading' ? '在线 · 正在预加载模型' : warm === 'failed' ? '在线 · 模型加载失败' : `在线 · ${engineLabel} · ${warmName}`;
  } catch (_) { $('.server-state').className = 'server-state offline'; $('#healthText').textContent = '服务不可用'; $('#warmStateText').textContent = '离线'; }
}

function setResourceBar(id, percent) {
  const bar = $(id); if (bar) bar.style.width = `${Math.max(0, Math.min(100, Number(percent) || 0))}%`;
}

async function refreshResources() {
  if (!currentEngine || activePage !== 'tasks') return;
  try {
    const data = await (await api('/api/v1/resources')).json();
    $('#cpuUsage').textContent = `${data.cpu.utilization_percent.toFixed(0)}%`;
    $('#cpuDetail').textContent = `${data.cpu.logical_cores} 线程 · 负载 ${data.cpu.load_1m}`;
    setResourceBar('#cpuBar', data.cpu.utilization_percent);
    $('#memoryUsage').textContent = `${data.memory.used_gib.toFixed(1)} / ${data.memory.total_gib.toFixed(1)} GiB`;
    $('#memoryDetail').textContent = `${data.memory.percent.toFixed(0)}% · 服务进程 ${data.process.rss_gib.toFixed(1)} GiB`;
    setResourceBar('#memoryBar', data.memory.percent);
    if (data.gpu) {
      $('#gpuUsage').textContent = `${data.gpu.utilization_percent.toFixed(0)}%`;
      $('#gpuDetail').textContent = `${data.gpu.name} · ${data.gpu.temperature_c.toFixed(0)}°C · ${data.gpu.power_w.toFixed(0)}W`;
      setResourceBar('#gpuBar', data.gpu.utilization_percent);
      $('#vramUsage').textContent = `${data.gpu.memory_used_gib.toFixed(1)} / ${data.gpu.memory_total_gib.toFixed(1)} GiB`;
      $('#vramDetail').textContent = `${data.gpu.memory_percent.toFixed(0)}% 已使用`;
      setResourceBar('#vramBar', data.gpu.memory_percent);
    } else {
      $('#gpuUsage').textContent = '不可用'; $('#gpuDetail').textContent = '未检测到 NVIDIA 监控接口';
      $('#vramUsage').textContent = '不可用'; $('#vramDetail').textContent = 'nvidia-smi 未就绪';
    }
  } catch (_) {}
}

async function cancelJob(id) { try { await api(`/api/v1/jobs/${id}`, {method:'DELETE'}); await refreshJobs(); } catch (error) { alert(error.message); } }
async function deleteJob(id) { if (!confirm('删除该任务记录、上传帧和已生成视频？此操作不可恢复。')) return; try { await api(`/api/v1/jobs/${id}/record`, {method:'DELETE'}); await refreshJobs(); } catch (error) { alert(error.message); } }
async function showVideo(id) { try { const blob = await (await api(`/api/v1/jobs/${id}/video`)).blob(); if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl); videoObjectUrl = URL.createObjectURL(blob); $('#resultVideo').src = videoObjectUrl; $('#downloadVideo').href = videoObjectUrl; $('#downloadVideo').download = `h3-${id}.mp4`; $('#videoDialog').showModal(); } catch (error) { alert(error.message); } }
async function showPreview(id) { try { const blob = await (await api(`/api/v1/jobs/${id}/preview`)).blob(); if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl); videoObjectUrl = URL.createObjectURL(blob); $('#resultVideo').src = videoObjectUrl; $('#downloadVideo').href = videoObjectUrl; $('#downloadVideo').download = `h3-${id}-preview.mp4`; $('#videoDialog').showModal(); } catch (error) { alert(error.message); } }
async function decidePreview(id, decision) { try { await api(`/api/v1/jobs/${id}/preview/${decision}`, {method:'POST'}); await refreshJobs(); } catch (error) { alert(error.message); } }
async function resumeJob(id) { try { await api(`/api/v1/jobs/${id}/resume`, {method:'POST'}); await refreshJobs(); } catch (error) { alert(error.message); } }

async function submit(event) {
  event.preventDefault(); const button = $('.submit-button', event.target), message = $('#formMessage'); button.disabled = true; message.hidden = true;
  try {
    if (!currentEngine) throw new Error('请先选择生成模式');
    const maximumDuration = currentMaxDuration();
    if (promptEditorMode === 'structured') {
      synchronizeDurationInputs();
      const total = shotTotal();
      if (total < 1 || total > maximumDuration) {
        throw new Error(`当前分辨率的分镜总时长必须在 1–${maximumDuration} 秒之间`);
      }
      if (Math.abs(total - resolvedVideoDuration()) > 0.4) throw new Error(`分镜总时长 ${total.toFixed(1)} 秒与实际视频时长 ${resolvedVideoDuration().toFixed(2)} 秒不一致，请检查分镜时长`);
      if (shots.some(shot => !shot.prompt.trim())) throw new Error('请填写每个分镜的镜头内容');
      if (currentEngine === 'reference' && !$('#referenceDefinitions').value.trim()) throw new Error('请填写对象定义（subject_definitions）');
      if (currentEngine === 'reference' && !$('#referenceRetention').value.trim()) throw new Error('请填写保留规则（retention_analysis）');
      if (currentEngine === 'reference' && !$('#referenceSummary').value.trim()) throw new Error('请填写总体摘要（summary）');
      if ($('#bgmEnabled').checked && !$('#bgmStyle').value.trim()) throw new Error('已开启BGM，请填写配乐风格');
    } else {
      const requested = Number(selected('duration_seconds')) || 0;
      if (requested < 1 || requested > maximumDuration) throw new Error(`当前分辨率的视频时长必须在 1–${maximumDuration} 秒之间`);
      if (!$('#freeformPrompt').value.trim()) throw new Error('请输入完整提示词');
    }
    compileStoryboard();
    const form = new FormData(event.target); form.delete('engine');
    form.set('service_family', currentEngine);
    form.set('model_variant', currentVariant());
    if (selected('execution_mode') === 'checkpoint') {
      form.set('checkpoint_retain', String($('#checkpointRetain').checked));
      form.set('checkpoint_preview', String($('#checkpointPreview').checked));
    } else {
      ['checkpoint_step','checkpoint_retain','checkpoint_preview','checkpoint_preview_steps','checkpoint_preview_resolution'].forEach(name => form.delete(name));
    }
    if (currentEngine === 'reference') {
      const files = Array.from($('#referenceImages').files || []);
      const videos = Array.from($('#referenceVideos').files || []);
      const audios = Array.from($('#referenceAudios').files || []);
      if (!files.length && !videos.length && !audios.length) throw new Error('请至少选择一项参考图片、视频或音频');
      if (files.length > 9) throw new Error('参考图片最多9张');
      if (videos.length > 3) throw new Error('参考视频最多3段');
      if (audios.length > 3) throw new Error('参考音频最多3段');
      if (promptEditorMode === 'structured') {
        const completePrompt = $('#compiledPrompt').value;
        audios.forEach((_, index) => {
          const label = `<Audio ${index + 1}>`;
          if (!completePrompt.toLowerCase().includes(label.toLowerCase())) {
            throw new Error(`参考音频 ${index + 1} 尚未使用：请在人物台词、画面内声音或BGM中引用 ${label}`);
          }
        });
      }
      files.forEach((file, index) => form.set(`reference_image_${index + 1}`, file));
      videos.forEach((file, index) => form.set(`reference_video_${index + 1}`, file));
      audios.forEach((file, index) => form.set(`reference_audio_${index + 1}`, file));
    }
    const customSize = selected('size_mode') === 'custom';
    form.set('mode', customSize ? 'advanced' : 'preset');
    form.set('advanced', String(customSize));
    if (!customSize) ['width','height'].forEach(name => form.delete(name));
    ['frames','advanced_seed','actual_steps','lora_steps','attention_keep_ratio','sparse_scope'].forEach(name => form.delete(name));
    for (const role of ['first_frame','last_frame']) if (!$(`[name="${role}"]`).files[0]) form.delete(role);
    const job = await (await api('/api/v1/generations', {method:'POST', body:form})).json(); jobs.push(job); renderJobs(); message.textContent = `任务 ${job.id.slice(0,8)} 已加入队列`; message.style.background = 'rgba(86,214,160,.1)'; message.style.color = '#8ce9bd'; message.hidden = false; $('#conversationFeed').lastElementChild?.scrollIntoView({behavior:'smooth', block:'center'});
  } catch (error) { message.textContent = error.message; message.style.background = ''; message.style.color = ''; message.hidden = false; }
  finally { button.disabled = false; }
}

async function boot() {
  addShot({duration_seconds:5});
  bindDropzone('#firstDrop'); bindDropzone('#lastDrop'); $('#generationForm').addEventListener('submit', submit);
  $$('[data-prompt-mode]').forEach(button => button.addEventListener('click', () => setPromptEditorMode(button.dataset.promptMode)));
  $('#freeformPrompt').addEventListener('input', event => { updateReferenceMentionMenu($('#freeformPromptEditor'), event.target); compileStoryboard(); });
  $('#freeformPrompt').addEventListener('keydown', event => { if (event.key === 'Escape') $('.reference-mention-menu', $('#freeformPromptEditor')).hidden = true; });
  $('#freeformPrompt').addEventListener('blur', () => setTimeout(() => { $('.reference-mention-menu', $('#freeformPromptEditor')).hidden = true; }, 120));
  $('[name="model_variant"]').addEventListener('change', () => { applyEngineIdentity(); updateContract(); });
  $('#executionMode').addEventListener('change', updateCheckpointControls);
  $('#checkpointPreview').addEventListener('change', updateCheckpointControls);
  $('#checkpointStep').addEventListener('input', updateSettingsSummaries);
  $('#referenceFiles').addEventListener('change', event => distributeReferenceFiles(event.target.files));
  bindFileDrop($('#referenceFileDrop'), $('#referenceFiles'), {maxFiles:15, accept:file => file.type.startsWith('image/') || file.type.startsWith('video/') || file.type.startsWith('audio/') || /\.(mp4|mov|mkv|webm|avi|wav|mp3|flac|m4a|ogg|opus)$/i.test(file.name)});
  $('#referenceImages').addEventListener('change', event => {
    referenceRoles.image = Array.from(event.target.files || []).slice(0, 9).map((_, index) => event.isTrusted ? '' : (referenceRoles.image[index] || ''));
    renderReferencePreviews('image');
    resetReferenceEditors();
    updateContract();
  });
  $('#referenceVideos').addEventListener('change', event => {
    referenceRoles.video = Array.from(event.target.files || []).slice(0, 3).map((_, index) => event.isTrusted ? '' : (referenceRoles.video[index] || ''));
    renderReferencePreviews('video');
    resetReferenceEditors();
    updateContract();
  });
  $('#referenceAudios').addEventListener('change', event => {
    referenceRoles.audio = Array.from(event.target.files || []).slice(0, 3).map((_, index) => event.isTrusted ? '' : (referenceRoles.audio[index] || ''));
    renderReferencePreviews('audio');
    resetReferenceEditors();
    updateContract();
  });
  $('#exitEngine').addEventListener('click', exitEngine);
  $('#chooseWorkspace').addEventListener('click', openWorkspaceDialog);
  $('#openWorkspacePath').addEventListener('click', () => browseWorkspace($('#workspacePathInput').value.trim()).catch(showWorkspaceError));
  $('#workspacePathInput').addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); browseWorkspace(event.target.value.trim()).catch(showWorkspaceError); } });
  $('#workspaceParent').addEventListener('click', () => { if (workspaceBrowseParent) browseWorkspace(workspaceBrowseParent).catch(showWorkspaceError); });
  $('#workspaceDefault').addEventListener('click', () => browseWorkspace(options.workspace.default_path).catch(showWorkspaceError));
  $('#selectWorkspace').addEventListener('click', activateWorkspace);
  $('#addShot').addEventListener('click', () => {
    const maximum = currentMaxDuration();
    const remaining = maximum - shotTotal();
    if (remaining < .5) {
      $('#enhancementMessage').textContent = `当前分辨率的分镜总时长已达到${maximum}秒，请先缩短现有分镜。`;
      $('#enhancementMessage').className = 'enhancement-message';
      $('#enhancementMessage').hidden = false;
      return;
    }
    addShot({duration_seconds:Math.min(3, remaining)});
  });
  $('#enhanceReferences').addEventListener('click', () => enhanceSection('references'));
  $('#enhanceVisuals').addEventListener('click', () => enhanceSection('visuals'));
  $('#enhanceSound').addEventListener('click', () => enhanceSection('sound'));
  $('#undoReferenceEnhancement').addEventListener('click', () => undoEnhancement('references'));
  $('#undoVisualEnhancement').addEventListener('click', () => undoEnhancement('visuals'));
  $('#undoSoundEnhancement').addEventListener('click', () => undoEnhancement('sound'));
  ['referenceDefinitions','referenceRetention','referenceSummary'].forEach(id => {
    const textarea = $(`#${id}`), root = textarea.parentElement;
    textarea.addEventListener('input', () => { enhancedReferenceProtocol = currentReferenceProtocol(); invalidatePromptEnhancement(id === 'referenceSummary' ? 'visuals' : 'references'); updateReferenceMentionMenu(root, textarea); compileStoryboard(); });
    textarea.addEventListener('keydown', event => { if (event.key === 'Escape') $('.reference-mention-menu', root).hidden = true; });
    textarea.addEventListener('blur', () => setTimeout(() => { $('.reference-mention-menu', root).hidden = true; }, 120));
  });
  $('#overallSoundscape').addEventListener('input', event => { enhancedSoundscape = event.target.value; invalidatePromptEnhancement('sound'); updateReferenceMentionMenu(event.target.parentElement, event.target); compileStoryboard(); });
  $('#bgmEnabled').addEventListener('change', event => { invalidatePromptEnhancement('sound'); $('#bgmStyleField').hidden = !event.target.checked; $('#bgmStateText').textContent = event.target.checked ? '开启' : '关闭 · N/A'; compileStoryboard(); });
  $('#bgmStyle').addEventListener('input', event => { invalidatePromptEnhancement('sound'); updateReferenceMentionMenu(event.target.parentElement, event.target); compileStoryboard(); });
  ['overallSoundscape','bgmStyle'].forEach(id => {
    const textarea = $(`#${id}`), root = textarea.parentElement;
    textarea.addEventListener('keydown', event => { if (event.key === 'Escape') $('.reference-mention-menu', root).hidden = true; });
    textarea.addEventListener('blur', () => setTimeout(() => { $('.reference-mention-menu', root).hidden = true; }, 120));
  });
  $('#upscaleEnabled').addEventListener('change', updateUpscaleControls);
  $$('.upscale-resolution-tabs button').forEach(button => button.addEventListener('click', () => setUpscaleResolution(button.dataset.upscaleResolution)));
  $('[name="upscale_resolution"]').addEventListener('change', () => setUpscaleResolution(selected('upscale_resolution')));
  $$('.workspace-tabs button').forEach(button => button.addEventListener('click', () => switchPage(button.dataset.page)));
  $('#sizeMode').addEventListener('change', () => updateSizeModeControls({initialize:true}));
  $$('[name="resolution"],[name="aspect_ratio"]').forEach(element => element.addEventListener('input', syncStoryboardTiming));
  $('[name="duration_seconds"]').addEventListener('input', () => {
    $('#freeformDurationValue').textContent = `${Number(selected('duration_seconds')).toFixed(1)} 秒`;
    updateContract();
  });
  $$('[name="width"],[name="height"]').forEach(element => element.addEventListener('input', () => updateSizeModeControls()));
  $('[name="sampling_steps"]').addEventListener('input', () => { updateJointAccelerationControls(); updateCheckpointControls(); });
  $('[name="acceleration"]').addEventListener('input', updateJointAccelerationControls);
  $('#checkpointStep').addEventListener('input', updateCheckpointControls);
  $('[name="checkpoint_preview_steps"]').addEventListener('input', updateCheckpointControls);
  $('#refreshJobs').addEventListener('click', refreshJobs);
  $('#settingsButton').addEventListener('click', async () => { $('#apiKeyInput').value = localStorage.getItem('h3serve_api_key') || ''; $('#mimoApiKeyInput').value = ''; const [status, referencePolicy, generationPolicy] = await Promise.all([serverMimoKeyStatus().catch(() => ({configured:false})), serverReferenceMediaSettings().catch(() => null), serverGenerationLimitSettings().catch(() => null)]); $('#mimoKeyStatus').textContent = status.configured ? '服务器已保存MiMo Key；留空不会覆盖。' : '服务器尚未配置MiMo Key。'; if (referencePolicy) renderReferenceMediaPolicy(referencePolicy); if (generationPolicy) renderGenerationLimitPolicy(generationPolicy); renderMemoryProfiles(); $('#settingsDialog').showModal(); });
  $('#saveSettings').addEventListener('click', async event => { event.preventDefault(); const value = $('#apiKeyInput').value.trim(); value ? localStorage.setItem('h3serve_api_key', value) : localStorage.removeItem('h3serve_api_key'); const mimo = $('#mimoApiKeyInput').value.trim(); try { if (mimo) await configureServerMimoKey(mimo); const referencePolicy = await configureServerReferenceMedia($('#globalReferenceImageResolution').value, $('#globalReferenceVideoResolution').value); renderReferenceMediaPolicy(referencePolicy); const generationPolicy = await configureServerGenerationLimits(readGenerationLimitEditor()); renderGenerationLimitPolicy(generationPolicy); sessionStorage.removeItem('h3serve_mimo_api_key'); await applyMemoryProfile(); $('#settingsDialog').close(); } catch (error) { $('#formMessage').textContent = error.message; $('#formMessage').hidden = false; } setTimeout(() => { reloadOptions().catch(() => {}); bootData(); }, 0); });
  $('#clearMimoKey').addEventListener('click', async () => { try { await configureServerMimoKey(''); sessionStorage.removeItem('h3serve_mimo_api_key'); $('#mimoApiKeyInput').value = ''; $('#mimoKeyStatus').textContent = '服务器尚未配置MiMo Key。'; } catch (error) { $('#mimoKeyStatus').textContent = error.message; } });
  try { await reloadOptions(); }
  catch (error) { $('#formMessage').textContent = error.message; $('#formMessage').hidden = false; }
  const rememberedMimoKey = sessionStorage.getItem('h3serve_mimo_api_key') || '';
  if (rememberedMimoKey) { await configureServerMimoKey(rememberedMimoKey).catch(() => {}); sessionStorage.removeItem('h3serve_mimo_api_key'); }
  await bootData();
  setPromptEditorMode('structured');
  setInterval(() => { checkHealth(); refreshJobs(); reconcileEngineState().catch(() => {}); }, 1500);
  setInterval(refreshResources, 3000);
}

async function bootData() { await Promise.allSettled([checkHealth(), refreshJobs(), refreshResources()]); }
document.addEventListener('DOMContentLoaded', boot);
