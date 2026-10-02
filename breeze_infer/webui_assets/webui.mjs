const HISTORY_KEY = 'breeze-stt-recent-urls';

export function buildSpeechPayload({ model, voice, text, instructions, guidanceScale, seed, requestMode, streamFormat }) {
  const input = String(text || '').trim();
  if (!input || !/[\p{L}\p{N}]/u.test(input)) throw new Error('Enter text containing a letter or digit.');
  if (!model) throw new Error('Select a model.');
  const mode = requestMode || 'auto';
  const payload = { model, voice: voice || 'default', input, response_format: 'wav', stream: mode !== 'non-streaming' };
  if (instructions?.trim()) payload.instructions = instructions.trim();
  if (String(guidanceScale ?? '').trim()) {
    const scale = Number(guidanceScale);
    if (!Number.isFinite(scale) || scale <= 0) throw new Error('CFG scale must be greater than zero.');
    payload.guidance_scale = scale;
  }
  if (String(seed ?? '').trim()) {
    const value = Number(seed);
    if (!Number.isSafeInteger(value)) throw new Error('Seed must be an integer.');
    payload.seed = value;
  }
  if (payload.stream && streamFormat && streamFormat !== 'default') payload.stream_format = streamFormat;
  return payload;
}

export function joinBytes(chunks) {
  const size = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
  const result = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.byteLength; }
  return result;
}

export function pcmToWav(chunks, sampleRate) {
  const pcm = joinBytes(chunks);
  if (pcm.byteLength % 2) throw new Error('Incomplete PCM sample.');
  const result = new Uint8Array(44 + pcm.byteLength);
  const view = new DataView(result.buffer);
  const label = (offset, value) => { for (let i = 0; i < value.length; i++) result[offset + i] = value.charCodeAt(i); };
  label(0, 'RIFF'); view.setUint32(4, 36 + pcm.byteLength, true); label(8, 'WAVE');
  label(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); label(36, 'data'); view.setUint32(40, pcm.byteLength, true);
  result.set(pcm, 44);
  return result;
}

export function wavToPcm(bytes) {
  const source = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (source.byteLength < 44) throw new Error('WAV audio is too short.');
  const view = new DataView(source.buffer, source.byteOffset, source.byteLength);
  const label = offset => String.fromCharCode(...source.subarray(offset, offset + 4));
  if (label(0) !== 'RIFF' || label(8) !== 'WAVE') throw new Error('Invalid WAV audio.');
  let sampleRate = null, channels = null, depth = null, encoding = null, pcm = null;
  for (let offset = 12; offset + 8 <= source.byteLength;) {
    const size = view.getUint32(offset + 4, true);
    const start = offset + 8;
    if (start + size > source.byteLength) throw new Error('Incomplete WAV audio.');
    if (label(offset) === 'fmt ') {
      if (size < 16) throw new Error('Invalid WAV format chunk.');
      encoding = view.getUint16(start, true); channels = view.getUint16(start + 2, true);
      sampleRate = view.getUint32(start + 4, true); depth = view.getUint16(start + 14, true);
    } else if (label(offset) === 'data') pcm = source.slice(start, start + size);
    offset = start + size + (size % 2);
  }
  if (encoding !== 1 || channels !== 1 || depth !== 16 || !sampleRate || !pcm)
    throw new Error('Expected mono 16-bit PCM WAV audio.');
  return { pcm, sampleRate };
}

export function parseSseBlock(block) {
  const data = block.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
  if (!data) return null;
  if (data === '[DONE]') return { type: 'done_marker' };
  return JSON.parse(data);
}

function decodeBase64(value) {
  const binary = atob(value);
  const result = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) result[i] = binary.charCodeAt(i);
  return result;
}

export async function readSpeechStream(response, onPcm) {
  const contentType = (response.headers.get('content-type') || '').split(';')[0].toLowerCase();
  const reader = response.body.getReader();
  if (contentType === 'text/event-stream') {
    const decoder = new TextDecoder();
    let pending = '', sawDone = false, chunks = 0;
    const process = block => {
      const event = parseSseBlock(block);
      if (!event) return;
      if (event.type === 'error') throw new Error(event.message || event.error || 'Speech stream failed.');
      if (event.type === 'done' || event.type === 'done_marker') { sawDone = true; return; }
      if (event.type !== 'audio.chunk') return;
      const encoded = decodeBase64(event.data);
      const audio = event.format === 'wav' ? wavToPcm(encoded) : { pcm: encoded, sampleRate: Number(event.sample_rate) || 24000 };
      onPcm(audio.pcm, audio.sampleRate); chunks++;
    };
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending = (pending + decoder.decode(value, { stream: true })).replaceAll('\r\n', '\n');
      let boundary;
      while ((boundary = pending.indexOf('\n\n')) !== -1) {
        process(pending.slice(0, boundary)); pending = pending.slice(boundary + 2);
      }
    }
    pending += decoder.decode();
    if (pending.trim()) process(pending);
    if (!sawDone) throw new Error('Speech stream ended before completion.');
    if (!chunks) throw new Error('Speech stream contained no audio.');
    return;
  }

  const sampleRate = Number(response.headers.get('x-sample-rate') || response.headers.get('x-audio-sample-rate')) || 24000;
  if (contentType === 'audio/pcm' || contentType === 'application/octet-stream') {
    let odd = null, chunks = 0;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (!value?.length) continue;
      let bytes = value;
      if (odd !== null) { bytes = joinBytes([new Uint8Array([odd]), bytes]); odd = null; }
      if (bytes.length % 2) { odd = bytes[bytes.length - 1]; bytes = bytes.subarray(0, -1); }
      if (bytes.length) { onPcm(bytes, sampleRate); chunks++; }
    }
    if (odd !== null) throw new Error('Speech stream ended mid-sample.');
    if (!chunks) throw new Error('Speech stream contained no audio.');
    return;
  }

  const all = [];
  while (true) { const { value, done } = await reader.read(); if (done) break; if (value?.length) all.push(value); }
  const audio = wavToPcm(joinBytes(all));
  onPcm(audio.pcm, audio.sampleRate);
}

const $ = id => document.getElementById(id);
const state = { voices: [], outputUrl: null, previewUrl: null, controller: null, context: null, sources: new Set(), nextPlayTime: 0, transition: null, polling: false };

function show(id, message, tone = '') { const element = $(id); element.textContent = message; element.classList.remove('error', 'success'); if (tone) element.classList.add(tone); }
function formatMb(value) { return typeof value === 'number' ? `${value.toFixed(1)} MB` : 'Unavailable'; }

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (response.ok) return response;
  let detail = `HTTP ${response.status}`;
  try { const payload = await response.json(); detail = payload.detail || payload.error || detail; } catch { /* use HTTP status */ }
  throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
}

function recentSttUrls() {
  try { const values = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); return Array.isArray(values) ? values.filter(value => typeof value === 'string').slice(0, 20) : []; }
  catch { return []; }
}

function renderSttHistory() {
  const list = $('sttRecent'); list.replaceChildren();
  for (const url of recentSttUrls()) list.append(new Option('', url));
}

function rememberSttUrl() {
  const value = $('sttUrl').value.trim().replace(/\/+$/, '');
  if (!value) return;
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify([value, ...recentSttUrls().filter(item => item !== value)].slice(0, 20))); } catch { /* storage is optional */ }
  renderSttHistory();
}

async function refreshCatalog() {
  try {
    let response;
    try { response = await api('/v1/audio/models'); } catch { response = await api('/v1/models'); }
    const models = (await response.json()).data || [];
    const selected = $('modelSelect').value;
    $('modelSelect').replaceChildren(...models.filter(item => item?.id).map(item => new Option(item.id, item.id)));
    if (models.some(item => item.id === selected)) $('modelSelect').value = selected;
    await refreshVoices();
  } catch (error) { show('modelActionStatus', `Catalog refresh failed: ${error.message}`, 'error'); }
}

async function refreshVoices(preferred = null) {
  try {
    const response = await api('/v1/audio/voices');
    const payload = await response.json();
    state.voices = Array.isArray(payload.data) ? payload.data : [];
    const old = preferred || $('voiceSelect').value || 'default';
    const options = [new Option('Default', 'default')];
    for (const voice of state.voices) if (voice?.id && voice.id !== 'default') options.push(new Option(voice.name || voice.id, voice.id));
    $('voiceSelect').replaceChildren(...options);
    $('voiceSelect').value = options.some(option => option.value === old) ? old : 'default';
    renderManageVoice();
  } catch (error) { show('manageStatus', `Voice refresh failed: ${error.message}`, 'error'); }
}

function selectedSavedVoice() {
  const id = $('voiceSelect').value;
  return state.voices.find(voice => voice.id === id && id !== 'voice-design') || null;
}

function renderManageVoice() {
  const voice = selectedSavedVoice();
  $('manageName').value = voice?.name || '';
  $('manageTranscript').value = voice?.ref_text || '';
  $('manageName').disabled = $('manageTranscript').disabled = !voice;
  $('saveVoiceButton').disabled = $('deleteVoiceButton').disabled = !voice;
  show('manageStatus', voice ? `Editing ${voice.name || voice.id}.` : 'Select a saved voice to edit.');
}

function previewVoiceFile() {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  const file = $('voiceFile').files[0];
  $('voicePreview').hidden = !file;
  $('voicePreview').src = file ? (state.previewUrl = URL.createObjectURL(file)) : '';
}

async function transcribeReference() {
  const file = $('voiceFile').files[0];
  if (!file) { show('transcribeStatus', 'Choose a reference audio file first.', 'error'); return; }
  const form = new FormData();
  form.append('file', file); form.append('stt_base_url', $('sttUrl').value.trim());
  if ($('sttKey').value.trim()) form.append('api_key', $('sttKey').value.trim());
  $('transcribeButton').disabled = true; show('transcribeStatus', 'Transcribing…');
  try {
    const result = await (await api('/webui/transcribe', { method: 'POST', body: form })).json();
    $('voiceTranscript').value = result.text;
    rememberSttUrl();
    show('transcribeStatus', `Transcript filled using ${result.model}.`, 'success');
  } catch (error) { show('transcribeStatus', error.message, 'error'); }
  finally { $('transcribeButton').disabled = false; }
}

async function uploadVoice() {
  const file = $('voiceFile').files[0], url = $('voiceUrl').value.trim(), transcript = $('voiceTranscript').value.trim();
  if (Boolean(file) === Boolean(url)) { show('uploadStatus', 'Choose exactly one reference file or URL.', 'error'); return; }
  if (!transcript) { show('uploadStatus', 'Enter or transcribe the reference transcript.', 'error'); return; }
  const form = new FormData();
  if (file) form.append('voice_file', file); else form.append('voice_url', url);
  form.append('name', $('voiceName').value.trim()); form.append('ref_text', transcript);
  form.append('preload', $('preloadVoice').checked ? 'true' : 'false');
  $('uploadButton').disabled = true; show('uploadStatus', 'Uploading voice…');
  try {
    const result = await (await api('/upload_voice', { method: 'POST', body: form })).json();
    await refreshVoices(result.voice_id || result.id);
    show('uploadStatus', `Uploaded ${result.name || result.voice_id}.`, 'success');
    await refreshStatus();
  } catch (error) { show('uploadStatus', error.message, 'error'); }
  finally { $('uploadButton').disabled = false; }
}

async function saveVoice() {
  const voice = selectedSavedVoice(); if (!voice) return;
  $('saveVoiceButton').disabled = true; show('manageStatus', 'Saving…');
  try {
    await api(`/v1/audio/voices/${encodeURIComponent(voice.id)}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: $('manageName').value.trim(), ref_text: $('manageTranscript').value.trim() }) });
    await refreshVoices(voice.id); show('manageStatus', 'Voice saved.', 'success');
  } catch (error) { show('manageStatus', error.message, 'error'); }
  finally { $('saveVoiceButton').disabled = !selectedSavedVoice(); }
}

async function deleteVoice() {
  const voice = selectedSavedVoice(); if (!voice || !confirm(`Delete voice "${voice.name || voice.id}"?`)) return;
  $('deleteVoiceButton').disabled = true; show('manageStatus', 'Deleting…');
  try {
    await api(`/v1/audio/voices/${encodeURIComponent(voice.id)}`, { method: 'DELETE' });
    await refreshVoices('default'); show('manageStatus', 'Voice deleted.', 'success');
  } catch (error) { show('manageStatus', error.message, 'error'); }
  finally { $('deleteVoiceButton').disabled = !selectedSavedVoice(); }
}

async function refreshStatus() {
  if (state.polling) return;
  state.polling = true;
  try {
    const [healthResponse, metricsResponse] = await Promise.all([fetch('/health'), fetch('/metrics')]);
    const health = await healthResponse.json(), metrics = await metricsResponse.json();
    const status = health.status || (health.model_loaded ? 'healthy' : 'unloaded');
    if ((state.transition === 'unloading' && status === 'unloaded') || (state.transition === 'loading' && status === 'healthy')) state.transition = null;
    const label = state.transition === 'loading' ? 'Loading…' : state.transition === 'unloading' ? 'Unloading…' : status === 'healthy' ? 'Loaded' : status === 'load_failed' ? 'Load failed' : 'Unloaded';
    show('modelStatus', label); show('modelBadge', label);
    const cuda = metrics.cuda || {};
    show('gpuDevice', cuda.device_name || (cuda.device_index != null ? `CUDA ${cuda.device_index}` : 'Unavailable'));
    show('processVram', formatMb(cuda.process_memory_mb));
    show('torchVram', `${formatMb(cuda.allocated_mb)} allocated / ${formatMb(cuda.reserved_mb)} reserved`);
    show('gpuCapacity', typeof cuda.device_total_mb === 'number' ? `${formatMb(cuda.device_free_mb)} / ${formatMb(cuda.device_total_mb)}` : 'Unavailable');
    $('loadButton').disabled = Boolean(state.transition) || status === 'healthy';
    $('unloadButton').disabled = Boolean(state.transition) || status !== 'healthy';
  } catch {
    show('modelStatus', 'Reconnecting…'); show('modelBadge', 'Reconnecting…');
  } finally { state.polling = false; }
}

async function changeModel(action) {
  state.transition = action === 'load' ? 'loading' : 'unloading';
  show('modelActionStatus', action === 'load' ? 'Loading model…' : 'Unloading model…');
  show('modelStatus', action === 'load' ? 'Loading…' : 'Unloading…');
  $('loadButton').disabled = $('unloadButton').disabled = true;
  try {
    await api(`/v1/model/${action}`, { method: 'POST' });
    show('modelActionStatus', action === 'load' ? 'Model loaded.' : 'Model unloaded.', 'success');
  } catch (error) {
    state.transition = null;
    show('modelActionStatus', error.message, 'error');
  }
  await refreshStatus();
}

function stopPlayback() {
  for (const source of state.sources) { try { source.stop(); } catch { /* already finished */ } }
  state.sources.clear(); state.nextPlayTime = 0;
  if (state.context) { state.context.close().catch(() => {}); state.context = null; }
}

function schedulePcm(pcm, sampleRate) {
  const context = state.context;
  if (!context || !pcm.length) return;
  const view = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  const buffer = context.createBuffer(1, pcm.byteLength / 2, sampleRate);
  const samples = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(i * 2, true) / 32768;
  const source = context.createBufferSource(); source.buffer = buffer; source.connect(context.destination);
  source.onended = () => state.sources.delete(source); state.sources.add(source);
  const when = Math.max(context.currentTime + 0.08, state.nextPlayTime);
  source.start(when); state.nextPlayTime = when + buffer.duration;
}

function clearOutput() {
  if (state.outputUrl) URL.revokeObjectURL(state.outputUrl);
  state.outputUrl = null; $('outputAudio').removeAttribute('src'); $('outputAudio').load();
  $('downloadAudio').removeAttribute('href'); $('downloadAudio').classList.add('disabled'); $('downloadAudio').setAttribute('aria-disabled', 'true');
}

function setOutput(wav) {
  state.outputUrl = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }));
  $('outputAudio').src = state.outputUrl;
  $('downloadAudio').href = state.outputUrl;
  $('downloadAudio').download = `breeze-${new Date().toISOString().replaceAll(':', '-')}.wav`;
  $('downloadAudio').classList.remove('disabled'); $('downloadAudio').removeAttribute('aria-disabled');
}

async function generate() {
  let payload;
  try { payload = buildSpeechPayload({ model: $('modelSelect').value, voice: $('voiceSelect').value, text: $('speechText').value, instructions: $('instructions').value, guidanceScale: $('guidanceScale').value, seed: $('seed').value, requestMode: $('requestMode').value, streamFormat: $('streamFormat').value }); }
  catch (error) { show('generationStatus', error.message, 'error'); return; }
  if (state.controller) return;
  clearOutput(); stopPlayback();
  const controller = new AbortController(); state.controller = controller;
  $('generateButton').disabled = true; $('stopButton').disabled = false;
  show('generationStatus', payload.stream ? 'Starting live speech…' : 'Generating speech…');
  const started = performance.now();
  const chunks = []; let sampleRate = 24000, firstAudio = null;
  try {
    if (payload.stream) { state.context = new (window.AudioContext || window.webkitAudioContext)(); await state.context.resume(); }
    const response = await api('/v1/audio/speech', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal });
    if (!payload.stream) {
      const wav = new Uint8Array(await response.arrayBuffer());
      wavToPcm(wav); setOutput(wav);
    } else {
      await readSpeechStream(response, (pcm, rate) => {
        chunks.push(pcm.slice()); sampleRate = rate;
        if (firstAudio === null) firstAudio = (performance.now() - started) / 1000;
        schedulePcm(pcm, rate);
        show('generationStatus', `Playing live speech… first audio ${firstAudio.toFixed(1)}s, ${chunks.length} chunks.`);
      });
      setOutput(pcmToWav(chunks, sampleRate));
    }
    show('generationStatus', `Completed in ${((performance.now() - started) / 1000).toFixed(1)}s. Replay or download below.`, 'success');
    await refreshStatus();
  } catch (error) {
    show('generationStatus', error.name === 'AbortError' ? 'Generation stopped.' : error.message, error.name === 'AbortError' ? '' : 'error');
    stopPlayback();
  } finally { state.controller = null; $('generateButton').disabled = false; $('stopButton').disabled = true; }
}

function updateModeHint() {
  const nonStreaming = $('requestMode').value === 'non-streaming';
  $('streamFormat').disabled = nonStreaming;
  show('modeHint', nonStreaming ? 'Wait for a completed response before playback.' : $('streamFormat').value === 'sse' ? 'Play SSE chunks as they arrive.' : 'Play streamed audio directly from this server.');
}

async function initialize() {
  $('refreshButton').onclick = () => { refreshStatus(); refreshCatalog(); };
  $('refreshVoicesButton').onclick = () => refreshVoices();
  $('loadButton').onclick = () => changeModel('load'); $('unloadButton').onclick = () => changeModel('unload');
  $('voiceSelect').onchange = renderManageVoice;
  $('voiceFile').onchange = previewVoiceFile;
  $('sttUrl').onchange = () => { $('sttKey').value = ''; rememberSttUrl(); };
  $('transcribeButton').onclick = transcribeReference; $('uploadButton').onclick = uploadVoice;
  $('saveVoiceButton').onclick = saveVoice; $('deleteVoiceButton').onclick = deleteVoice;
  $('generateButton').onclick = generate;
  $('stopButton').onclick = () => { state.controller?.abort(); stopPlayback(); };
  $('requestMode').onchange = updateModeHint; $('streamFormat').onchange = updateModeHint;
  updateModeHint(); renderSttHistory();
  try { const config = await (await api('/webui/config')).json(); $('sttUrl').value = config.stt_base_url || ''; }
  catch (error) { show('transcribeStatus', `STT configuration unavailable: ${error.message}`, 'error'); }
  await Promise.all([refreshCatalog(), refreshStatus()]);
  setInterval(refreshStatus, 3000);
}

if (typeof document !== 'undefined') initialize();
