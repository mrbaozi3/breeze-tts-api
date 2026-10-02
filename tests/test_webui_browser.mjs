import test from 'node:test';
import assert from 'node:assert/strict';
import { buildSpeechPayload, pcmToWav, wavToPcm, parseSseBlock, readSpeechStream } from '../breeze_infer/webui_assets/webui.mjs';

test('generation payload preserves Gradio request controls', () => {
  const payload = buildSpeechPayload({ model: 'breeze-tts-2', voice: 'voice-a', text: 'Hello there', instructions: 'Speak warmly', guidanceScale: '4', seed: '42', requestMode: 'streaming', streamFormat: 'sse' });
  assert.deepEqual(payload, { model: 'breeze-tts-2', voice: 'voice-a', input: 'Hello there', response_format: 'wav', stream: true, instructions: 'Speak warmly', guidance_scale: 4, seed: 42, stream_format: 'sse' });
  const completed = buildSpeechPayload({ model: 'breeze-tts-2', text: 'Hello', requestMode: 'non-streaming', streamFormat: 'sse' });
  assert.equal(completed.stream, false);
  assert.equal(completed.stream_format, undefined);
  assert.throws(() => buildSpeechPayload({ model: 'breeze-tts-2', text: '?!' }), /letter or digit/);
});

test('PCM chunks become a replayable WAV', () => {
  const wav = pcmToWav([new Uint8Array([0, 0]), new Uint8Array([255, 127])], 24000);
  const decoded = wavToPcm(wav);
  assert.equal(decoded.sampleRate, 24000);
  assert.deepEqual([...decoded.pcm], [0, 0, 255, 127]);
});

test('SSE parser reads split WAV chunks and completion event', async () => {
  const wav = pcmToWav([new Uint8Array([0, 0, 255, 127])], 24000);
  const event = JSON.stringify({ type: 'audio.chunk', data: Buffer.from(wav).toString('base64'), format: 'wav', sample_rate: 24000 });
  const sse = `data: ${event}\n\ndata: {"type":"done"}\n\ndata: [DONE]\n\n`;
  assert.equal(parseSseBlock('data: [DONE]').type, 'done_marker');
  const bytes = new TextEncoder().encode(sse);
  const stream = new ReadableStream({ start(controller) { controller.enqueue(bytes.subarray(0, 17)); controller.enqueue(bytes.subarray(17)); controller.close(); } });
  const response = new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
  const chunks = [];
  await readSpeechStream(response, (pcm, rate) => chunks.push({ pcm, rate }));
  assert.equal(chunks.length, 1);
  assert.equal(chunks[0].rate, 24000);
  assert.deepEqual([...chunks[0].pcm], [0, 0, 255, 127]);
});

test('raw PCM stream handles sample boundaries', async () => {
  const stream = new ReadableStream({ start(controller) { controller.enqueue(new Uint8Array([1])); controller.enqueue(new Uint8Array([2, 3, 4])); controller.close(); } });
  const response = new Response(stream, { headers: { 'Content-Type': 'audio/pcm', 'X-Sample-Rate': '24000' } });
  const chunks = [];
  await readSpeechStream(response, (pcm) => chunks.push(pcm));
  assert.deepEqual([...chunks[0]], [1, 2, 3, 4]);
});
