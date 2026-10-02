import test from 'node:test';
import assert from 'node:assert/strict';
import { TranscriptFlattener, extractBlockText, extractMessageText, resolveSystemAndTurns } from '../lib/flatten.js';

test('TranscriptFlattener - extractBlockText', () => {
  assert.equal(extractBlockText({ type: 'text', text: 'hello world' }), 'hello world');
  assert.equal(extractBlockText({ type: 'reasoning', text: 'internal thought' }), '');

  const toolCallBlock = {
    type: 'tool-call',
    id: 'call_1',
    name: 'search',
    arguments: JSON.stringify({ query: 'test query' }),
  };
  const extractedTc = extractBlockText(toolCallBlock);
  assert.match(extractedTc, /<<<TOOL_CALL>>>/);
  assert.match(extractedTc, /search/);
  assert.match(extractedTc, /test query/);

  const toolResultBlock = {
    type: 'tool-result',
    toolCallId: 'call_1',
    content: [{ type: 'text', text: 'found 3 results' }],
  };
  const extractedTr = extractBlockText(toolResultBlock);
  assert.match(extractedTr, /\[Tool Result for call_1\]/);
  assert.match(extractedTr, /found 3 results/);
});

test('TranscriptFlattener - computeFingerprint determinism and chaining', () => {
  const h1 = TranscriptFlattener.computeFingerprint('', 'user', 'hello');
  const h2 = TranscriptFlattener.computeFingerprint('', 'user', 'hello');
  const hDiff = TranscriptFlattener.computeFingerprint('', 'user', 'different');

  assert.equal(h1, h2);
  assert.notEqual(h1, hDiff);
  assert.equal(typeof h1, 'string');
  assert.equal(h1.length, 64); // SHA-256 hex string

  const chained1 = TranscriptFlattener.computeFingerprint(h1, 'assistant', 'hi');
  const chained2 = TranscriptFlattener.computeFingerprint(h1, 'assistant', 'hi');
  assert.equal(chained1, chained2);
  assert.notEqual(chained1, h1);
});

test('TranscriptFlattener - flattenTurns and fingerprint chain', () => {
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'first turn' }] },
    { role: 'assistant', content: [{ type: 'text', text: 'first reply' }] },
    { role: 'user', content: [{ type: 'text', text: 'second turn' }] },
  ];

  const turns = TranscriptFlattener.flattenTurns(messages);
  assert.equal(turns.length, 3);
  assert.equal(turns[0].role, 'user');
  assert.equal(turns[0].text, 'first turn');
  assert.equal(turns[1].role, 'assistant');
  assert.equal(turns[1].text, 'first reply');
  assert.equal(turns[2].role, 'user');
  assert.equal(turns[2].text, 'second turn');

  // Verify fingerprint chain uniqueness
  assert.notEqual(turns[0].fingerprint, turns[1].fingerprint);
  assert.notEqual(turns[1].fingerprint, turns[2].fingerprint);
});

test('TranscriptFlattener - buildFullPrompt with system, tools, and history', () => {
  const tools = [
    {
      name: 'calculate',
      description: 'Perform math calculation',
      parameters: { type: 'object', properties: { expr: { type: 'string' } } },
    },
  ];

  const turns = [
    { role: 'user', text: 'what is 2 + 2?', fingerprint: 'f1' },
    { role: 'assistant', text: 'it is 4', fingerprint: 'f2' },
    { role: 'user', text: 'and 3 + 3?', fingerprint: 'f3' },
  ];

  const prompt = TranscriptFlattener.buildFullPrompt('You are a helpful assistant.', tools, turns);

  assert.match(prompt, /You are a helpful assistant\./);
  assert.match(prompt, /# Tool Use Rules/);
  assert.match(prompt, /calculate/);
  assert.match(prompt, /# Conversation History/);
  assert.match(prompt, /what is 2 \+ 2\?/);
  assert.match(prompt, /it is 4/);
  assert.match(prompt, /and 3 \+ 3\?/);
});

test('TranscriptFlattener - buildIncrementalPrompt', () => {
  const newTurns = [
    { role: 'user', text: 'next turn message', fingerprint: 'f4' },
  ];
  const prompt = TranscriptFlattener.buildIncrementalPrompt(newTurns);
  assert.equal(prompt, 'next turn message');
});

test('TranscriptFlattener - DSH 0.2.0 ToolResultMessage and FileBlock', () => {
  // DSH 0.2.0 first-class ToolResultMessage
  const toolMsgSuccess = {
    role: 'tool',
    toolCallId: 'call_abc_123',
    content: [{ type: 'text', text: 'status: 200 ok' }],
  };
  const textSuccess = extractMessageText(toolMsgSuccess);
  assert.match(textSuccess, /\[Tool Result for call_abc_123\]/);
  assert.match(textSuccess, /status: 200 ok/);

  const toolMsgError = {
    role: 'tool',
    toolCallId: 'call_xyz_456',
    isError: true,
    content: [{ type: 'text', text: 'file not found' }],
  };
  const textError = extractMessageText(toolMsgError);
  assert.match(textError, /\[Tool Result for call_xyz_456 \(Error\)\]/);
  assert.match(textError, /file not found/);

  // FileBlock
  const fileBlock = {
    type: 'file',
    attachment: { id: 'file_789', name: 'document.pdf' },
  };
  assert.equal(extractBlockText(fileBlock), '[File attached: file_789]');

  // ToolAddition / ToolRemoval blocks produce no text
  assert.equal(extractBlockText({ type: 'tool-addition', toolName: 'testTool' }), '');
  assert.equal(extractBlockText({ type: 'tool-removal', toolName: 'testTool' }), '');
});

test('TranscriptFlattener - resolveSystemAndTurns for DSH 0.1.x and 0.2.x', () => {
  // DSH 0.1.x: options.system is passed explicitly, messages starts with user
  const legacySystem = 'You are a coding assistant.';
  const legacyMessages = [
    { role: 'user', content: [{ type: 'text', text: 'hello' }] },
  ];
  const res1 = resolveSystemAndTurns(legacySystem, legacyMessages);
  assert.equal(res1.effectiveSystem, 'You are a coding assistant.');
  assert.equal(res1.turns.length, 1);
  assert.equal(res1.turns[0].role, 'user');
  assert.equal(res1.turns[0].text, 'hello');

  // DSH 0.2.0: loop-built request omits options.system, places system in messages[0]
  const dsh2Messages = [
    { role: 'system', content: [{ type: 'text', text: 'You are DeepSeek Harness.' }] },
    { role: 'user', content: [{ type: 'text', text: 'solve this issue' }] },
    { role: 'assistant', content: [{ type: 'text', text: 'analyzing...' }] },
    {
      role: 'tool',
      toolCallId: 'call_1',
      content: [{ type: 'text', text: 'test output' }],
    },
  ];
  const res2 = resolveSystemAndTurns(undefined, dsh2Messages);
  assert.equal(res2.effectiveSystem, 'You are DeepSeek Harness.');
  // The system message should not be mapped into turns
  assert.equal(res2.turns.length, 3);
  assert.equal(res2.turns[0].role, 'user');
  assert.equal(res2.turns[0].text, 'solve this issue');
  assert.equal(res2.turns[1].role, 'assistant');
  assert.equal(res2.turns[1].text, 'analyzing...');
  assert.equal(res2.turns[2].role, 'user'); // tool result maps to user turn in conversation transcript
  assert.match(res2.turns[2].text, /\[Tool Result for call_1\]/);

  // When both are absent
  const res3 = resolveSystemAndTurns('', [
    { role: 'user', content: [{ type: 'text', text: 'hi' }] },
  ]);
  assert.equal(res3.effectiveSystem, undefined);
  assert.equal(res3.turns.length, 1);
});
