import { createHash } from 'node:crypto';
import type { ContentBlock, Message, ToolSchema } from '@deepseek-ai/dsh-llm';
import { buildToolSystemPrompt, TOOL_CALL_START, TOOL_CALL_END } from './tool-protocol.js';

export interface FlattenedTurn {
  role: 'user' | 'assistant';
  text: string;
  fingerprint: string;
}

export function extractBlockText(block: ContentBlock): string {
  const b = block as any;
  switch (b.type) {
    case 'text':
      return b.text ?? '';
    case 'reasoning':
      return ''; // Reasoning is internal, omit from flattened prompt
    case 'tool-call':
      return `${TOOL_CALL_START}\n${JSON.stringify({ name: b.name, arguments: tryParseJson(b.arguments) }, null, 2)}\n${TOOL_CALL_END}`;
    case 'tool-result': {
      // DSH 0.1.x backward compatibility: block-level tool result
      const innerText = Array.isArray(b.content)
        ? b.content.map(extractBlockText).filter(Boolean).join('\n')
        : (typeof b.content === 'string' ? b.content : '');
      const errFlag = b.isError ? ' (Error)' : '';
      return `[Tool Result for ${b.toolCallId ?? 'unknown'}${errFlag}]:\n${innerText}`;
    }
    case 'image':
      return `[Image attached: ${b.attachment?.id ?? 'image'}]`;
    case 'file':
      return `[File attached: ${b.attachment?.id ?? b.attachment?.name ?? 'file'}]`;
    case 'tool-addition':
    case 'tool-removal':
      return ''; // Developer tool updates in DSH 0.2.0
    default:
      return '';
  }
}

function tryParseJson(str: string): unknown {
  try {
    return JSON.parse(str);
  } catch {
    return str;
  }
}

export function extractMessageText(message: Message): string {
  const msg = message as any;

  // DSH 0.2.0 first-class ToolResultMessage (role === 'tool')
  if (msg.role === 'tool') {
    const innerText = Array.isArray(msg.content)
      ? msg.content.map(extractBlockText).filter(Boolean).join('\n')
      : (typeof msg.content === 'string' ? msg.content : '');
    const errFlag = msg.isError ? ' (Error)' : '';
    const toolCallId = msg.toolCallId ?? 'unknown';
    return `[Tool Result for ${toolCallId}${errFlag}]:\n${innerText}`;
  }

  if (Array.isArray(msg.content)) {
    return msg.content.map(extractBlockText).filter(Boolean).join('\n');
  }
  if (typeof msg.content === 'string') {
    return msg.content;
  }
  return '';
}

export function resolveSystemAndTurns(
  systemPrompt: string | undefined,
  messages: readonly Message[] = [],
  baseFingerprint = '',
): { effectiveSystem: string | undefined; turns: FlattenedTurn[] } {
  let effectiveSystem = systemPrompt && systemPrompt.trim().length > 0 ? systemPrompt.trim() : undefined;
  let historyMessages = messages;

  // DSH 0.2.0 compatibility: loop-built requests omit options.system and supply
  // the system prompt as the leading system-role message in options.messages.
  if (!effectiveSystem && messages.length > 0 && (messages[0] as any).role === 'system') {
    const firstText = extractMessageText(messages[0]).trim();
    if (firstText.length > 0) {
      effectiveSystem = firstText;
    }
    historyMessages = messages.slice(1);
  }

  const turns = TranscriptFlattener.flattenTurns(historyMessages, baseFingerprint);
  return { effectiveSystem, turns };
}

export class TranscriptFlattener {
  static resolveSystemAndTurns(
    systemPrompt: string | undefined,
    messages: readonly Message[] = [],
    baseFingerprint = '',
  ): { effectiveSystem: string | undefined; turns: FlattenedTurn[] } {
    return resolveSystemAndTurns(systemPrompt, messages, baseFingerprint);
  }

  static computeFingerprint(prevFingerprint: string, role: string, content: string): string {
    const hash = createHash('sha256');
    hash.update(prevFingerprint);
    hash.update(':');
    hash.update(role);
    hash.update(':');
    hash.update(content);
    return hash.digest('hex');
  }

  static flattenTurns(messages: readonly Message[], baseFingerprint = ''): FlattenedTurn[] {
    const turns: FlattenedTurn[] = [];
    let currentHash = baseFingerprint;

    for (const msg of messages) {
      const text = extractMessageText(msg);
      if (!text && msg.role !== 'assistant') continue;

      const role = msg.role === 'assistant' ? 'assistant' : 'user';
      currentHash = this.computeFingerprint(currentHash, role, text);
      turns.push({
        role,
        text,
        fingerprint: currentHash,
      });
    }

    return turns;
  }

  static buildFullPrompt(
    systemPrompt: string | undefined,
    tools: readonly ToolSchema[] | undefined,
    turns: readonly FlattenedTurn[],
  ): string {
    const sections: string[] = [];

    // 1. System section
    const parts: string[] = [];
    if (systemPrompt && systemPrompt.trim().length > 0) {
      parts.push(systemPrompt.trim());
    }
    if (tools && tools.length > 0) {
      const toolRules = buildToolSystemPrompt(tools);
      if (toolRules) parts.push(toolRules);
    }
    if (parts.length > 0) {
      sections.push(parts.join('\n\n'));
    }

    // 2. Transcript history
    if (turns.length === 0) {
      return sections.join('\n\n');
    }

    // If there is only one user turn and no prior turns, simply format it
    if (turns.length === 1 && turns[0].role === 'user') {
      if (sections.length > 0) {
        return `${sections.join('\n\n')}\n\n${turns[0].text}`;
      }
      return turns[0].text;
    }

    // Multiple turns: format conversational history
    const historyLines: string[] = ['# Conversation History'];
    for (let i = 0; i < turns.length - 1; i++) {
      const t = turns[i];
      const speaker = t.role === 'assistant' ? 'Assistant' : 'User';
      historyLines.push(`**${speaker}**:\n${t.text}\n`);
    }

    const lastTurn = turns[turns.length - 1];
    if (lastTurn.role === 'user') {
      historyLines.push(`**User**:\n${lastTurn.text}`);
    } else {
      historyLines.push(`**Assistant**:\n${lastTurn.text}`);
    }

    sections.push(historyLines.join('\n'));
    return sections.join('\n\n---\n\n');
  }

  static buildIncrementalPrompt(
    newTurns: readonly FlattenedTurn[],
  ): string {
    return newTurns.map((t) => t.text).join('\n\n');
  }
}
