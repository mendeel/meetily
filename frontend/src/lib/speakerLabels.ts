/**
 * Speaker / channel helpers for realtime diarization UI.
 * Contract: speaker is "you" | "others" | "speaker_N"; channel is "mic" | "system".
 */

export type TranscriptSpeakerId = 'you' | 'others' | `speaker_${number}` | string;
export type TranscriptChannel = 'mic' | 'system' | string;

/** Meeting-level map of speaker id → custom display name */
export type SpeakerAliases = Record<string, string>;

function lookupAlias(
  speaker: string,
  aliases?: SpeakerAliases | null,
): string | null {
  if (!aliases) return null;
  if (aliases[speaker]) return aliases[speaker];
  const normalized = speaker.trim().toLowerCase();
  for (const [key, value] of Object.entries(aliases)) {
    if (key.trim().toLowerCase() === normalized && value.trim()) {
      return value;
    }
  }
  return null;
}

/** Human-readable label for a speaker id (optional meeting aliases override defaults) */
export function formatSpeakerLabel(
  speaker: string | undefined | null,
  aliases?: SpeakerAliases | null,
): string | null {
  if (!speaker) return null;

  const alias = lookupAlias(speaker, aliases);
  if (alias) return alias;

  const normalized = speaker.trim().toLowerCase();
  if (normalized === 'you') return 'You';
  if (normalized === 'others') return 'Others';

  const speakerMatch = normalized.match(/^speaker[_-]?(\d+)$/i);
  if (speakerMatch) {
    return `Speaker ${speakerMatch[1]}`;
  }

  // Fallback: title-case unknown labels
  return speaker.charAt(0).toUpperCase() + speaker.slice(1);
}

/** Whether this speaker is the local user (mic channel) */
export function isLocalSpeaker(speaker: string | undefined | null): boolean {
  return (speaker ?? '').trim().toLowerCase() === 'you';
}

/** One Speakers-panel row: speakers that currently share the same display label */
export type SpeakerDisplayGroup = {
  displayName: string;
  speakerIds: string[];
};

/**
 * Aggregate speaker IDs by current display label (alias if set, else default).
 * Order follows first occurrence in `speakerIds`. Empty aliases keep Speaker 1 / 2 distinct.
 */
export function groupSpeakersByDisplayName(
  speakerIds: string[],
  aliases?: SpeakerAliases | null,
): SpeakerDisplayGroup[] {
  const groups = new Map<string, string[]>();
  const order: string[] = [];

  for (const id of speakerIds) {
    const label = formatSpeakerLabel(id, aliases) ?? id;
    const existing = groups.get(label);
    if (existing) {
      existing.push(id);
    } else {
      order.push(label);
      groups.set(label, [id]);
    }
  }

  return order.map(displayName => ({
    displayName,
    speakerIds: groups.get(displayName)!,
  }));
}

/**
 * Format a clipboard/export line: "Speaker · [MM:SS] text" (or without speaker).
 */
export function formatTranscriptExportLine(opts: {
  speaker?: string | null;
  timeLabel: string;
  text: string;
  showSpeaker?: boolean;
  aliases?: SpeakerAliases | null;
}): string {
  const { speaker, timeLabel, text, showSpeaker = true, aliases } = opts;
  const label = showSpeaker ? formatSpeakerLabel(speaker, aliases) : null;
  if (label) {
    return `${label} · ${timeLabel} ${text}`;
  }
  return `${timeLabel} ${text}`;
}
