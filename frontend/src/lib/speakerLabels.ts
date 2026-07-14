/**
 * Speaker / channel helpers for realtime diarization UI.
 * Contract: speaker is "you" | "others" | "speaker_N"; channel is "mic" | "system".
 */

export type TranscriptSpeakerId = 'you' | 'others' | `speaker_${number}` | string;
export type TranscriptChannel = 'mic' | 'system' | string;

/** Human-readable label for a speaker id */
export function formatSpeakerLabel(speaker: string | undefined | null): string | null {
  if (!speaker) return null;

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

/**
 * Format a clipboard/export line: "Speaker · [MM:SS] text" (or without speaker).
 */
export function formatTranscriptExportLine(opts: {
  speaker?: string | null;
  timeLabel: string;
  text: string;
  showSpeaker?: boolean;
}): string {
  const { speaker, timeLabel, text, showSpeaker = true } = opts;
  const label = showSpeaker ? formatSpeakerLabel(speaker) : null;
  if (label) {
    return `${label} · ${timeLabel} ${text}`;
  }
  return `${timeLabel} ${text}`;
}
