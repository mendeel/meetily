'use client';

import { useEffect, useState } from 'react';
import { listen } from '@tauri-apps/api/event';

type DictationPhasePayload = {
  phase: string;
  app?: string | null;
  message?: string | null;
};

function labelForPhase(phase: string, message?: string | null): string {
  switch (phase.toLowerCase()) {
    case 'listening':
    case 'transcribing':
      return 'Listening…';
    case 'polishing':
    case 'inserting':
      return 'Polishing…';
    case 'done':
      return 'Done';
    case 'error':
      return message?.trim() || 'Error';
    case 'idle':
      return '';
    default:
      return phase;
  }
}

export default function DictationPillPage() {
  const [phase, setPhase] = useState('listening');
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;

    listen<DictationPhasePayload>('dictation-phase', (event) => {
      setPhase(event.payload.phase);
      setMessage(event.payload.message ?? null);
    }).then((fn) => {
      if (cancelled) {
        fn();
        return;
      }
      unlisten = fn;
    });

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  const label = labelForPhase(phase, message);
  if (!label) {
    return <div className="h-screen w-screen bg-transparent" />;
  }

  const isListening = phase.toLowerCase() === 'listening';

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-transparent">
      <div
        className="flex items-center gap-2.5 rounded-full bg-[#1c1c1e]/95 px-4 py-2 text-sm font-medium text-white shadow-lg select-none"
        style={{ backdropFilter: 'blur(8px)' }}
      >
        <span
          className={`h-2.5 w-2.5 shrink-0 rounded-full bg-red-500 ${
            isListening ? 'animate-pulse' : ''
          }`}
          aria-hidden
        />
        <span className="whitespace-nowrap tracking-tight">{label}</span>
      </div>
    </div>
  );
}
