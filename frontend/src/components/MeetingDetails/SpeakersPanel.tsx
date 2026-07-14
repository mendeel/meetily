'use client';

import { useCallback, useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  formatSpeakerLabel,
  groupSpeakersByDisplayName,
  SpeakerAliases,
} from '@/lib/speakerLabels';
import { storageService } from '@/services/storageService';
import { Pencil, RotateCcw, Users, X } from 'lucide-react';
import { toast } from 'sonner';

interface SpeakersPanelProps {
  meetingId: string;
  aliases: SpeakerAliases;
  onAliasesChange: (aliases: SpeakerAliases) => void;
}

export function SpeakersPanel({
  meetingId,
  aliases,
  onAliasesChange,
}: SpeakersPanelProps) {
  const [speakers, setSpeakers] = useState<string[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const groups = groupSpeakersByDisplayName(speakers, aliases);

  const loadSpeakers = useCallback(async () => {
    if (!meetingId) return;
    setIsLoading(true);
    try {
      const ids = await storageService.getMeetingSpeakers(meetingId);
      setSpeakers(ids);
    } catch (err) {
      console.error('Failed to load meeting speakers:', err);
      toast.error('Failed to load speakers');
    } finally {
      setIsLoading(false);
    }
  }, [meetingId]);

  useEffect(() => {
    if (isOpen) {
      loadSpeakers();
    }
  }, [isOpen, loadSpeakers]);

  const startEdit = (displayName: string) => {
    setEditingKey(displayName);
    setDraft(displayName);
  };

  const cancelEdit = () => {
    setEditingKey(null);
    setDraft('');
  };

  const saveGroupAlias = async (speakerIds: string[], displayName: string) => {
    setIsSaving(true);
    try {
      let updated: SpeakerAliases = aliases;
      for (const speakerId of speakerIds) {
        updated = await storageService.setSpeakerAlias(
          meetingId,
          speakerId,
          displayName
        );
      }
      onAliasesChange(updated);
      setEditingKey(null);
      setDraft('');
    } catch (err) {
      console.error('Failed to save speaker alias:', err);
      toast.error('Failed to rename speaker');
    } finally {
      setIsSaving(false);
    }
  };

  const clearGroupAliases = async (speakerIds: string[]) => {
    await saveGroupAlias(speakerIds, '');
  };

  return (
    <div className="w-full">
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="w-full justify-center"
        onClick={() => setIsOpen(prev => !prev)}
        title="Rename speakers"
      >
        <Users size={16} className="mr-1.5" />
        Speakers
      </Button>

      {isOpen && (
        <div className="mt-2 rounded-md border border-gray-200 bg-gray-50 p-2 space-y-2">
          <div className="flex items-center justify-between px-1">
            <p className="text-xs font-medium text-gray-600">Speaker names</p>
            <button
              type="button"
              className="text-gray-400 hover:text-gray-600"
              onClick={() => setIsOpen(false)}
              aria-label="Close speakers panel"
            >
              <X size={14} />
            </button>
          </div>

          {isLoading ? (
            <p className="text-xs text-gray-400 px-1 py-2">Loading…</p>
          ) : groups.length === 0 ? (
            <p className="text-xs text-gray-400 px-1 py-2">No speakers found</p>
          ) : (
            <ul className="space-y-1.5">
              {groups.map(group => {
                const { displayName, speakerIds } = group;
                const hasAlias = speakerIds.some(id =>
                  Boolean(aliases[id]?.trim())
                );
                const isEditing = editingKey === displayName;
                const isMerged = speakerIds.length > 1;
                const isYouOnly =
                  speakerIds.length === 1 &&
                  speakerIds[0].toLowerCase() === 'you';
                const sourceHint = isMerged
                  ? `${speakerIds.length} labels`
                  : undefined;
                const sourceTitle = isMerged
                  ? speakerIds
                      .map(id => formatSpeakerLabel(id) ?? id)
                      .join(', ')
                  : speakerIds[0];

                return (
                  <li
                    key={displayName}
                    className="flex items-center gap-1.5 rounded bg-white border border-gray-100 px-2 py-1.5"
                  >
                    {isEditing ? (
                      <input
                        autoFocus
                        className="flex-1 min-w-0 text-sm border border-blue-200 rounded px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-blue-400"
                        value={draft}
                        disabled={isSaving}
                        onChange={e => setDraft(e.target.value)}
                        onKeyDown={e => {
                          if (e.key === 'Enter') {
                            e.preventDefault();
                            saveGroupAlias(speakerIds, draft);
                          } else if (e.key === 'Escape') {
                            cancelEdit();
                          }
                        }}
                        onBlur={() => {
                          if (!isSaving) {
                            saveGroupAlias(speakerIds, draft);
                          }
                        }}
                      />
                    ) : (
                      <>
                        <div className="flex-1 min-w-0">
                          <span
                            className={`block truncate text-sm ${
                              isYouOnly
                                ? 'text-blue-600 font-medium'
                                : 'text-teal-700 font-medium'
                            }`}
                            title={sourceTitle}
                          >
                            {displayName}
                          </span>
                          {sourceHint && (
                            <span
                              className="block truncate text-[10px] text-gray-400"
                              title={sourceTitle}
                            >
                              {sourceHint}
                            </span>
                          )}
                        </div>
                        <button
                          type="button"
                          className="text-gray-400 hover:text-gray-700 p-0.5"
                          title={
                            isMerged
                              ? `Rename all (${speakerIds.length})`
                              : 'Rename'
                          }
                          onClick={() => startEdit(displayName)}
                        >
                          <Pencil size={12} />
                        </button>
                        {hasAlias && (
                          <button
                            type="button"
                            className="text-gray-400 hover:text-gray-700 p-0.5"
                            title={
                              isMerged
                                ? 'Clear custom names'
                                : 'Clear custom name'
                            }
                            onClick={() => clearGroupAliases(speakerIds)}
                            disabled={isSaving}
                          >
                            <RotateCcw size={12} />
                          </button>
                        )}
                      </>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
          <p className="text-[10px] text-gray-400 px-1">
            Clear a name to restore the default label.
          </p>
        </div>
      )}
    </div>
  );
}
