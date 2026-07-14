'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { invoke } from '@tauri-apps/api/core';
import { AlertCircle, CheckCircle2, ExternalLink, Mic2, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';

export type TriggerMode = 'PushToTalk' | 'Toggle';
export type SttEngine = 'nemotron' | 'parakeet' | 'whisper';
export type PolishProfile = 'ide' | 'chat' | 'email' | 'default';

export interface DictationConfig {
  enabled: boolean;
  trigger_mode: TriggerMode;
  hold_shortcut: string;
  toggle_shortcut: string;
  stt_engine: SttEngine;
  polish_enabled: boolean;
  default_profile: PolishProfile;
  profile_overrides: Record<string, PolishProfile>;
}

const DEFAULT_CONFIG: DictationConfig = {
  enabled: true,
  trigger_mode: 'PushToTalk',
  hold_shortcut: 'CommandOrControl+Shift+Space',
  toggle_shortcut: 'CommandOrControl+Shift+D',
  stt_engine: 'nemotron',
  polish_enabled: true,
  default_profile: 'default',
  profile_overrides: {},
};

export function DictationSettings() {
  const [config, setConfig] = useState<DictationConfig>(DEFAULT_CONFIG);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [backendUnavailable, setBackendUnavailable] = useState(false);
  const [accessibilityTrusted, setAccessibilityTrusted] = useState<boolean | null>(null);
  const [checkingAccessibility, setCheckingAccessibility] = useState(false);
  const [micGranted, setMicGranted] = useState<boolean | null>(null);
  const [checkingMic, setCheckingMic] = useState(false);
  const [testingDictation, setTestingDictation] = useState(false);
  const [testListening, setTestListening] = useState(false);

  const loadAccessibilityStatus = useCallback(async () => {
    setCheckingAccessibility(true);
    try {
      const trusted = await invoke<boolean>('dictation_accessibility_status');
      setAccessibilityTrusted(trusted);
    } catch (error) {
      console.error('Failed to check accessibility status:', error);
      setAccessibilityTrusted(null);
    } finally {
      setCheckingAccessibility(false);
    }
  }, []);

  const requestMicPermission = useCallback(async () => {
    setCheckingMic(true);
    try {
      const granted = await invoke<boolean>('trigger_microphone_permission');
      setMicGranted(granted);
      if (granted) {
        toast.success('Microphone access granted');
      } else {
        toast.error('Microphone access denied', {
          description: 'Enable Meetily in System Settings → Privacy & Security → Microphone.',
        });
      }
    } catch (error) {
      console.error('Failed to request microphone permission:', error);
      setMicGranted(false);
      toast.error('Could not request microphone permission', {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setCheckingMic(false);
    }
  }, []);

  useEffect(() => {
    const loadConfig = async () => {
      try {
        const loaded = await invoke<DictationConfig>('dictation_get_config');
        setConfig({
          ...DEFAULT_CONFIG,
          ...loaded,
          profile_overrides: loaded.profile_overrides ?? {},
        });
        setBackendUnavailable(false);
      } catch (error) {
        console.error('Failed to load dictation config:', error);
        setBackendUnavailable(true);
        setConfig(DEFAULT_CONFIG);
      } finally {
        setLoading(false);
      }
    };

    loadConfig();
    loadAccessibilityStatus();
  }, [loadAccessibilityStatus]);

  const saveConfig = async (next: DictationConfig, options?: { silentIfUnchanged?: boolean }) => {
    if (options?.silentIfUnchanged && JSON.stringify(next) === JSON.stringify(config)) {
      return;
    }
    setSaving(true);
    setConfig(next);
    try {
      await invoke('dictation_set_config', { config: next });
      setBackendUnavailable(false);
      toast.success('Dictation settings saved');
    } catch (error) {
      console.error('Failed to save dictation config:', error);
      setBackendUnavailable(true);
      toast.error('Failed to save dictation settings', {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setSaving(false);
    }
  };

  const updateField = <K extends keyof DictationConfig>(key: K, value: DictationConfig[K]) => {
    void saveConfig({ ...config, [key]: value });
  };

  const handleOpenAccessibilitySettings = async () => {
    try {
      await invoke('dictation_open_accessibility_settings');
    } catch (error) {
      console.error('Failed to open accessibility settings:', error);
      toast.error('Could not open Accessibility settings', {
        description: error instanceof Error ? error.message : String(error),
      });
    }
  };

  const handleTestDictation = async () => {
    setTestingDictation(true);
    try {
      if (testListening) {
        await invoke('dictation_stop');
        setTestListening(false);
        toast.message('Test dictation stopped — processing…');
      } else {
        await invoke('dictation_start');
        setTestListening(true);
        toast.message('Listening — speak, then click Stop Test');
      }
    } catch (error) {
      console.error('Test dictation failed:', error);
      setTestListening(false);
      toast.error('Test dictation failed', {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setTestingDictation(false);
    }
  };

  if (loading) {
    return (
      <div className="animate-pulse space-y-4 mt-6">
        <div className="h-4 bg-gray-200 rounded w-1/4" />
        <div className="h-8 bg-gray-200 rounded" />
        <div className="h-8 bg-gray-200 rounded" />
      </div>
    );
  }

  return (
    <div className="space-y-6 mt-6">
      <div>
        <h3 className="text-lg font-semibold mb-2">Dictation Settings</h3>
        <p className="text-sm text-gray-600 mb-4">
          System-wide AI dictation is separate from meeting recording. It uses your microphone
          only—hold or toggle a global shortcut to dictate into any app.
        </p>
      </div>

      {backendUnavailable && (
        <div className="flex items-start gap-3 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <AlertCircle className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-yellow-800">
            <p className="font-medium">Dictation backend not available yet</p>
            <p className="mt-1">
              Settings are shown with defaults. Saving will work once dictation commands are
              registered in the app.
            </p>
          </div>
        </div>
      )}

      {/* Enable */}
      <div className="flex items-center justify-between p-4 border rounded-lg">
        <div className="flex-1">
          <div className="font-medium">Enable Dictation</div>
          <div className="text-sm text-gray-600">
            Register global shortcuts and allow dictation sessions
          </div>
        </div>
        <Switch
          checked={config.enabled}
          onCheckedChange={(enabled) => updateField('enabled', enabled)}
          disabled={saving}
        />
      </div>

      {/* Trigger mode */}
      <div className="p-4 border rounded-lg space-y-3">
        <div>
          <div className="font-medium">Trigger Mode</div>
          <div className="text-sm text-gray-600">
            Push-to-talk holds while the key is down; Toggle starts and stops on each press
          </div>
        </div>
        <Select
          value={config.trigger_mode}
          onValueChange={(value: TriggerMode) => updateField('trigger_mode', value)}
          disabled={saving}
        >
          <SelectTrigger className="max-w-xs focus:ring-1 focus:ring-blue-500 focus:border-blue-500">
            <SelectValue placeholder="Select trigger mode" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="PushToTalk">Push-to-talk</SelectItem>
            <SelectItem value="Toggle">Toggle</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Shortcuts */}
      <div className="p-4 border rounded-lg space-y-4">
        <div>
          <div className="font-medium">Keyboard Shortcuts</div>
          <div className="text-sm text-gray-600">
            Use Tauri-style accelerators (e.g. CommandOrControl+Shift+Space)
          </div>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700" htmlFor="hold-shortcut">
              Hold shortcut (push-to-talk)
            </label>
            <Input
              id="hold-shortcut"
              value={config.hold_shortcut}
              onChange={(e) => setConfig((prev) => ({ ...prev, hold_shortcut: e.target.value }))}
              onBlur={(e) => {
                const hold_shortcut = e.target.value.trim() || DEFAULT_CONFIG.hold_shortcut;
                void saveConfig({ ...config, hold_shortcut }, { silentIfUnchanged: true });
              }}
              disabled={saving}
              placeholder="CommandOrControl+Shift+Space"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700" htmlFor="toggle-shortcut">
              Toggle shortcut
            </label>
            <Input
              id="toggle-shortcut"
              value={config.toggle_shortcut}
              onChange={(e) => setConfig((prev) => ({ ...prev, toggle_shortcut: e.target.value }))}
              onBlur={(e) => {
                const toggle_shortcut = e.target.value.trim() || DEFAULT_CONFIG.toggle_shortcut;
                void saveConfig({ ...config, toggle_shortcut }, { silentIfUnchanged: true });
              }}
              disabled={saving}
              placeholder="CommandOrControl+Shift+D"
            />
          </div>
        </div>
      </div>

      {/* STT engine */}
      <div className="p-4 border rounded-lg space-y-3">
        <div>
          <div className="font-medium">Speech-to-Text Engine</div>
          <div className="text-sm text-gray-600">
            Nemotron is the default for dictation; requires a downloaded model
          </div>
        </div>
        <Select
          value={config.stt_engine}
          onValueChange={(value: SttEngine) => updateField('stt_engine', value)}
          disabled={saving}
        >
          <SelectTrigger className="max-w-xs focus:ring-1 focus:ring-blue-500 focus:border-blue-500">
            <SelectValue placeholder="Select STT engine" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="nemotron">Nemotron (default)</SelectItem>
            <SelectItem value="parakeet">Parakeet</SelectItem>
            <SelectItem value="whisper">Whisper</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Polish */}
      <div className="flex items-center justify-between p-4 border rounded-lg">
        <div className="flex-1">
          <div className="font-medium">LLM Polish</div>
          <div className="text-sm text-gray-600">
            Clean up and format dictated text with your summary LLM after transcription
          </div>
        </div>
        <Switch
          checked={config.polish_enabled}
          onCheckedChange={(polish_enabled) => updateField('polish_enabled', polish_enabled)}
          disabled={saving}
        />
      </div>

      {/* Default profile */}
      <div className="p-4 border rounded-lg space-y-3">
        <div>
          <div className="font-medium">Default Polish Profile</div>
          <div className="text-sm text-gray-600">
            Used when the frontmost app has no built-in or custom override
          </div>
        </div>
        <Select
          value={config.default_profile}
          onValueChange={(value: PolishProfile) => updateField('default_profile', value)}
          disabled={saving || !config.polish_enabled}
        >
          <SelectTrigger className="max-w-xs focus:ring-1 focus:ring-blue-500 focus:border-blue-500">
            <SelectValue placeholder="Select profile" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="default">Default</SelectItem>
            <SelectItem value="ide">IDE / code</SelectItem>
            <SelectItem value="chat">Chat / messaging</SelectItem>
            <SelectItem value="email">Email</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Microphone */}
      <div className="p-4 border rounded-lg space-y-4">
        <div>
          <div className="font-medium">Microphone Permission</div>
          <div className="text-sm text-gray-600">
            Required to capture speech for dictation. Click Request to prompt the system dialog.
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {micGranted === true && (
            <span className="inline-flex items-center gap-1.5 text-sm text-green-700">
              <CheckCircle2 className="w-4 h-4" />
              Microphone granted
            </span>
          )}
          {micGranted === false && (
            <span className="inline-flex items-center gap-1.5 text-sm text-amber-700">
              <AlertCircle className="w-4 h-4" />
              Microphone denied or unavailable
            </span>
          )}
          {micGranted === null && (
            <span className="text-sm text-gray-500">Not checked yet</span>
          )}
          <button
            type="button"
            onClick={() => void requestMicPermission()}
            disabled={checkingMic}
            className="flex items-center gap-2 px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50 transition-colors disabled:opacity-50"
          >
            <Mic2 className={`w-4 h-4 ${checkingMic ? 'animate-pulse' : ''}`} />
            {checkingMic ? 'Requesting…' : 'Request microphone'}
          </button>
        </div>
      </div>

      {/* Accessibility */}
      <div className="p-4 border rounded-lg space-y-4">
        <div>
          <div className="font-medium">Accessibility Permission</div>
          <div className="text-sm text-gray-600">
            Required on macOS to paste dictated text at the cursor. Without it, text falls back to
            the clipboard.
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {accessibilityTrusted === true && (
            <span className="inline-flex items-center gap-1.5 text-sm text-green-700">
              <CheckCircle2 className="w-4 h-4" />
              Accessibility trusted
            </span>
          )}
          {accessibilityTrusted === false && (
            <span className="inline-flex items-center gap-1.5 text-sm text-amber-700">
              <AlertCircle className="w-4 h-4" />
              Not trusted — clipboard fallback will be used
            </span>
          )}
          {accessibilityTrusted === null && (
            <span className="text-sm text-gray-500">Status unavailable</span>
          )}
          <button
            type="button"
            onClick={() => void loadAccessibilityStatus()}
            disabled={checkingAccessibility}
            className="flex items-center gap-2 px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${checkingAccessibility ? 'animate-spin' : ''}`} />
            Refresh status
          </button>
          <button
            type="button"
            onClick={() => void handleOpenAccessibilitySettings()}
            className="flex items-center gap-2 px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
            Open Accessibility Settings
          </button>
        </div>
      </div>

      {/* Test */}
      <div className="p-4 border rounded-lg space-y-3">
        <div>
          <div className="font-medium">Test Dictation</div>
          <div className="text-sm text-gray-600">
            Starts the same flow as the global shortcut. Speak, then click again to stop and
            insert.
          </div>
        </div>
        <button
          type="button"
          onClick={() => void handleTestDictation()}
          disabled={testingDictation || !config.enabled || backendUnavailable}
          className="flex items-center gap-2 px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50 transition-colors disabled:opacity-50"
        >
          <Mic2 className="w-4 h-4" />
          {testingDictation ? 'Working…' : testListening ? 'Stop Test' : 'Test Dictation'}
        </button>
      </div>

      {/* Help */}
      <div className="flex items-start gap-3 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <Mic2 className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-blue-800">
          <p className="font-medium">Not meeting recording</p>
          <p className="mt-1">
            Dictation captures mic audio only for short utterances and inserts text into the active
            app. Stop meeting recording before dictating (and vice versa). Microphone permission is
            required.
          </p>
        </div>
      </div>
    </div>
  );
}
