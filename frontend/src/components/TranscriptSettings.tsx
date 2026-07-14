import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Switch } from './ui/switch';
import { Eye, EyeOff, Lock, Unlock, Download } from 'lucide-react';
import { ModelManager } from './WhisperModelManager';
import { ParakeetModelManager } from './ParakeetModelManager';
import { NemotronModelManager } from './NemotronModelManager';
import { useConfig } from '@/contexts/ConfigContext';


export interface TranscriptModelProps {
    provider: 'localWhisper' | 'parakeet' | 'nemotron' | 'deepgram' | 'elevenLabs' | 'groq' | 'openai';
    model: string;
    apiKey?: string | null;
}

export interface TranscriptSettingsProps {
    transcriptModelConfig: TranscriptModelProps;
    setTranscriptModelConfig: (config: TranscriptModelProps) => void;
    onModelSelect?: () => void;
}

const LOCAL_PROVIDERS: TranscriptModelProps['provider'][] = ['localWhisper', 'parakeet', 'nemotron'];

export function TranscriptSettings({ transcriptModelConfig, setTranscriptModelConfig, onModelSelect }: TranscriptSettingsProps) {
    const [apiKey, setApiKey] = useState<string | null>(transcriptModelConfig.apiKey || null);
    const [showApiKey, setShowApiKey] = useState<boolean>(false);
    const [isApiKeyLocked, setIsApiKeyLocked] = useState<boolean>(true);
    const [isLockButtonVibrating, setIsLockButtonVibrating] = useState<boolean>(false);
    const [uiProvider, setUiProvider] = useState<TranscriptModelProps['provider']>(transcriptModelConfig.provider);
    const [diarizationDownloading, setDiarizationDownloading] = useState(false);
    const [diarizationDownloadProgress, setDiarizationDownloadProgress] = useState<number | null>(null);
    const {
        showSpeakerLabels,
        toggleSpeakerLabels,
        neuralDiarization,
        toggleNeuralDiarization,
        diarizationModelAvailable,
    } = useConfig();

    // Sync uiProvider when backend config changes (e.g., after model selection or initial load)
    useEffect(() => {
        setUiProvider(transcriptModelConfig.provider);
    }, [transcriptModelConfig.provider]);

    useEffect(() => {
        if (LOCAL_PROVIDERS.includes(transcriptModelConfig.provider)) {
            setApiKey(null);
        }
    }, [transcriptModelConfig.provider]);

    useEffect(() => {
        let unlistenProgress: (() => void) | undefined;
        let unlistenComplete: (() => void) | undefined;
        (async () => {
            try {
                const { listen } = await import('@tauri-apps/api/event');
                unlistenProgress = await listen<{ percent?: number; status?: string }>(
                    'diarization-model-download-progress',
                    (event) => {
                        const pct = event.payload?.percent;
                        if (typeof pct === 'number') {
                            setDiarizationDownloadProgress(pct);
                        }
                        if (event.payload?.status === 'completed') {
                            setDiarizationDownloading(false);
                            setDiarizationDownloadProgress(100);
                        }
                    }
                );
                unlistenComplete = await listen('diarization-model-download-complete', () => {
                    setDiarizationDownloading(false);
                    setDiarizationDownloadProgress(100);
                });
            } catch {
                // Outside Tauri
            }
        })();
        return () => {
            unlistenProgress?.();
            unlistenComplete?.();
        };
    }, []);

    const downloadDiarizationModels = async () => {
        if (diarizationDownloading) return;
        setDiarizationDownloading(true);
        setDiarizationDownloadProgress(0);
        try {
            await invoke('diarization_download_models');
        } catch (err) {
            console.error('Failed to download diarization models:', err);
            setDiarizationDownloading(false);
            setDiarizationDownloadProgress(null);
        }
    };

    const fetchApiKey = async (provider: string) => {
        try {

            const data = await invoke('api_get_transcript_api_key', { provider }) as string;

            setApiKey(data || '');
        } catch (err) {
            console.error('Error fetching API key:', err);
            setApiKey(null);
        }
    };
    const modelOptions = {
        localWhisper: [], // Model selection handled by ModelManager component
        parakeet: [], // Model selection handled by ParakeetModelManager component
        nemotron: [], // Model selection handled by NemotronModelManager component
        deepgram: ['nova-2-phonecall'],
        elevenLabs: ['eleven_multilingual_v2'],
        groq: ['llama-3.3-70b-versatile'],
        openai: ['gpt-4o'],
    };
    const requiresApiKey = transcriptModelConfig.provider === 'deepgram' || transcriptModelConfig.provider === 'elevenLabs' || transcriptModelConfig.provider === 'openai' || transcriptModelConfig.provider === 'groq';

    const handleInputClick = () => {
        if (isApiKeyLocked) {
            setIsLockButtonVibrating(true);
            setTimeout(() => setIsLockButtonVibrating(false), 500);
        }
    };

    const handleWhisperModelSelect = (modelName: string) => {
        // Always update config when model is selected, regardless of current provider
        // This ensures the model is set when user switches back
        setTranscriptModelConfig({
            ...transcriptModelConfig,
            provider: 'localWhisper', // Ensure provider is set correctly
            model: modelName
        });
        // Close modal after selection
        if (onModelSelect) {
            onModelSelect();
        }
    };

    const handleParakeetModelSelect = (modelName: string) => {
        // Always update config when model is selected, regardless of current provider
        // This ensures the model is set when user switches back
        setTranscriptModelConfig({
            ...transcriptModelConfig,
            provider: 'parakeet', // Ensure provider is set correctly
            model: modelName
        });
        // Close modal after selection
        if (onModelSelect) {
            onModelSelect();
        }
    };

    const handleNemotronModelSelect = (modelName: string) => {
        setTranscriptModelConfig({
            ...transcriptModelConfig,
            provider: 'nemotron',
            model: modelName
        });
        if (onModelSelect) {
            onModelSelect();
        }
    };

    return (
        <div>
            <div>
                {/* <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">Transcript Settings</h3>
                </div> */}
                <div className="space-y-4 pb-6">
                    <div>
                        <Label className="block text-sm font-medium text-gray-700 mb-1">
                            Transcript Model
                        </Label>
                        <div className="flex space-x-2 mx-1">
                            <Select
                                value={uiProvider}
                                onValueChange={(value) => {
                                    const provider = value as TranscriptModelProps['provider'];
                                    setUiProvider(provider);
                                    if (!LOCAL_PROVIDERS.includes(provider)) {
                                        fetchApiKey(provider);
                                    }
                                }}
                            >
                                <SelectTrigger className='focus:ring-1 focus:ring-blue-500 focus:border-blue-500'>
                                    <SelectValue placeholder="Select provider" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="parakeet">⚡ Parakeet (Recommended - Real-time / Accurate)</SelectItem>
                                    <SelectItem value="nemotron">🌐 Nemotron 3.5 (Multilingual Streaming)</SelectItem>
                                    <SelectItem value="localWhisper">🏠 Local Whisper (High Accuracy)</SelectItem>
                                    {/* <SelectItem value="deepgram">☁️ Deepgram (Backup)</SelectItem>
                                    <SelectItem value="elevenLabs">☁️ ElevenLabs</SelectItem>
                                    <SelectItem value="groq">☁️ Groq</SelectItem>
                                    <SelectItem value="openai">☁️ OpenAI</SelectItem> */}
                                </SelectContent>
                            </Select>

                            {!LOCAL_PROVIDERS.includes(uiProvider) && (
                                <Select
                                    value={transcriptModelConfig.model}
                                    onValueChange={(value) => {
                                        const model = value as TranscriptModelProps['model'];
                                        setTranscriptModelConfig({ ...transcriptModelConfig, provider: uiProvider, model });
                                    }}
                                >
                                    <SelectTrigger className='focus:ring-1 focus:ring-blue-500 focus:border-blue-500'>
                                        <SelectValue placeholder="Select model" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {modelOptions[uiProvider].map((model) => (
                                            <SelectItem key={model} value={model}>{model}</SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            )}

                        </div>
                    </div>

                    {uiProvider === 'localWhisper' && (
                        <div className="mt-6">
                            <ModelManager
                                selectedModel={transcriptModelConfig.provider === 'localWhisper' ? transcriptModelConfig.model : undefined}
                                onModelSelect={handleWhisperModelSelect}
                                autoSave={true}
                            />
                        </div>
                    )}

                    {uiProvider === 'parakeet' && (
                        <div className="mt-6">
                            <ParakeetModelManager
                                selectedModel={transcriptModelConfig.provider === 'parakeet' ? transcriptModelConfig.model : undefined}
                                onModelSelect={handleParakeetModelSelect}
                                autoSave={true}
                            />
                        </div>
                    )}

                    {uiProvider === 'nemotron' && (
                        <div className="mt-6">
                            <NemotronModelManager
                                selectedModel={transcriptModelConfig.provider === 'nemotron' ? transcriptModelConfig.model : undefined}
                                onModelSelect={handleNemotronModelSelect}
                                autoSave={true}
                            />
                        </div>
                    )}

                    {/* Speaker / diarization preferences */}
                    <div className="mt-6 space-y-4 border-t border-gray-200 pt-4">
                        <div className="flex items-center justify-between gap-4">
                            <div>
                                <p className="text-sm font-medium text-gray-700">Speaker labels</p>
                                <p className="text-xs text-gray-500">
                                    Show You / Speaker N next to each transcript line
                                </p>
                            </div>
                            <Switch
                                checked={showSpeakerLabels}
                                onCheckedChange={toggleSpeakerLabels}
                            />
                        </div>
                        <div className="flex items-center justify-between gap-4">
                            <div className="min-w-0 flex-1">
                                <p className="text-sm font-medium text-gray-700">Neural diarization</p>
                                <p className="text-xs text-gray-500">
                                    {diarizationModelAvailable
                                        ? 'Identify remote speakers (Speaker 1, Speaker 2, …) on system audio'
                                        : 'Download a diarization model to enable Speaker 1…N labels (channel labels still work)'}
                                </p>
                                {!diarizationModelAvailable && (
                                    <div className="mt-2 flex items-center gap-2">
                                        <Button
                                            type="button"
                                            variant="outline"
                                            size="sm"
                                            onClick={downloadDiarizationModels}
                                            disabled={diarizationDownloading}
                                            className="h-8"
                                        >
                                            <Download className="mr-1.5 h-3.5 w-3.5" />
                                            {diarizationDownloading
                                                ? diarizationDownloadProgress != null
                                                    ? `Downloading… ${diarizationDownloadProgress}%`
                                                    : 'Downloading…'
                                                : 'Download model'}
                                        </Button>
                                    </div>
                                )}
                            </div>
                            <Switch
                                checked={neuralDiarization}
                                onCheckedChange={toggleNeuralDiarization}
                                disabled={!diarizationModelAvailable}
                                aria-label="Neural diarization"
                            />
                        </div>
                    </div>

                    {requiresApiKey && (
                        <div>
                            <Label className="block text-sm font-medium text-gray-700 mb-1">
                                API Key
                            </Label>
                            <div className="relative mx-1">
                                <Input
                                    type={showApiKey ? "text" : "password"}
                                    className={`pr-24 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 ${isApiKeyLocked ? 'bg-gray-100 cursor-not-allowed' : ''
                                        }`}
                                    value={apiKey || ''}
                                    onChange={(e) => setApiKey(e.target.value)}
                                    disabled={isApiKeyLocked}
                                    onClick={handleInputClick}
                                    placeholder="Enter your API key"
                                />
                                {isApiKeyLocked && (
                                    <div
                                        onClick={handleInputClick}
                                        className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-50 rounded-md cursor-not-allowed"
                                    />
                                )}
                                <div className="absolute inset-y-0 right-0 pr-1 flex items-center">
                                    <Button
                                        type="button"
                                        variant="ghost"
                                        size="icon"
                                        onClick={() => setIsApiKeyLocked(!isApiKeyLocked)}
                                        className={`transition-colors duration-200 ${isLockButtonVibrating ? 'animate-vibrate text-red-500' : ''
                                            }`}
                                        title={isApiKeyLocked ? "Unlock to edit" : "Lock to prevent editing"}
                                    >
                                        {isApiKeyLocked ? <Lock className="h-4 w-4" /> : <Unlock className="h-4 w-4" />}
                                    </Button>
                                    <Button
                                        type="button"
                                        variant="ghost"
                                        size="icon"
                                        onClick={() => setShowApiKey(!showApiKey)}
                                    >
                                        {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                    </Button>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div >
    )
}
