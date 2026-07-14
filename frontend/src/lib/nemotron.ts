// Types for Nemotron (NVIDIA ASR Streaming) integration
export interface NemotronModelInfo {
  name: string;
  path: string;
  size_mb: number;
  accuracy: ModelAccuracy;
  speed: ProcessingSpeed;
  status: ModelStatus;
  description?: string;
  quantization: QuantizationType;
}

export type QuantizationType = 'FP32' | 'Int8';
export type ModelAccuracy = 'High' | 'Good' | 'Decent';
export type ProcessingSpeed = 'Slow' | 'Medium' | 'Fast' | 'Very Fast' | 'Ultra Fast';

export type ModelStatus =
  | 'Available'
  | 'Missing'
  | { Downloading: number }
  | { Error: string }
  | { Corrupted: { file_size: number; expected_min_size: number } };

export interface NemotronEngineState {
  currentModel: string | null;
  availableModels: NemotronModelInfo[];
  isLoading: boolean;
  error: string | null;
}

// User-friendly model display configuration
export interface ModelDisplayInfo {
  friendlyName: string;
  icon: string;
  tagline: string;
  recommended?: boolean;
  tier: 'fastest' | 'balanced' | 'precise';
}

export const MODEL_DISPLAY_CONFIG: Record<string, ModelDisplayInfo> = {
  'nemotron-3.5-asr-streaming-0.6b-int8': {
    friendlyName: 'Nemotron 3.5',
    icon: '🌐',
    tagline: 'Multilingual streaming • 40+ languages',
    recommended: true,
    tier: 'balanced'
  }
};

// Model configuration for Nemotron models (matching Rust implementation)
// Source: https://huggingface.co/smcleod/nemotron-3.5-asr-streaming-0.6b-int8
export const NEMOTRON_MODEL_CONFIGS: Record<string, Partial<NemotronModelInfo>> = {
  'nemotron-3.5-asr-streaming-0.6b-int8': {
    description: 'Multilingual streaming ASR, optimized for size and speed',
    size_mb: 650,
    accuracy: 'High',
    speed: 'Very Fast',
    quantization: 'Int8'
  }
};

// Helper functions
export function getModelIcon(accuracy: ModelAccuracy): string {
  switch (accuracy) {
    case 'High': return '🔥';
    case 'Good': return '⚡';
    case 'Decent': return '🚀';
    default: return '📊';
  }
}

export function getModelDisplayName(modelName: string): string {
  const displayInfo = MODEL_DISPLAY_CONFIG[modelName];
  return displayInfo?.friendlyName || modelName;
}

export function getModelDisplayInfo(modelName: string): ModelDisplayInfo | null {
  return MODEL_DISPLAY_CONFIG[modelName] || null;
}

export function getStatusColor(status: ModelStatus): string {
  if (status === 'Available') return 'green';
  if (status === 'Missing') return 'gray';
  if (typeof status === 'object' && 'Downloading' in status) return 'blue';
  if (typeof status === 'object' && 'Error' in status) return 'red';
  return 'gray';
}

export function formatFileSize(sizeMb: number): string {
  if (sizeMb >= 1000) {
    return `${(sizeMb / 1000).toFixed(1)}GB`;
  }
  return `${sizeMb}MB`;
}

export function isQuantizedModel(modelName: string): boolean {
  return modelName.includes('int8');
}

export function getModelPerformanceBadge(quantization: QuantizationType): { label: string; color: string } {
  switch (quantization) {
    case 'FP32':
      return { label: 'Full Precision', color: 'blue' };
    case 'Int8':
      return { label: 'Int8 Quantized', color: 'green' };
    default:
      return { label: 'Standard', color: 'gray' };
  }
}

export function getRecommendedModel(_systemSpecs?: { ram: number; cores: number }): string {
  return 'nemotron-3.5-asr-streaming-0.6b-int8';
}

// Tauri command wrappers for Nemotron backend
import { invoke } from '@tauri-apps/api/core';

export class NemotronAPI {
  static async init(): Promise<void> {
    await invoke('nemotron_init');
  }

  static async getAvailableModels(): Promise<NemotronModelInfo[]> {
    return await invoke('nemotron_get_available_models');
  }

  static async loadModel(modelName: string): Promise<void> {
    await invoke('nemotron_load_model', { modelName });
  }

  static async getCurrentModel(): Promise<string | null> {
    return await invoke('nemotron_get_current_model');
  }

  static async isModelLoaded(): Promise<boolean> {
    return await invoke('nemotron_is_model_loaded');
  }

  static async transcribeAudio(audioData: number[]): Promise<string> {
    return await invoke('nemotron_transcribe_audio', { audioData });
  }

  static async getModelsDirectory(): Promise<string> {
    return await invoke('nemotron_get_models_directory');
  }

  static async downloadModel(modelName: string): Promise<void> {
    await invoke('nemotron_download_model', { modelName });
  }

  static async retryDownload(modelName: string): Promise<void> {
    await invoke('nemotron_retry_download', { modelName });
  }

  static async cancelDownload(modelName: string): Promise<void> {
    await invoke('nemotron_cancel_download', { modelName });
  }

  static async deleteCorruptedModel(modelName: string): Promise<string> {
    return await invoke('nemotron_delete_corrupted_model', { modelName });
  }

  static async hasAvailableModels(): Promise<boolean> {
    return await invoke('nemotron_has_available_models');
  }

  static async validateModelReady(): Promise<string> {
    return await invoke('nemotron_validate_model_ready');
  }

  static async openModelsFolder(): Promise<void> {
    await invoke('open_nemotron_models_folder');
  }
}
