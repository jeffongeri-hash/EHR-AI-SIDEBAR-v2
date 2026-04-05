import React, { useEffect, useState } from 'react'
import { ChevronDown, Cpu, Cloud, RefreshCw, Zap, Stethoscope } from 'lucide-react'
import clsx from 'clsx'
import { chatApi } from '@/services/api'
import { useApp } from '@/contexts/AppContext'
import type { ModelInfo } from '@/types'

// Models known to be medical-specialised
const MEDICAL_MODEL_KEYS = new Set([
  'biomistral-7b', 'meditron-7b', 'clinicalcamel-13b', 'clinicalcamel-70b',
  'BioMistral/BioMistral-7B', 'epfl-llm/meditron-7b',
  'wanglab/ClinicalCamel-13B', 'wanglab/ClinicalCamel-70B',
])

export default function ModelSelector() {
  const { state, dispatch } = useApp()
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)

  const loadModels = async () => {
    setLoading(true)
    try {
      const models = await chatApi.listModels()
      dispatch({ type: 'SET_MODELS', payload: models })
    } catch {
      // silently fail – models list is non-critical
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadModels() }, [])

  const selected = state.availableModels.find((m) => m.name === state.selectedModel)
  const byProvider = state.availableModels.reduce<Record<string, ModelInfo[]>>((acc, m) => {
    ;(acc[m.provider] = acc[m.provider] || []).push(m)
    return acc
  }, {})

  const providerLabel: Record<string, string> = {
    claude:      '☁  Anthropic Claude',
    ollama:      '💻 Local (Ollama)',
    huggingface: '🤗 HuggingFace',
    tinyllama:   '⚡ TinyLlama (Local)',
    local_hf:    '🖥  Local HuggingFace',
  }

  // Split local_hf models into medical vs general for better UX
  const medicalModels = state.availableModels.filter(
    (m) => MEDICAL_MODEL_KEYS.has(m.name) || (m.description ?? '').toLowerCase().includes('medical')
      || (m.description ?? '').toLowerCase().includes('clinical')
      || (m.description ?? '').toLowerCase().includes('pubmed')
  )
  const medicalNames = new Set(medicalModels.map((m) => m.name))

  const groupedProviders = Object.entries(byProvider).reduce<Record<string, ModelInfo[]>>(
    (acc, [provider, models]) => {
      if (provider === 'local_hf') {
        const general = models.filter((m) => !medicalNames.has(m.name))
        const medical = models.filter((m) =>  medicalNames.has(m.name))
        if (general.length) acc['local_hf'] = general
        if (medical.length) acc['medical'] = medical
      } else {
        acc[provider] = models
      }
      return acc
    }, {}
  )

  const allProviderLabels: Record<string, string> = {
    ...providerLabel,
    medical: '🩺 Medical Specialised',
  }

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className={clsx(
          'flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm border transition-colors',
          'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800',
          'hover:bg-gray-50 dark:hover:bg-gray-700',
          'text-gray-700 dark:text-gray-200'
        )}
      >
        {selected?.provider === 'tinyllama' ? (
          <Zap size={14} className="text-yellow-500" />
        ) : selected?.is_local ? (
          <Cpu size={14} />
        ) : (
          <Cloud size={14} />
        )}
        <span className="max-w-[140px] truncate">
          {selected?.display_name || state.selectedModel}
        </span>
        <ChevronDown size={14} className={clsx('transition-transform', open && 'rotate-180')} />
      </button>

      {open && (
        <div className="absolute top-full mt-1 right-0 w-72 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-xl z-50 overflow-hidden animate-fade-in">
          <div className="flex items-center justify-between px-3 py-2 border-b border-gray-100 dark:border-gray-700">
            <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Select Model
            </span>
            <button
              onClick={loadModels}
              className="p-1 text-gray-400 hover:text-brand-500 transition-colors"
              title="Refresh models"
            >
              <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
            </button>
          </div>

          <div className="max-h-80 overflow-y-auto py-1">
            {Object.entries(groupedProviders).map(([provider, models]) => (
              <div key={provider}>
                <div className={clsx(
                  'px-3 py-1.5 text-xs font-medium bg-gray-50 dark:bg-gray-900/50',
                  provider === 'medical'
                    ? 'text-emerald-600 dark:text-emerald-400'
                    : 'text-gray-400 dark:text-gray-500'
                )}>
                  {allProviderLabels[provider] || provider}
                </div>
                {models.map((m) => {
                  const isMedical = medicalNames.has(m.name)
                  return (
                    <button
                      key={m.name}
                      onClick={() => {
                        dispatch({ type: 'SET_PROVIDER', payload: m.provider })
                        dispatch({ type: 'SET_MODEL', payload: m.name })
                        setOpen(false)
                      }}
                      className={clsx(
                        'w-full flex items-start gap-2 px-3 py-2 text-left transition-colors',
                        isMedical
                          ? 'hover:bg-emerald-50 dark:hover:bg-emerald-900/10'
                          : 'hover:bg-brand-50 dark:hover:bg-brand-900/20',
                        m.name === state.selectedModel && (
                          isMedical
                            ? 'bg-emerald-50 dark:bg-emerald-900/10'
                            : 'bg-brand-50 dark:bg-brand-900/20'
                        )
                      )}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1 flex-wrap">
                          <span className="text-sm font-medium text-gray-800 dark:text-gray-100 truncate">
                            {m.display_name}
                          </span>
                          {isMedical && (
                            <span className="inline-flex items-center gap-0.5 text-[10px] font-semibold
                                            bg-emerald-100 dark:bg-emerald-900/30
                                            text-emerald-700 dark:text-emerald-400
                                            rounded px-1.5 py-0.5">
                              <Stethoscope size={9} />
                              medical
                            </span>
                          )}
                          {m.supports_vision && (
                            <span className="text-xs bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded px-1">
                              vision
                            </span>
                          )}
                          {m.is_local && (
                            <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded px-1">
                              local
                            </span>
                          )}
                        </div>
                        {m.description && (
                          <p className="text-xs text-gray-400 dark:text-gray-500 truncate mt-0.5">
                            {m.description}
                          </p>
                        )}
                        <div className="flex items-center gap-2 mt-0.5">
                          <span className="text-xs text-gray-400">
                            {(m.context_length / 1000).toFixed(0)}k ctx
                          </span>
                          {m.size_gb && (
                            <span className="text-xs text-gray-400">{m.size_gb}GB</span>
                          )}
                        </div>
                      </div>
                      {m.name === state.selectedModel && (
                        <div className="w-2 h-2 rounded-full bg-brand-500 mt-1.5 flex-shrink-0" />
                      )}
                    </button>
                  )
                })}
              </div>
            ))}

            {state.availableModels.length === 0 && !loading && (
              <div className="px-3 py-4 text-center text-sm text-gray-400">
                No models found. Check backend connection.
              </div>
            )}
          </div>

          {/* Temperature */}
          <div className="border-t border-gray-100 dark:border-gray-700 px-3 py-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-gray-500">Temperature</span>
              <span className="text-xs font-mono text-gray-600 dark:text-gray-400">
                {state.temperature.toFixed(1)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={2}
              step={0.1}
              value={state.temperature}
              onChange={(e) =>
                dispatch({ type: 'SET_TEMPERATURE', payload: parseFloat(e.target.value) })
              }
              className="w-full h-1.5 accent-brand-500"
            />
          </div>
        </div>
      )}

      {open && (
        <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
      )}
    </div>
  )
}
