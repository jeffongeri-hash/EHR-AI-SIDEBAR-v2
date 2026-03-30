import React, { useEffect, useState } from 'react'
import { GitCompare, Loader2, Clock, Zap, AlertCircle, ChevronDown } from 'lucide-react'
import clsx from 'clsx'
import toast from 'react-hot-toast'
import { fineTuneApi } from '@/services/api'
import type { ModelCompareResponse, ModelCompareResult } from '@/types'

const EHR_SYSTEM_PROMPT =
  'You are an intelligent EHR assistant with expertise in medical documentation, ' +
  'clinical notes, lab results, and patient data analysis. Always be accurate and ' +
  'flag any ambiguities. Do NOT provide diagnoses or treatment recommendations.'

const SAMPLE_PROMPTS = [
  'Summarise the key findings from a patient with HbA1c of 9.2%, LDL of 145 mg/dL, and BP 148/92 mmHg.',
  'Extract all medications from: "Patient discharged on metformin 1000mg BID, lisinopril 10mg QD, atorvastatin 40mg QHS."',
  'What does an ST-elevation in leads II, III, and aVF indicate?',
  'A 72-year-old female presents with sudden onset confusion, left-sided weakness, and slurred speech. What is the most likely diagnosis?',
]

// ── Result panel ──────────────────────────────────────────────────────────────

function ResultPanel({
  label,
  tag,
  result,
  accent,
}: {
  label: string
  tag: string
  result: ModelCompareResult | null
  accent: 'blue' | 'green'
}) {
  const colors = {
    blue: {
      border: 'border-blue-200 dark:border-blue-700',
      header: 'bg-blue-50 dark:bg-blue-900/20',
      badge: 'bg-blue-100 dark:bg-blue-800 text-blue-700 dark:text-blue-300',
      latencyBar: 'bg-blue-400',
    },
    green: {
      border: 'border-green-200 dark:border-green-700',
      header: 'bg-green-50 dark:bg-green-900/20',
      badge: 'bg-green-100 dark:bg-green-800 text-green-700 dark:text-green-300',
      latencyBar: 'bg-green-500',
    },
  }[accent]

  return (
    <div className={clsx('flex-1 min-w-0 border rounded-lg overflow-hidden', colors.border)}>
      {/* Header */}
      <div className={clsx('flex items-center justify-between px-3 py-2', colors.header)}>
        <div className="flex items-center gap-2 min-w-0">
          <span className={clsx('text-xs font-semibold px-2 py-0.5 rounded-full shrink-0', colors.badge)}>
            {label}
          </span>
          <span className="text-xs text-gray-500 dark:text-gray-400 truncate font-mono">{tag}</span>
        </div>
        {result && (
          <div className="flex items-center gap-2 shrink-0 ml-2 text-xs text-gray-400">
            <Clock size={11} />
            <span>{(result.latency_ms / 1000).toFixed(2)}s</span>
            {result.tokens_generated != null && (
              <>
                <Zap size={11} />
                <span>{result.tokens_generated} tok</span>
              </>
            )}
          </div>
        )}
      </div>

      {/* Body */}
      <div className="p-3 min-h-[120px] bg-white dark:bg-gray-800">
        {!result ? (
          <p className="text-xs text-gray-400 italic">Waiting for response…</p>
        ) : result.error ? (
          <div className="flex items-start gap-2 text-red-500">
            <AlertCircle size={13} className="mt-0.5 shrink-0" />
            <p className="text-xs">{result.error}</p>
          </div>
        ) : (
          <p className="text-xs text-gray-700 dark:text-gray-200 whitespace-pre-wrap leading-relaxed">
            {result.response || <span className="italic text-gray-400">(empty response)</span>}
          </p>
        )}
      </div>
    </div>
  )
}

// ── Latency bar chart ─────────────────────────────────────────────────────────

function LatencyChart({ comparison }: { comparison: ModelCompareResponse }) {
  const maxMs = Math.max(comparison.base.latency_ms, comparison.fine_tuned.latency_ms, 1)
  const baseWidth = (comparison.base.latency_ms / maxMs) * 100
  const ftWidth = (comparison.fine_tuned.latency_ms / maxMs) * 100
  const faster =
    comparison.base.latency_ms < comparison.fine_tuned.latency_ms ? 'base' : 'fine-tuned'

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 space-y-2 bg-white dark:bg-gray-800">
      <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
        Latency comparison
      </p>
      {[
        { label: 'Base', ms: comparison.base.latency_ms, width: baseWidth, color: 'bg-blue-400' },
        { label: 'Fine-tuned', ms: comparison.fine_tuned.latency_ms, width: ftWidth, color: 'bg-green-500' },
      ].map(({ label, ms, width, color }) => (
        <div key={label} className="space-y-0.5">
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>{label}</span>
            <span className="font-mono">{(ms / 1000).toFixed(2)}s</span>
          </div>
          <div className="h-2 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className={clsx('h-full rounded-full transition-all duration-700', color)}
              style={{ width: `${width}%` }}
            />
          </div>
        </div>
      ))}
      <p className="text-xs text-gray-400">
        <span className="font-semibold text-gray-600 dark:text-gray-300 capitalize">{faster}</span> model
        responded faster
      </p>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function ModelComparison() {
  const [trainedModels, setTrainedModels] = useState<Array<{ name: string; path: string; size_mb: number }>>([])
  const [prompt, setPrompt] = useState('')
  const [baseModel, setBaseModel] = useState('llama3.2')
  const [fineTunedModel, setFineTunedModel] = useState('')
  const [maxTokens, setMaxTokens] = useState(512)
  const [temperature, setTemperature] = useState(0.7)
  const [useCustomSystem, setUseCustomSystem] = useState(false)
  const [systemPrompt, setSystemPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ModelCompareResponse | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  useEffect(() => {
    fineTuneApi.listTrainedModels().then((models) => {
      setTrainedModels(models)
      if (models.length > 0) setFineTunedModel(models[0].name)
    })
  }, [])

  const handleCompare = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a prompt')
      return
    }
    if (!fineTunedModel) {
      toast.error('No fine-tuned model selected. Train a model first.')
      return
    }
    setLoading(true)
    setResult(null)
    try {
      const res = await fineTuneApi.compareModels({
        prompt: prompt.trim(),
        base_model: baseModel,
        fine_tuned_model: fineTunedModel,
        max_tokens: maxTokens,
        temperature,
        system_prompt: useCustomSystem ? systemPrompt : undefined,
      })
      setResult(res)
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : 'Comparison failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <GitCompare size={15} className="text-brand-500" />
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200">
          Compare Models
        </h3>
      </div>

      {trainedModels.length === 0 && (
        <div className="flex items-start gap-2 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-lg">
          <AlertCircle size={13} className="text-amber-500 mt-0.5 shrink-0" />
          <p className="text-xs text-amber-700 dark:text-amber-300">
            No fine-tuned models found. Complete a training job first, then come back to compare.
          </p>
        </div>
      )}

      {/* Model selectors */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-gray-500 block mb-1">Base Model (Ollama)</label>
          <input
            value={baseModel}
            onChange={(e) => setBaseModel(e.target.value)}
            placeholder="llama3.2"
            className="w-full text-xs bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-200"
          />
        </div>
        <div>
          <label className="text-xs text-gray-500 block mb-1">Fine-tuned Model</label>
          {trainedModels.length > 0 ? (
            <select
              value={fineTunedModel}
              onChange={(e) => setFineTunedModel(e.target.value)}
              className="w-full text-xs bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-200"
            >
              {trainedModels.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name} ({m.size_mb} MB)
                </option>
              ))}
            </select>
          ) : (
            <input
              value={fineTunedModel}
              onChange={(e) => setFineTunedModel(e.target.value)}
              placeholder="model-directory-name"
              className="w-full text-xs bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-200"
            />
          )}
        </div>
      </div>

      {/* Prompt */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <label className="text-xs text-gray-500">Test Prompt</label>
          <div className="relative group">
            <button className="text-xs text-brand-500 hover:text-brand-600 flex items-center gap-0.5">
              Samples <ChevronDown size={10} />
            </button>
            <div className="absolute right-0 top-5 z-20 hidden group-hover:block w-72 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-1">
              {SAMPLE_PROMPTS.map((p, i) => (
                <button
                  key={i}
                  onClick={() => setPrompt(p)}
                  className="w-full text-left text-xs px-2 py-1.5 hover:bg-gray-50 dark:hover:bg-gray-700 rounded text-gray-700 dark:text-gray-200 truncate"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        </div>
        <textarea
          rows={4}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a clinical question or task to compare both models…"
          className="w-full text-xs bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-200 placeholder-gray-400 resize-none"
        />
      </div>

      {/* Advanced options */}
      <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
        <button
          onClick={() => setShowAdvanced((v) => !v)}
          className="w-full flex items-center justify-between px-3 py-2 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 text-xs text-gray-600 dark:text-gray-300"
        >
          Advanced options
          <ChevronDown size={12} className={clsx('transition-transform', showAdvanced && 'rotate-180')} />
        </button>
        {showAdvanced && (
          <div className="p-3 space-y-2 bg-white dark:bg-gray-800">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs text-gray-500 block mb-1">Max tokens</label>
                <input
                  type="number"
                  min={64}
                  max={4096}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  className="w-full text-xs bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-200"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500 block mb-1">Temperature</label>
                <input
                  type="number"
                  min={0}
                  max={2}
                  step={0.1}
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full text-xs bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-200"
                />
              </div>
            </div>
            <label className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-300 cursor-pointer">
              <input
                type="checkbox"
                checked={useCustomSystem}
                onChange={(e) => setUseCustomSystem(e.target.checked)}
                className="accent-brand-500"
              />
              Custom system prompt
            </label>
            {useCustomSystem && (
              <textarea
                rows={3}
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                placeholder={EHR_SYSTEM_PROMPT}
                className="w-full text-xs bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-200 placeholder-gray-400 resize-none"
              />
            )}
          </div>
        )}
      </div>

      {/* Compare button */}
      <button
        onClick={handleCompare}
        disabled={loading || !prompt.trim() || !fineTunedModel}
        className="w-full flex items-center justify-center gap-2 text-xs bg-brand-500 hover:bg-brand-600 disabled:bg-brand-300 text-white rounded-lg px-4 py-2 font-medium transition-colors"
      >
        {loading ? (
          <>
            <Loader2 size={13} className="animate-spin" />
            Running inference on both models…
          </>
        ) : (
          <>
            <GitCompare size={13} />
            Compare Models
          </>
        )}
      </button>

      {/* Results */}
      {(loading || result) && (
        <div className="space-y-3">
          <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
            Results
          </p>

          <div className="flex gap-3">
            <ResultPanel
              label="Base"
              tag={result?.base.model_name ?? baseModel}
              result={result?.base ?? null}
              accent="blue"
            />
            <ResultPanel
              label="Fine-tuned"
              tag={result?.fine_tuned.model_name ?? fineTunedModel}
              result={result?.fine_tuned ?? null}
              accent="green"
            />
          </div>

          {result && <LatencyChart comparison={result} />}
        </div>
      )}
    </div>
  )
}
