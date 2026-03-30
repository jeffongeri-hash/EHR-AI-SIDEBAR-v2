import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  Zap,
  Play,
  X,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  CheckCircle,
  AlertCircle,
  Loader2,
  BookOpen,
  Plus,
  Trash2,
  UploadCloud,
  FileText,
  Database,
  GitCompare,
} from 'lucide-react'
import ModelComparison from '@/components/ModelComparison'
import clsx from 'clsx'
import toast from 'react-hot-toast'
import { fineTuneApi } from '@/services/api'
import type { FineTuneJob, TrainingConfig, TrainingDataset, TrainingExample } from '@/types'

const DEFAULT_CONFIG: TrainingConfig = {
  base_model: 'meta-llama/Llama-3.2-3B-Instruct',
  output_model_name: 'ehr-llama-3b',
  num_epochs: 3,
  batch_size: 4,
  learning_rate: 0.0002,
  warmup_steps: 100,
  max_seq_length: 2048,
  gradient_accumulation_steps: 4,
  use_lora: true,
  lora_config: { r: 16, lora_alpha: 32, lora_dropout: 0.1, target_modules: ['q_proj', 'v_proj'], bias: 'none' },
  load_in_4bit: true,
  fp16: true,
  gradient_checkpointing: true,
  push_to_hub: false,
  training_examples: [],
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { color: string; icon: React.ReactNode }> = {
    queued: { color: 'text-gray-500 bg-gray-100 dark:bg-gray-700', icon: <RefreshCw size={11} /> },
    preparing: { color: 'text-blue-500 bg-blue-50 dark:bg-blue-900/20', icon: <Loader2 size={11} className="animate-spin" /> },
    training: { color: 'text-brand-600 bg-brand-50 dark:bg-brand-900/20', icon: <Loader2 size={11} className="animate-spin" /> },
    completed: { color: 'text-green-600 bg-green-50 dark:bg-green-900/20', icon: <CheckCircle size={11} /> },
    failed: { color: 'text-red-600 bg-red-50 dark:bg-red-900/20', icon: <AlertCircle size={11} /> },
  }
  const s = map[status] || map.queued
  return (
    <span className={clsx('flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium', s.color)}>
      {s.icon} {status}
    </span>
  )
}

function JobCard({ job, onRefresh, onCancel }: {
  job: FineTuneJob
  onRefresh: () => void
  onCancel: () => void
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      <div className="flex items-center gap-2 p-2.5 bg-white dark:bg-gray-800">
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-gray-800 dark:text-gray-100 truncate">
            {job.config.output_model_name}
          </p>
          <p className="text-xs text-gray-400 truncate">{job.config.base_model}</p>
        </div>
        <StatusBadge status={job.status} />
        <div className="flex items-center gap-1">
          <button onClick={onRefresh} className="p-1 text-gray-400 hover:text-brand-500">
            <RefreshCw size={12} />
          </button>
          {['queued', 'training', 'preparing'].includes(job.status) && (
            <button onClick={onCancel} className="p-1 text-gray-400 hover:text-red-500">
              <X size={12} />
            </button>
          )}
          <button onClick={() => setExpanded((v) => !v)} className="p-1 text-gray-400">
            {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>
        </div>
      </div>

      {/* Progress bar */}
      {['training', 'preparing'].includes(job.status) && (
        <div className="px-2.5 pb-2 bg-white dark:bg-gray-800">
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Epoch {job.current_epoch} · loss {job.current_loss?.toFixed(4) ?? '—'}</span>
            <span>{Math.round(job.progress * 100)}%</span>
          </div>
          <div className="h-1.5 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-brand-500 rounded-full transition-all duration-500"
              style={{ width: `${job.progress * 100}%` }}
            />
          </div>
        </div>
      )}

      {expanded && (
        <div className="border-t border-gray-100 dark:border-gray-700 p-2.5 bg-gray-50 dark:bg-gray-900/50 space-y-2">
          <div className="grid grid-cols-2 gap-1 text-xs">
            {[
              ['Epochs', job.config.num_epochs],
              ['Batch size', job.config.batch_size],
              ['LR', job.config.learning_rate],
              ['LoRA r', job.config.lora_config.r],
              ['4-bit', job.config.load_in_4bit ? 'Yes' : 'No'],
              ['Seq len', job.config.max_seq_length],
            ].map(([k, v]) => (
              <div key={String(k)} className="flex justify-between">
                <span className="text-gray-400">{k}</span>
                <span className="text-gray-700 dark:text-gray-300 font-mono">{v}</span>
              </div>
            ))}
          </div>
          {job.logs.length > 0 && (
            <div className="bg-gray-900 rounded p-2 max-h-32 overflow-y-auto">
              {job.logs.slice(-20).map((log, i) => (
                <p key={i} className="text-xs font-mono text-green-400">{log}</p>
              ))}
            </div>
          )}
          {job.error && (
            <p className="text-xs text-red-500 bg-red-50 dark:bg-red-900/20 rounded p-2">
              {job.error}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function ExampleEditor({
  examples,
  onChange,
}: {
  examples: TrainingExample[]
  onChange: (e: TrainingExample[]) => void
}) {
  const addExample = () =>
    onChange([...examples, { instruction: '', input: '', output: '', system: '' }])

  const updateExample = (i: number, field: keyof TrainingExample, value: string) => {
    const updated = [...examples]
    updated[i] = { ...updated[i], [field]: value }
    onChange(updated)
  }

  const removeExample = (i: number) => onChange(examples.filter((_, idx) => idx !== i))

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          Training Examples ({examples.length})
        </span>
        <button
          onClick={addExample}
          className="flex items-center gap-1 text-xs text-brand-500 hover:text-brand-600"
        >
          <Plus size={12} /> Add
        </button>
      </div>
      {examples.map((ex, i) => (
        <div key={i} className="border border-gray-200 dark:border-gray-700 rounded-lg p-2 space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gray-500">Example {i + 1}</span>
            <button onClick={() => removeExample(i)} className="text-gray-400 hover:text-red-500">
              <Trash2 size={11} />
            </button>
          </div>
          {(['instruction', 'input', 'output', 'system'] as const).map((field) => (
            <div key={field}>
              <label className="text-xs text-gray-400 capitalize">{field}</label>
              <textarea
                rows={field === 'output' ? 2 : 1}
                value={ex[field] || ''}
                onChange={(e) => updateExample(i, field, e.target.value)}
                placeholder={
                  field === 'instruction' ? 'e.g. Summarise this clinical note' :
                  field === 'input' ? 'Optional context / input data' :
                  field === 'output' ? 'Expected model response' :
                  'System prompt (optional)'
                }
                className="w-full text-xs bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1 mt-0.5 resize-none text-gray-700 dark:text-gray-200 placeholder-gray-400"
              />
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

// ── Training Document Upload ───────────────────────────────────────────────────

const ACCEPTED_FORMATS = '.pdf,.docx,.jpg,.jpeg,.png,.tiff,.tif,.bmp,.webp'

function TrainingDocumentUpload({
  onDatasetReady,
}: {
  onDatasetReady: (datasetPath: string) => void
}) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [chunkSize, setChunkSize] = useState(512)
  const [fmt, setFmt] = useState<'text' | 'instruction'>('text')
  const [datasets, setDatasets] = useState<TrainingDataset[]>([])
  const [loadingDatasets, setLoadingDatasets] = useState(false)

  const loadDatasets = async () => {
    setLoadingDatasets(true)
    try {
      setDatasets(await fineTuneApi.listDatasets())
    } finally {
      setLoadingDatasets(false)
    }
  }

  useEffect(() => { loadDatasets() }, [])

  const uploadFile = useCallback(async (file: File) => {
    setUploading(true)
    try {
      const result = await fineTuneApi.uploadDocument(file, {
        chunk_size: chunkSize,
        overlap: 64,
        format: fmt,
      })
      toast.success(`Extracted ${result.chunk_count} training chunks from "${result.filename}"`)
      await loadDatasets()
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }, [chunkSize, fmt])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) uploadFile(file)
  }, [uploadFile])

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) uploadFile(file)
    e.target.value = ''
  }

  const deleteDataset = async (datasetId: string) => {
    try {
      await fineTuneApi.deleteDataset(datasetId)
      setDatasets((prev) => prev.filter((d) => d.dataset_id !== datasetId))
      toast.success('Dataset deleted')
    } catch {
      toast.error('Failed to delete dataset')
    }
  }

  return (
    <div className="space-y-3">
      {/* Upload options */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-gray-500 block mb-1">Chunk size (words)</label>
          <input
            type="number"
            min={64}
            max={2048}
            value={chunkSize}
            onChange={(e) => setChunkSize(Number(e.target.value))}
            className="w-full text-xs bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-200"
          />
        </div>
        <div>
          <label className="text-xs text-gray-500 block mb-1">Format</label>
          <select
            value={fmt}
            onChange={(e) => setFmt(e.target.value as 'text' | 'instruction')}
            className="w-full text-xs bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-200"
          >
            <option value="text">Raw text (pre-train)</option>
            <option value="instruction">Instruction format</option>
          </select>
        </div>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !uploading && fileInputRef.current?.click()}
        className={clsx(
          'flex flex-col items-center justify-center gap-2 border-2 border-dashed rounded-lg p-4 cursor-pointer transition-colors',
          dragging
            ? 'border-brand-400 bg-brand-50 dark:bg-brand-900/20'
            : 'border-gray-200 dark:border-gray-600 hover:border-brand-300 hover:bg-gray-50 dark:hover:bg-gray-800/50'
        )}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_FORMATS}
          className="hidden"
          onChange={onFileChange}
        />
        {uploading ? (
          <Loader2 size={20} className="text-brand-500 animate-spin" />
        ) : (
          <UploadCloud size={20} className="text-gray-400" />
        )}
        <div className="text-center">
          <p className="text-xs font-medium text-gray-700 dark:text-gray-300">
            {uploading ? 'Processing document…' : 'Drop a document to add training data'}
          </p>
          <p className="text-xs text-gray-400 mt-0.5">PDF, DOCX, JPEG, PNG · Max 50 MB</p>
        </div>
      </div>

      {/* Saved datasets */}
      {datasets.length > 0 && (
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Saved Datasets ({datasets.length})
            </span>
            <button onClick={loadDatasets} className="p-1 text-gray-400 hover:text-brand-500">
              <RefreshCw size={11} className={loadingDatasets ? 'animate-spin' : ''} />
            </button>
          </div>
          {datasets.map((ds) => (
            <div
              key={ds.dataset_id}
              className="flex items-center gap-2 p-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 group"
            >
              <Database size={13} className="text-brand-400 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-gray-700 dark:text-gray-200 truncate">{ds.filename}</p>
                <p className="text-xs text-gray-400">{ds.chunk_count} chunks · {ds.size_kb} KB</p>
              </div>
              <button
                onClick={() => onDatasetReady(ds.dataset_path)}
                className="text-xs text-brand-500 hover:text-brand-600 font-medium shrink-0"
                title="Use this dataset for training"
              >
                Use
              </button>
              <button
                onClick={() => deleteDataset(ds.dataset_id)}
                className="p-0.5 text-gray-300 hover:text-red-500 shrink-0"
              >
                <Trash2 size={11} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────

type PanelTab = 'train' | 'compare'

export default function FineTuningPanel() {
  const [activeTab, setActiveTab] = useState<PanelTab>('train')
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG)
  const [jobs, setJobs] = useState<FineTuneJob[]>([])
  const [loadingJobs, setLoadingJobs] = useState(false)
  const [starting, setStarting] = useState(false)
  const [showConfig, setShowConfig] = useState(true)

  const loadJobs = async () => {
    setLoadingJobs(true)
    try {
      setJobs(await fineTuneApi.listJobs())
    } finally {
      setLoadingJobs(false)
    }
  }

  const loadTemplate = async () => {
    try {
      const data = await fineTuneApi.getEHRTemplate()
      setConfig((c) => ({ ...c, training_examples: data.examples }))
      toast.success('EHR template loaded')
    } catch {
      toast.error('Failed to load template')
    }
  }

  const startTraining = async () => {
    if (!config.output_model_name.trim()) {
      toast.error('Please enter an output model name')
      return
    }
    if (!config.training_examples?.length && !config.dataset_path) {
      toast.error('Please add training examples or specify a dataset path')
      return
    }
    setStarting(true)
    try {
      await fineTuneApi.createJob(config)
      toast.success('Training job started!')
      await loadJobs()
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : 'Failed to start training')
    } finally {
      setStarting(false)
    }
  }

  const refreshJob = async (jobId: string) => {
    try {
      const updated = await fineTuneApi.getJob(jobId)
      setJobs((prev) => prev.map((j) => (j.job_id === jobId ? updated : j)))
    } catch {}
  }

  const cancelJob = async (jobId: string) => {
    try {
      await fineTuneApi.cancelJob(jobId)
      await loadJobs()
      toast.success('Job cancelled')
    } catch {
      toast.error('Failed to cancel job')
    }
  }

  useEffect(() => { loadJobs() }, [])

  return (
    <div className="flex flex-col h-full overflow-y-auto p-3 gap-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Zap size={18} className="text-brand-500" />
        <h2 className="text-sm font-semibold text-gray-800 dark:text-gray-100">Fine-Tuning</h2>
      </div>

      {/* Tab switcher */}
      <div className="flex gap-1 p-1 bg-gray-100 dark:bg-gray-800 rounded-lg">
        {([
          { id: 'train', label: 'Train', icon: <Zap size={12} /> },
          { id: 'compare', label: 'Compare', icon: <GitCompare size={12} /> },
        ] as const).map(({ id, label, icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={clsx(
              'flex-1 flex items-center justify-center gap-1.5 text-xs py-1.5 rounded-md font-medium transition-colors',
              activeTab === id
                ? 'bg-white dark:bg-gray-700 text-brand-600 dark:text-brand-400 shadow-sm'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
            )}
          >
            {icon}
            {label}
          </button>
        ))}
      </div>

      {/* Compare tab */}
      {activeTab === 'compare' && <ModelComparison />}

      {/* Train tab */}
      {activeTab === 'train' && <>

      {/* Config section */}
      <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
        <button
          className="w-full flex items-center justify-between p-2.5 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700"
          onClick={() => setShowConfig((v) => !v)}
        >
          <span className="text-xs font-semibold text-gray-600 dark:text-gray-300">Training Configuration</span>
          {showConfig ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>

        {showConfig && (
          <div className="p-3 space-y-3">
            {/* Model */}
            <div>
              <label className="text-xs text-gray-500 block mb-1">Base Model</label>
              <input
                value={config.base_model}
                onChange={(e) => setConfig((c) => ({ ...c, base_model: e.target.value }))}
                placeholder="meta-llama/Llama-3.2-3B-Instruct"
                className="w-full text-xs bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-200"
              />
            </div>

            <div>
              <label className="text-xs text-gray-500 block mb-1">Output Model Name</label>
              <input
                value={config.output_model_name}
                onChange={(e) => setConfig((c) => ({ ...c, output_model_name: e.target.value }))}
                placeholder="my-ehr-model"
                className="w-full text-xs bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-200"
              />
            </div>

            {/* Hyperparams */}
            <div className="grid grid-cols-2 gap-2">
              {[
                { key: 'num_epochs', label: 'Epochs', type: 'number', min: 1 },
                { key: 'batch_size', label: 'Batch Size', type: 'number', min: 1 },
                { key: 'learning_rate', label: 'Learning Rate', type: 'number', step: '0.0001' },
                { key: 'max_seq_length', label: 'Max Seq Len', type: 'number', min: 128 },
              ].map(({ key, label, ...rest }) => (
                <div key={key}>
                  <label className="text-xs text-gray-500 block mb-1">{label}</label>
                  <input
                    {...rest}
                    value={(config as any)[key] as number}
                    onChange={(e) =>
                      setConfig((c) => ({ ...c, [key]: parseFloat(e.target.value) }))
                    }
                    className="w-full text-xs bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-200"
                  />
                </div>
              ))}
            </div>

            {/* Checkboxes */}
            <div className="flex flex-wrap gap-3">
              {[
                { key: 'use_lora', label: 'LoRA' },
                { key: 'load_in_4bit', label: '4-bit QLoRA' },
                { key: 'fp16', label: 'FP16' },
                { key: 'gradient_checkpointing', label: 'Grad Checkpoint' },
              ].map(({ key, label }) => (
                <label key={key} className="flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={(config as any)[key] as boolean}
                    onChange={(e) =>
                      setConfig((c) => ({ ...c, [key]: e.target.checked }))
                    }
                    className="accent-brand-500"
                  />
                  {label}
                </label>
              ))}
            </div>

            {/* Training document upload */}
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <FileText size={12} className="text-brand-500" />
                <span className="text-xs font-semibold text-gray-600 dark:text-gray-300">
                  Upload Training Documents
                </span>
              </div>
              <TrainingDocumentUpload
                onDatasetReady={(path) => {
                  setConfig((c) => ({ ...c, dataset_path: path }))
                  toast.success('Dataset path set — ready to train!')
                }}
              />
            </div>

            {/* Dataset path alternative */}
            <div>
              <label className="text-xs text-gray-500 block mb-1">
                Dataset Path (or HF dataset name)
              </label>
              <input
                value={config.dataset_path || ''}
                onChange={(e) => setConfig((c) => ({ ...c, dataset_path: e.target.value }))}
                placeholder="e.g. medalpaca/medical_meadow_medqa or /path/to/data.jsonl"
                className="w-full text-xs bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1.5 text-gray-700 dark:text-gray-200"
              />
            </div>

            {/* Training examples */}
            <ExampleEditor
              examples={config.training_examples || []}
              onChange={(ex) => setConfig((c) => ({ ...c, training_examples: ex }))}
            />

            {/* Actions */}
            <div className="flex gap-2">
              <button
                onClick={loadTemplate}
                className="flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-300 border border-gray-200 dark:border-gray-600 rounded-lg px-3 py-1.5 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <BookOpen size={12} /> Load EHR Template
              </button>
              <button
                onClick={startTraining}
                disabled={starting}
                className="flex-1 flex items-center justify-center gap-1.5 text-xs bg-brand-500 hover:bg-brand-600 disabled:bg-brand-300 text-white rounded-lg px-3 py-1.5 font-medium transition-colors"
              >
                {starting ? (
                  <><Loader2 size={12} className="animate-spin" /> Starting…</>
                ) : (
                  <><Play size={12} /> Start Training</>
                )}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Jobs */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
            Training Jobs ({jobs.length})
          </span>
          <button onClick={loadJobs} className="p-1 text-gray-400 hover:text-brand-500">
            <RefreshCw size={12} className={loadingJobs ? 'animate-spin' : ''} />
          </button>
        </div>
        <div className="space-y-2">
          {jobs.length === 0 ? (
            <p className="text-xs text-gray-400 text-center py-4">No training jobs yet</p>
          ) : (
            jobs.map((job) => (
              <JobCard
                key={job.job_id}
                job={job}
                onRefresh={() => refreshJob(job.job_id)}
                onCancel={() => cancelJob(job.job_id)}
              />
            ))
          )}
        </div>
      </div>

      </>}
    </div>
  )
}
