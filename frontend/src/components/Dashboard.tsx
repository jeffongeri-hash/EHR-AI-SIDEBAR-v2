import React, { useEffect, useState } from 'react'
import {
  Activity,
  Bot,
  CheckCircle2,
  Clock,
  Cpu,
  FileText,
  FolderOpen,
  UploadCloud,
  XCircle,
  Zap,
} from 'lucide-react'
import clsx from 'clsx'
import { useApp } from '@/contexts/AppContext'
import { fineTuneApi, healthApi } from '@/services/api'

// ── Helpers ───────────────────────────────────────────────────────────────────

function greeting() {
  const h = new Date().getHours()
  if (h < 12) return 'Good morning'
  if (h < 17) return 'Good afternoon'
  return 'Good evening'
}

function fmtDate() {
  return new Date().toLocaleDateString('en-US', {
    weekday: 'long', month: 'long', day: 'numeric', year: 'numeric',
  })
}

function fmtTime(iso: string) {
  try {
    return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  } catch {
    return '—'
  }
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusDot({ ok }: { ok: boolean | null }) {
  if (ok === null) return <span className="w-2 h-2 rounded-full bg-gray-300 animate-pulse" />
  return (
    <span className={clsx(
      'w-2 h-2 rounded-full',
      ok ? 'bg-green-400' : 'bg-red-400'
    )} />
  )
}

function StatusCard({
  icon,
  label,
  status,
  detail,
}: {
  icon: React.ReactNode
  label: string
  status: boolean | null
  detail: string
}) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-xl bg-gray-800 border border-gray-700">
      <div className="w-8 h-8 rounded-lg bg-gray-700 border border-gray-600 flex items-center justify-center text-gray-500 dark:text-gray-400 flex-shrink-0">
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs font-semibold text-gray-200">{label}</p>
        <p className="text-xs text-gray-400 truncate">{detail}</p>
      </div>
      <StatusDot ok={status} />
    </div>
  )
}

function StatPill({ value, label }: { value: number | string; label: string }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center py-3 px-2 rounded-xl bg-gray-800 border border-gray-700">
      <span className="text-2xl font-bold text-gray-100">{value}</span>
      <span className="text-xs text-gray-400 mt-0.5">{label}</span>
    </div>
  )
}

function QuickAction({
  icon,
  label,
  description,
  onClick,
  color,
}: {
  icon: React.ReactNode
  label: string
  description: string
  onClick: () => void
  color: string
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 w-full p-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:border-brand-300 dark:hover:border-brand-600 hover:shadow-sm transition-all text-left group"
    >
      <div className={clsx('w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 transition-transform group-hover:scale-110', color)}>
        {icon}
      </div>
      <div>
        <p className="text-sm font-semibold text-gray-100">{label}</p>
        <p className="text-xs text-gray-400">{description}</p>
      </div>
    </button>
  )
}

// ── Main Dashboard ─────────────────────────────────────────────────────────────

export default function Dashboard() {
  const { state, dispatch } = useApp()

  const [backendOk, setBackendOk] = useState<boolean | null>(null)
  const [ollamaOk, setOllamaOk] = useState<boolean | null>(null)
  const [claudeOk, setClaudeOk] = useState<boolean | null>(null)
  const [jobCount, setJobCount] = useState(0)

  useEffect(() => {
    // Backend health
    healthApi.check()
      .then(() => setBackendOk(true))
      .catch(() => setBackendOk(false))

    // Ollama: look for local models already in state
    // Also use models endpoint as a proxy for Ollama
    const hasLocal = state.availableModels.some((m) => m.is_local)
    setOllamaOk(hasLocal ? true : null)

    // Claude: check if any claude models available
    const hasClaude = state.availableModels.some((m) => m.provider === 'claude')
    setClaudeOk(hasClaude ? true : null)

    // Fine-tune jobs
    fineTuneApi.listJobs()
      .then((jobs) => setJobCount(jobs.length))
      .catch(() => {})
  }, [state.availableModels])

  const docCount = state.documents.length
  const completedDocs = state.documents.filter((d) => d.status === 'completed').length
  const modelCount = state.availableModels.length

  const recentDocs = [...state.documents]
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
    .slice(0, 5)

  return (
    <div className="flex flex-col h-full overflow-y-auto bg-gray-950">
      {/* Header */}
      <div className="px-6 pt-6 pb-4 bg-gray-900 border-b border-gray-800">
        <p className="text-xs text-gray-500 mb-0.5">{fmtDate()}</p>
        <h1 className="text-xl font-bold text-gray-50">{greeting()}</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          EHR AI Sidebar — document analysis &amp; clinical AI
        </p>
      </div>

      <div className="flex-1 p-6 space-y-6">

        {/* System Status */}
        <section>
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
            System Status
          </h2>
          <div className="grid grid-cols-3 gap-3">
            <StatusCard
              icon={<Activity size={15} />}
              label="Backend"
              status={backendOk}
              detail={backendOk === null ? 'Checking…' : backendOk ? 'API online' : 'Unreachable'}
            />
            <StatusCard
              icon={<Cpu size={15} />}
              label="Ollama"
              status={ollamaOk}
              detail={ollamaOk ? 'Local models ready' : 'Pull a model to start'}
            />
            <StatusCard
              icon={<Bot size={15} />}
              label="Claude API"
              status={claudeOk}
              detail={claudeOk ? 'Connected' : 'Add API key in Settings'}
            />
          </div>
        </section>

        {/* Stats */}
        <section>
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
            Overview
          </h2>
          <div className="flex gap-3">
            <StatPill value={docCount} label="Documents" />
            <StatPill value={completedDocs} label="Processed" />
            <StatPill value={jobCount} label="Training Jobs" />
            <StatPill value={modelCount} label="Models" />
          </div>
        </section>

        {/* Quick Actions */}
        <section>
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
            Quick Actions
          </h2>
          <div className="grid grid-cols-2 gap-3">
            <QuickAction
              icon={<UploadCloud size={18} className="text-white" />}
              label="Upload Document"
              description="PDF, DOCX, or image"
              color="bg-brand-500"
              onClick={() => dispatch({ type: 'SET_TAB', payload: 'documents' })}
            />
            <QuickAction
              icon={<Zap size={18} className="text-white" />}
              label="Fine-Tune Model"
              description="Train on your EHR data"
              color="bg-purple-500"
              onClick={() => dispatch({ type: 'SET_TAB', payload: 'fine-tune' })}
            />
          </div>
        </section>

        {/* Recent Documents */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Recent Documents
            </h2>
            {docCount > 0 && (
              <button
                onClick={() => dispatch({ type: 'SET_TAB', payload: 'documents' })}
                className="text-xs text-blue-400 hover:text-blue-300 font-medium"
              >
                View all
              </button>
            )}
          </div>

          {recentDocs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 rounded-xl border-2 border-dashed border-gray-700">
              <FolderOpen size={28} className="text-gray-600 mb-2" />
              <p className="text-sm text-gray-400">No documents yet</p>
              <button
                onClick={() => dispatch({ type: 'SET_TAB', payload: 'documents' })}
                className="mt-2 text-xs text-blue-400 hover:text-blue-300 font-medium"
              >
                Upload your first document →
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              {recentDocs.map((doc) => (
                <div
                  key={doc.document_id}
                  onClick={() => dispatch({ type: 'SET_TAB', payload: 'documents' })}
                  className="flex items-center gap-3 p-3 rounded-xl bg-gray-800 border border-gray-700 hover:border-blue-600 hover:shadow-sm transition-all cursor-pointer"
                >
                  <div className="w-8 h-8 rounded-lg bg-blue-900/20 flex items-center justify-center flex-shrink-0">
                    <FileText size={15} className="text-brand-500" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-100 truncate">
                      {doc.filename}
                    </p>
                    <p className="text-xs text-gray-400">
                      {doc.page_count} {doc.page_count === 1 ? 'page' : 'pages'} · {fmtTime(doc.created_at)}
                    </p>
                  </div>
                  {doc.status === 'completed' ? (
                    <CheckCircle2 size={15} className="text-green-500 flex-shrink-0" />
                  ) : doc.status === 'failed' ? (
                    <XCircle size={15} className="text-red-400 flex-shrink-0" />
                  ) : (
                    <Clock size={15} className="text-gray-300 flex-shrink-0 animate-pulse" />
                  )}
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Getting started tip when empty */}
        {docCount === 0 && (
          <section className="rounded-xl bg-blue-900/20 border border-blue-800/50 p-4">
            <p className="text-sm font-semibold text-blue-300 mb-1">
              Getting started
            </p>
            <ol className="space-y-1 text-xs text-blue-400 list-decimal list-inside">
              <li>Upload a medical document (PDF, DOCX, or scanned image)</li>
              <li>Select it in the Documents tab to attach it to chat</li>
              <li>Ask the AI questions about it in the chat panel →</li>
              <li>Optionally fine-tune Llama on your own documents</li>
            </ol>
          </section>
        )}
      </div>
    </div>
  )
}
