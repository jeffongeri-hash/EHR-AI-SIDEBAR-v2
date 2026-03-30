import React from 'react'
import { AppProvider, useApp } from '@/contexts/AppContext'
import { ToastProvider } from '@/contexts/ToastContext'
import ErrorBoundary from '@/components/ErrorBoundary'
import Sidebar from '@/components/Sidebar'
import Dashboard from '@/components/Dashboard'
import ChatInterface from '@/components/ChatInterface'
import DocumentUpload from '@/components/DocumentUpload'
import FineTuningPanel from '@/components/FineTuningPanel'
import SettingsPanel from '@/components/SettingsPanel'

// ── Branded panel header ───────────────────────────────────────────────────────
function PanelHeader({
  title,
  subtitle,
  badge,
  badgeColor = 'cyan',
}: {
  title: string
  subtitle: string
  badge?: string
  badgeColor?: 'cyan' | 'blue' | 'violet' | 'gray'
}) {
  const badgeStyles: Record<string, string> = {
    cyan:   'bg-cyan-400/20 text-cyan-300 border border-cyan-500/30',
    blue:   'bg-blue-500/20 text-blue-300 border border-blue-500/30',
    violet: 'bg-violet-500/20 text-violet-300 border border-violet-500/30',
    gray:   'bg-gray-700 text-gray-300 border border-gray-600',
  }
  return (
    <div className="flex-shrink-0 flex items-center justify-between px-5 py-3.5 bg-gray-900 border-b border-gray-800">
      <div>
        <h1 className="text-sm font-semibold text-gray-100">{title}</h1>
        <p className="text-xs text-gray-500 mt-0.5">{subtitle}</p>
      </div>
      {badge && (
        <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${badgeStyles[badgeColor]}`}>
          {badge}
        </span>
      )}
    </div>
  )
}

// ── Left panel content (switches per tab) ─────────────────────────────────────
function LeftPanel() {
  const { state } = useApp()

  switch (state.activeTab) {
    case 'home':
      return <Dashboard />

    case 'documents':
      return (
        <div className="flex flex-col h-full">
          <PanelHeader
            title="Documents"
            subtitle="Upload PDFs, DOCX files, or medical images for OCR & analysis"
            badge="OCR Enabled"
            badgeColor="cyan"
          />
          <div className="flex-1 overflow-hidden">
            <DocumentUpload />
          </div>
        </div>
      )

    case 'fine-tune':
      return (
        <div className="flex flex-col h-full">
          <PanelHeader
            title="Fine-Tuning"
            subtitle="Train Llama or DeepSeek on your own EHR documents"
            badge="GPU Training"
            badgeColor="violet"
          />
          <div className="flex-1 overflow-hidden">
            <FineTuningPanel />
          </div>
        </div>
      )

    case 'settings':
      return <SettingsPanel />

    default:
      return <Dashboard />
  }
}

// ── Root layout ───────────────────────────────────────────────────────────────
function AppContent() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-gray-950">
      {/* Navigation sidebar */}
      <ErrorBoundary label="Navigation">
        <Sidebar />
      </ErrorBoundary>

      {/* Left content panel */}
      <div className="flex-1 flex flex-col overflow-hidden bg-gray-900 border-r border-gray-800 min-w-0">
        <ErrorBoundary label="Content panel">
          <LeftPanel />
        </ErrorBoundary>
      </div>

      {/* Right chat panel — always visible */}
      <div className="w-[400px] flex-shrink-0 flex flex-col bg-gray-900 border-l border-gray-800">
        <ErrorBoundary label="Chat">
          <ChatInterface />
        </ErrorBoundary>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <AppProvider>
      <ToastProvider>
        <AppContent />
      </ToastProvider>
    </AppProvider>
  )
}
