import React, { useState, useEffect } from 'react'
import {
  CheckCircle,
  Eye,
  EyeOff,
  FileText,
  Lock,
  Save,
  Settings,
  Shield,
  ShieldCheck,
  AlertTriangle,
  Wifi,
  WifiOff,
} from 'lucide-react'
import clsx from 'clsx'
import { useToast } from '@/contexts/ToastContext'
import { useApp } from '@/contexts/AppContext'

const LS_API_KEY = 'ehr_api_key'
const LS_ANTHROPIC_KEY = 'anthropic_api_key'
const LS_OLLAMA_URL = 'ollama_base_url'
const LS_HF_TOKEN = 'hf_token'

interface ConnectionStatus {
  backend: boolean
  ollama: boolean
  anthropic: boolean
}

export default function SettingsPanel() {
  const { state } = useApp()
  const { showSuccess, showError, showWarning } = useToast()

  const [apiKey, setApiKey] = useState(() => localStorage.getItem(LS_API_KEY) ?? '')
  const [anthropicKey, setAnthropicKey] = useState(() => localStorage.getItem(LS_ANTHROPIC_KEY) ?? '')
  const [ollamaUrl, setOllamaUrl] = useState(() => localStorage.getItem(LS_OLLAMA_URL) ?? 'http://localhost:11434')
  const [hfToken, setHfToken] = useState(() => localStorage.getItem(LS_HF_TOKEN) ?? '')
  const [saved, setSaved] = useState(false)
  const [testing, setTesting] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>({
    backend: false,
    ollama: false,
    anthropic: false
  })
  const [show, setShow] = useState<Record<string, boolean>>({})

  const toggle = (k: string) => setShow((s) => ({ ...s, [k]: !s[k] }))

  // Test connections on mount
  useEffect(() => {
    testConnections()
  }, [])

  const testConnections = async () => {
    setTesting(true)
    const newStatus: ConnectionStatus = {
      backend: false,
      ollama: false,
      anthropic: false
    }

    try {
      // Test backend
      const backendResponse = await fetch('/api/health')
      newStatus.backend = backendResponse.ok
    } catch {
      newStatus.backend = false
    }

    try {
      // Test Ollama
      const ollamaResponse = await fetch(`${ollamaUrl}/api/tags`)
      newStatus.ollama = ollamaResponse.ok
    } catch {
      newStatus.ollama = false
    }

    // Anthropic test would require making an API call, so we'll just check if key exists
    newStatus.anthropic = anthropicKey.trim().length > 0

    setConnectionStatus(newStatus)
    setTesting(false)
  }

  const validateInputs = () => {
    const errors: string[] = []

    if (ollamaUrl && !ollamaUrl.match(/^https?:\/\/.+/)) {
      errors.push('Ollama URL must be a valid HTTP/HTTPS URL')
    }

    if (anthropicKey && !anthropicKey.startsWith('sk-ant-')) {
      errors.push('Anthropic API key should start with "sk-ant-"')
    }

    if (hfToken && !hfToken.startsWith('hf_')) {
      errors.push('HuggingFace token should start with "hf_"')
    }

    return errors
  }

  const handleSave = () => {
    const validationErrors = validateInputs()
    if (validationErrors.length > 0) {
      showError('Validation failed', validationErrors.join('. '))
      return
    }

    // Persist settings
    if (apiKey.trim()) {
      localStorage.setItem(LS_API_KEY, apiKey.trim())
    } else {
      localStorage.removeItem(LS_API_KEY)
    }

    if (anthropicKey.trim()) {
      localStorage.setItem(LS_ANTHROPIC_KEY, anthropicKey.trim())
    } else {
      localStorage.removeItem(LS_ANTHROPIC_KEY)
    }

    if (ollamaUrl.trim()) {
      localStorage.setItem(LS_OLLAMA_URL, ollamaUrl.trim())
    } else {
      localStorage.removeItem(LS_OLLAMA_URL)
    }

    if (hfToken.trim()) {
      localStorage.setItem(LS_HF_TOKEN, hfToken.trim())
    } else {
      localStorage.removeItem(LS_HF_TOKEN)
    }

    setSaved(true)
    showSuccess('Settings saved', 'Reload the page to apply changes')
    setTimeout(() => setSaved(false), 3000)
    
    // Re-test connections after save
    testConnections()
  }

  const apiKeySet = !!apiKey.trim()

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div className="flex-shrink-0 px-5 py-3 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
        <h1 className="text-sm font-semibold text-gray-800 dark:text-gray-100">Settings</h1>
        <p className="text-xs text-gray-400 mt-0.5">API keys, security, and model defaults</p>
      </div>

      {/* Connection Status */}
      <div className="flex-shrink-0 px-5 py-3 bg-gray-50 dark:bg-gray-800/50 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
            Connection Status
          </span>
          <button
            onClick={testConnections}
            disabled={testing}
            className="text-xs text-brand-600 dark:text-brand-400 hover:text-brand-700 dark:hover:text-brand-300 disabled:opacity-50"
          >
            {testing ? 'Testing...' : 'Refresh'}
          </button>
        </div>
        <div className="grid grid-cols-3 gap-2">
          <div className={clsx(
            'flex items-center gap-1 text-xs p-1.5 rounded',
            connectionStatus.backend
              ? 'text-green-700 dark:text-green-300 bg-green-100 dark:bg-green-900/20'
              : 'text-red-700 dark:text-red-300 bg-red-100 dark:bg-red-900/20'
          )}>
            {connectionStatus.backend ? <Wifi size={12} /> : <WifiOff size={12} />}
            <span>Backend</span>
          </div>
          <div className={clsx(
            'flex items-center gap-1 text-xs p-1.5 rounded',
            connectionStatus.ollama
              ? 'text-green-700 dark:text-green-300 bg-green-100 dark:bg-green-900/20'
              : 'text-red-700 dark:text-red-300 bg-red-100 dark:bg-red-900/20'
          )}>
            {connectionStatus.ollama ? <Wifi size={12} /> : <WifiOff size={12} />}
            <span>Ollama</span>
          </div>
          <div className={clsx(
            'flex items-center gap-1 text-xs p-1.5 rounded',
            connectionStatus.anthropic
              ? 'text-green-700 dark:text-green-300 bg-green-100 dark:bg-green-900/20'
              : 'text-yellow-700 dark:text-yellow-300 bg-yellow-100 dark:bg-yellow-900/20'
          )}>
            {connectionStatus.anthropic ? <CheckCircle size={12} /> : <AlertTriangle size={12} />}
            <span>Claude</span>
          </div>
        </div>
      </div>

      <div className="flex-1 p-5 space-y-6">

        {/* ── HIPAA Security section ─────────────────────────────────────── */}
        <section>
          <div className="flex items-center gap-2 mb-3">
            <Shield size={14} className="text-brand-500" />
            <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              HIPAA Security
            </span>
          </div>

          {/* Security status badges */}
          <div className="grid grid-cols-2 gap-2 mb-4">
            <div className={`flex items-center gap-2 p-2.5 rounded-lg border text-xs ${
              apiKeySet
                ? 'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800'
                : 'bg-amber-50 border-amber-200 dark:bg-amber-900/20 dark:border-amber-800'
            }`}>
              {apiKeySet
                ? <ShieldCheck size={13} className="text-green-500" />
                : <Lock size={13} className="text-amber-500" />
              }
              <span className={apiKeySet ? 'text-green-700 dark:text-green-300' : 'text-amber-700 dark:text-amber-300'}>
                {apiKeySet ? 'Auth enabled' : 'No API key set'}
              </span>
            </div>
            <div className="flex items-center gap-2 p-2.5 rounded-lg border bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800 text-xs">
              <ShieldCheck size={13} className="text-blue-500" />
              <span className="text-blue-700 dark:text-blue-300">TLS headers on</span>
            </div>
          </div>

          {/* EHR API Key */}
          <div>
            <label className="text-xs font-medium text-gray-700 dark:text-gray-300 block mb-1">
              EHR API Key <span className="text-red-400">*</span>
            </label>
            <div className="relative">
              <input
                type={show['api_key'] ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Paste your EHR_API_KEY here…"
                className="w-full text-sm bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-2 pr-9 text-gray-700 dark:text-gray-200 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-brand-400"
              />
              <button
                onClick={() => toggle('api_key')}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                {show['api_key'] ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
            <p className="text-xs text-gray-400 mt-1">
              Must match <code className="font-mono bg-gray-100 dark:bg-gray-700 px-1 rounded">EHR_API_KEY</code> in your backend <code className="font-mono bg-gray-100 dark:bg-gray-700 px-1 rounded">.env</code>.
              Sent as the <code className="font-mono bg-gray-100 dark:bg-gray-700 px-1 rounded">X-API-Key</code> header on every request.
            </p>
          </div>
        </section>

        {/* ── AI API Keys ───────────────────────────────────────────────── */}
        <section>
          <div className="flex items-center gap-2 mb-3">
            <Settings size={14} className="text-gray-400" />
            <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              AI API Keys &amp; Endpoints
            </span>
          </div>

          {[
            { id: 'anthropic', label: 'Anthropic API Key', val: anthropicKey, set: setAnthropicKey, placeholder: 'sk-ant-…', hint: 'Required for Claude models' },
            { id: 'ollama', label: 'Ollama Base URL', val: ollamaUrl, set: setOllamaUrl, placeholder: 'http://localhost:11434', hint: 'URL of your local Ollama instance', plain: true },
            { id: 'hf', label: 'HuggingFace Token', val: hfToken, set: setHfToken, placeholder: 'hf_…', hint: 'Required to download gated models (e.g. Llama 3)' },
          ].map(({ id, label, val, set, placeholder, hint, plain }) => (
            <div key={id} className="mb-4">
              <label className="text-xs font-medium text-gray-700 dark:text-gray-300 block mb-1">{label}</label>
              <div className="relative">
                <input
                  type={plain ? 'text' : show[id] ? 'text' : 'password'}
                  value={val}
                  onChange={(e) => set(e.target.value)}
                  placeholder={placeholder}
                  className="w-full text-sm bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-2 pr-9 text-gray-700 dark:text-gray-200 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-brand-400"
                />
                {!plain && (
                  <button onClick={() => toggle(id)} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600">
                    {show[id] ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                )}
              </div>
              {hint && <p className="text-xs text-gray-400 mt-0.5">{hint}</p>}
            </div>
          ))}
        </section>

        {/* ── Model Defaults ────────────────────────────────────────────── */}
        <section>
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Model Defaults</span>
          </div>
          <div>
            <label className="text-xs font-medium text-gray-700 dark:text-gray-300 block mb-1">Default Provider</label>
            <select
              value={state.selectedProvider}
              className="w-full text-sm bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-2 text-gray-700 dark:text-gray-200"
            >
              <option value="claude">Anthropic Claude</option>
              <option value="ollama">Ollama (Local)</option>
            </select>
          </div>
        </section>

        {/* Save */}
        <button
          onClick={handleSave}
          className="flex items-center justify-center gap-2 w-full bg-brand-500 hover:bg-brand-600 text-white rounded-xl px-4 py-2.5 text-sm font-medium transition-colors"
        >
          {saved ? <CheckCircle size={16} /> : <Save size={16} />}
          {saved ? 'Saved!' : 'Save Settings'}
        </button>

        {/* Env file reference */}
        <div className="text-xs text-gray-400 bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3 space-y-1.5">
          <p className="font-semibold text-gray-500 dark:text-gray-400 flex items-center gap-1.5">
            <FileText size={12} /> Backend .env reference
          </p>
          {[
            'EHR_API_KEY=<hex-32-bytes>',
            'EHR_ENCRYPTION_KEY=<base64-32-bytes>',
            'ENCRYPT_UPLOADS=true',
            'ANTHROPIC_API_KEY=sk-ant-…',
            'OLLAMA_BASE_URL=http://localhost:11434',
          ].map((line) => (
            <p key={line} className="font-mono text-gray-500 text-xs">{line}</p>
          ))}
          <p className="text-gray-400 text-xs mt-1">
            Generate keys: <code className="font-mono bg-gray-100 dark:bg-gray-700 px-1 rounded">python -c "import secrets; print(secrets.token_hex(32))"</code>
          </p>
        </div>
      </div>
    </div>
  )
}
