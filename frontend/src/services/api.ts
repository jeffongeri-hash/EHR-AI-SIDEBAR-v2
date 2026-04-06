import axios, { AxiosInstance } from 'axios'
import type {
  ChatRequest,
  ChatResponse,
  ConversationHistory,
  DocumentSummary,
  DocumentUploadResponse,
  FineTuneJob,
  ModelCompareRequest,
  ModelCompareResponse,
  ModelInfo,
  ProcessedDocument,
  TrainingConfig,
  TrainingDataset,
} from '@/types'
import { clearAuth, getToken } from '@/services/auth'

// Original EHR AI Sidebar backend with full document upload and chat functionality
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

const http: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000, // Restored original timeout for document processing
})

// Attach JWT Bearer token from in-memory store (never from localStorage)
http.interceptors.request.use((config) => {
  const token = getToken()
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`
  }
  return config
})

// On 401, clear the in-memory token so the app shows the login screen
http.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      clearAuth()
      // Emit a custom event so AppContext can react without a circular import
      window.dispatchEvent(new CustomEvent('ehr:unauthorized'))
    }
    return Promise.reject(err)
  }
)

// ── Chat ──────────────────────────────────────────────────────────────────────

export const chatApi = {
  send: (req: ChatRequest) =>
    http.post<ChatResponse>('/chat/', req).then((r) => r.data),

  getConversation: (id: string) =>
    http.get<ConversationHistory>(`/chat/conversations/${id}`).then((r) => r.data),

  clearConversation: (id: string) =>
    http.delete(`/chat/conversations/${id}`).then((r) => r.data),

  listModels: () =>
    http.get<{ models: ModelInfo[] }>('/chat/models').then((r) => r.data.models),

  /** Returns an EventSource for SSE streaming. */
  streamChat: (req: ChatRequest): EventSource => {
    // Build query string for GET-based SSE (or POST via fetch for full support)
    const url = `${BASE_URL}/chat/stream`
    // Use fetch + ReadableStream for POST SSE
    return new EventSource(url)
  },

  /** POST-based streaming using fetch ReadableStream */
  streamChatFetch: async function* (
    req: ChatRequest,
    signal?: AbortSignal
  ): AsyncGenerator<string> {
    const token = getToken()
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    if (token) headers['Authorization'] = `Bearer ${token}`

    const response = await fetch(`${BASE_URL}/chat/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(req),
      signal,
    })

    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    if (!response.body) return

    const reader = response.body.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const text = decoder.decode(value, { stream: true })
      const lines = text.split('\n')
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6)
          if (data === '[DONE]') return
          if (data.startsWith('[ERROR]')) throw new Error(data.slice(8))
          yield data
        }
      }
    }
  },
}

// ── Documents ─────────────────────────────────────────────────────────────────

export const documentsApi = {
  upload: (
    file: File,
    options: {
      ocr_engine?: string
      enhance_images?: boolean
      languages?: string
      extract_tables?: boolean
    } = {}
  ) => {
    const form = new FormData()
    form.append('file', file)
    form.append('ocr_engine', options.ocr_engine || 'auto')
    form.append('enhance_images', String(options.enhance_images ?? true))
    form.append('languages', options.languages || 'en')
    form.append('extract_tables', String(options.extract_tables ?? true))
    return http
      .post<DocumentUploadResponse>('/documents/upload', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300_000,
      })
      .then((r) => r.data)
  },

  list: () =>
    http.get<DocumentSummary[]>('/documents/').then((r) => r.data),

  get: (id: string) =>
    http.get<ProcessedDocument>(`/documents/${id}`).then((r) => r.data),

  getText: (id: string, page?: number) =>
    http
      .get<{ text: string }>(`/documents/${id}/text`, { params: { page } })
      .then((r) => r.data.text),

  getTables: (id: string) =>
    http.get(`/documents/${id}/tables`).then((r) => r.data),

  delete: (id: string) =>
    http.delete(`/documents/${id}`).then((r) => r.data),
}

// ── Fine-Tuning ───────────────────────────────────────────────────────────────

export const fineTuneApi = {
  createJob: (config: TrainingConfig) =>
    http.post<{ job_id: string; status: string; message: string }>('/fine-tune/jobs', config).then((r) => r.data),

  listJobs: () =>
    http.get<FineTuneJob[]>('/fine-tune/jobs').then((r) => r.data),

  getJob: (id: string) =>
    http.get<FineTuneJob>(`/fine-tune/jobs/${id}`).then((r) => r.data),

  cancelJob: (id: string) =>
    http.delete(`/fine-tune/jobs/${id}`).then((r) => r.data),

  listTrainedModels: () =>
    http.get<Array<{ name: string; path: string; size_mb: number }>>('/fine-tune/models').then((r) => r.data),

  getEHRTemplate: () =>
    http.get('/fine-tune/templates/ehr').then((r) => r.data),

  getRecommendedConfigs: () =>
    http.get('/fine-tune/configs/recommended').then((r) => r.data),

  uploadDocument: (
    file: File,
    options: { chunk_size?: number; overlap?: number; format?: 'text' | 'instruction' } = {}
  ) => {
    const form = new FormData()
    form.append('file', file)
    form.append('chunk_size', String(options.chunk_size ?? 512))
    form.append('overlap', String(options.overlap ?? 64))
    form.append('format', options.format ?? 'text')
    return http
      .post<{
        dataset_id: string
        dataset_path: string
        filename: string
        chunk_count: number
        page_count: number
        format: string
        preview: string
      }>('/fine-tune/upload-document', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300_000,
      })
      .then((r) => r.data)
  },

  listDatasets: () =>
    http.get<TrainingDataset[]>('/fine-tune/datasets').then((r) => r.data),

  deleteDataset: (datasetId: string) =>
    http.delete(`/fine-tune/datasets/${datasetId}`).then((r) => r.data),

  compareModels: (req: ModelCompareRequest) =>
    http
      .post<ModelCompareResponse>('/fine-tune/compare', req, { timeout: 600_000 })
      .then((r) => r.data),
}

// ── Health ────────────────────────────────────────────────────────────────────

export const healthApi = {
  check: () => http.get('/health').then((r) => r.data),
}

// ── Optimized Medical AI ─────────────────────────────────────────────────────
// Ultra-lightweight medical AI functions for 8GB macOS systems

export const medicalApi = {
  // Standard medical chat with built-in protocols
  chat: async (message: string, model = 'medical_expert') => {
    const response = await http.post('/chat', {
      message,
      model
    })
    return response.data
  },

  // Emergency medical protocols
  emergency: async (message: string) => {
    const response = await http.post('/chat/emergency', {
      message
    })
    return response.data
  },

  // List available medical protocols
  protocols: async () => {
    const response = await http.get('/protocols')
    return response.data
  },

  // Get available lightweight models
  models: async () => {
    const response = await http.get('/models')
    return response.data
  },

  // Memory usage information
  memory: async () => {
    const response = await http.get('/memory')
    return response.data
  }
}

// ── Compatible Chat Function ──────────────────────────────────────────────────
// Adapter function to make the ultra-lightweight backend compatible with existing UI

export const compatibleChatStream = async (
  req: ChatRequest,
  onMessage: (text: string) => void,
  signal?: AbortSignal
) => {
  try {
    // Use the medical chat API with the user's message
    const response = await medicalApi.chat(req.message, 'medical_expert')
    
    // Simulate streaming for UI compatibility
    const text = response.response
    const words = text.split(' ')
    
    for (let i = 0; i < words.length; i++) {
      if (signal?.aborted) break
      
      // Stream words gradually to simulate real-time response
      const partial = words.slice(0, i + 1).join(' ')
      onMessage(partial)
      
      // Small delay to simulate streaming
      await new Promise(resolve => setTimeout(resolve, 50))
    }
    
    return response
  } catch (error) {
    console.error('Medical chat error:', error)
    throw error
  }
}
