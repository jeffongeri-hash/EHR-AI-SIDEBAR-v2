// ── Enums ─────────────────────────────────────────────────────────────────────

export type MessageRole = 'user' | 'assistant' | 'system'
export type ModelProvider = 'claude' | 'ollama' | 'huggingface' | 'tinyllama'
export type DocumentStatus = 'pending' | 'processing' | 'completed' | 'failed'
export type FineTuneStatus = 'queued' | 'preparing' | 'training' | 'completed' | 'failed'
export type OCREngine = 'easyocr' | 'tesseract' | 'auto'

// ── Document Types ────────────────────────────────────────────────────────────

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
  page: number
}

export interface TextBlock {
  text: string
  confidence: number
  bbox: BoundingBox
  block_type: string
  font_size?: number
  is_bold?: boolean
}

export interface TableCell {
  text: string
  row: number
  col: number
  row_span: number
  col_span: number
}

export interface Table {
  cells: TableCell[]
  rows: number
  cols: number
  bbox: BoundingBox
  page: number
}

export interface DocumentPage {
  page_number: number
  width: number
  height: number
  text_blocks: TextBlock[]
  tables: Table[]
  raw_text: string
  confidence: number
}

export interface DocumentMetadata {
  filename: string
  file_size: number
  file_type: string
  page_count: number
  created_at: string
  ocr_engine?: string
  processing_time_ms?: number
}

export interface ProcessedDocument {
  document_id: string
  metadata: DocumentMetadata
  pages: DocumentPage[]
  full_text: string
  status: DocumentStatus
  error?: string
}

export interface DocumentUploadResponse {
  document_id: string
  filename: string
  status: DocumentStatus
  message: string
}

export interface DocumentSummary {
  document_id: string
  filename: string
  page_count: number
  status: DocumentStatus
  created_at: string
}

// ── Chat Types ────────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: MessageRole
  content: string
  timestamp: string
  document_ids?: string[]
  metadata?: Record<string, unknown>
}

export interface ChatRequest {
  message: string
  conversation_id?: string
  document_ids?: string[]
  model_provider: ModelProvider
  model_name?: string
  system_prompt?: string
  temperature: number
  max_tokens: number
  stream?: boolean
}

export interface ChatResponse {
  conversation_id: string
  message: ChatMessage
  model_used: string
  usage?: { input_tokens: number; output_tokens: number }
  document_context?: string[]
}

export interface ConversationHistory {
  conversation_id: string
  messages: ChatMessage[]
  created_at: string
  updated_at: string
  document_ids: string[]
}

// ── Model Types ───────────────────────────────────────────────────────────────

export interface ModelInfo {
  name: string
  provider: ModelProvider
  display_name: string
  context_length: number
  supports_vision: boolean
  is_local: boolean
  size_gb?: number
  description?: string
}

// ── Fine-Tuning Types ─────────────────────────────────────────────────────────

export interface TrainingExample {
  instruction: string
  input?: string
  output: string
  system?: string
}

export interface LoRAConfig {
  r: number
  lora_alpha: number
  lora_dropout: number
  target_modules: string[]
  bias: string
}

export interface TrainingConfig {
  base_model: string
  output_model_name: string
  dataset_path?: string
  training_examples?: TrainingExample[]
  num_epochs: number
  batch_size: number
  learning_rate: number
  warmup_steps: number
  max_seq_length: number
  gradient_accumulation_steps: number
  use_lora: boolean
  lora_config: LoRAConfig
  load_in_4bit: boolean
  fp16: boolean
  gradient_checkpointing: boolean
  push_to_hub: boolean
  hf_repo_id?: string
}

export interface TrainingDataset {
  dataset_id: string
  filename: string
  dataset_path: string
  chunk_count: number
  size_kb: number
}

export interface FineTuneJob {
  job_id: string
  config: TrainingConfig
  status: FineTuneStatus
  progress: number
  current_epoch: number
  current_loss?: number
  created_at: string
  started_at?: string
  completed_at?: string
  error?: string
  output_path?: string
  logs: string[]
}

// ── Model Comparison Types ────────────────────────────────────────────────────

export interface ModelCompareRequest {
  prompt: string
  base_model: string
  fine_tuned_model: string
  max_tokens?: number
  temperature?: number
  system_prompt?: string
}

export interface ModelCompareResult {
  model_name: string
  response: string
  latency_ms: number
  tokens_generated?: number
  error?: string
}

export interface ModelCompareResponse {
  prompt: string
  base: ModelCompareResult
  fine_tuned: ModelCompareResult
}

// ── UI State Types ────────────────────────────────────────────────────────────

export type Tab = 'home' | 'documents' | 'fine-tune' | 'settings'

export interface UIMessage extends ChatMessage {
  id: string
  isStreaming?: boolean
  error?: string
}
