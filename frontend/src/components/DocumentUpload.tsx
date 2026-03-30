import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  FileText,
  Image,
  Upload,
  X,
  CheckCircle,
  AlertCircle,
  Loader2,
  Trash2,
  Eye,
  ChevronDown,
  ChevronUp,
  CloudUpload,
} from 'lucide-react'
import clsx from 'clsx'
import { useToast } from '@/contexts/ToastContext'
import { documentsApi } from '@/services/api'
import { useApp } from '@/contexts/AppContext'
import type { DocumentSummary } from '@/types'

const ACCEPTED: Record<string, string[]> = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/tiff': ['.tiff', '.tif'],
  'image/bmp': ['.bmp'],
  'image/webp': ['.webp'],
}

function FileIcon({ type }: { type: string }) {
  if (type.includes('pdf')) return <FileText size={16} className="text-red-500" />
  if (type.includes('word') || type.includes('docx')) return <FileText size={16} className="text-blue-600" />
  return <Image size={16} className="text-blue-500" />
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'completed': return <CheckCircle size={14} className="text-green-500" />
    case 'failed': return <AlertCircle size={14} className="text-red-500" />
    case 'processing':
    case 'pending': return <Loader2 size={14} className="text-brand-500 animate-spin" />
    default: return null
  }
}

function DocCard({ doc, selected, onToggle, onDelete }: {
  doc: DocumentSummary
  selected: boolean
  onToggle: () => void
  onDelete: () => void
}) {
  const [showInfo, setShowInfo] = useState(false)

  return (
    <div
      className={clsx(
        'rounded-lg border p-2.5 transition-all cursor-pointer',
        selected
          ? 'border-brand-400 bg-brand-50 dark:bg-brand-900/20'
          : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:border-brand-300'
      )}
      onClick={onToggle}
    >
      <div className="flex items-start gap-2">
        <FileIcon type={doc.filename} />
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-gray-800 dark:text-gray-100 truncate">
            {doc.filename}
          </p>
          <div className="flex items-center gap-1.5 mt-0.5">
            <StatusIcon status={doc.status} />
            <span className="text-xs text-gray-400">
              {doc.page_count} page{doc.page_count !== 1 ? 's' : ''}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={(e) => { e.stopPropagation(); setShowInfo((v) => !v) }}
            className="p-1 text-gray-400 hover:text-brand-500 transition-colors"
          >
            {showInfo ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onDelete() }}
            className="p-1 text-gray-400 hover:text-red-500 transition-colors"
          >
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      {showInfo && (
        <div className="mt-2 pt-2 border-t border-gray-100 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400 space-y-0.5">
          <p>ID: <span className="font-mono">{doc.document_id.slice(0, 8)}…</span></p>
          <p>Status: <span className="capitalize">{doc.status}</span></p>
        </div>
      )}
    </div>
  )
}

export default function DocumentUpload() {
  const { state, dispatch } = useApp()
  const { showSuccess, showError, showInfo } = useToast()
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({})
  const [ocrEngine, setOcrEngine] = useState<'auto' | 'easyocr' | 'tesseract'>('auto')
  const [enhanceImages, setEnhanceImages] = useState(true)

  const onDrop = useCallback(async (accepted: File[]) => {
    if (!accepted.length) return
    setUploading(true)

    for (const file of accepted) {
      const fileId = `${file.name}-${Date.now()}`
      setUploadProgress(prev => ({ ...prev, [fileId]: 0 }))
      
      try {
        // Simulate progress for better UX
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => ({
            ...prev,
            [fileId]: Math.min(prev[fileId] + 10, 90)
          }))
        }, 200)

        const result = await documentsApi.upload(file, {
          ocr_engine: ocrEngine,
          enhance_images: enhanceImages,
        })
        
        clearInterval(progressInterval)
        setUploadProgress(prev => ({ ...prev, [fileId]: 100 }))
        
        const summary: DocumentSummary = {
          document_id: result.document_id,
          filename: result.filename,
          page_count: 0,
          status: result.status,
          created_at: new Date().toISOString(),
        }
        dispatch({ type: 'ADD_DOCUMENT', payload: summary })
        showSuccess('Upload successful', `${file.name} has been uploaded and is being processed.`)
        
        setTimeout(() => {
          setUploadProgress(prev => {
            const { [fileId]: _, ...rest } = prev
            return rest
          })
        }, 1000)
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Upload failed'
        showError('Upload failed', `Failed to upload ${file.name}: ${msg}`)
        setUploadProgress(prev => {
          const { [fileId]: _, ...rest } = prev
          return rest
        })
      }
    }
    setUploading(false)
  }, [ocrEngine, enhanceImages, dispatch, showSuccess, showError])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    disabled: uploading,
    maxSize: 50 * 1024 * 1024,
  })

  const handleDelete = async (docId: string) => {
    try {
      await documentsApi.delete(docId)
      dispatch({ type: 'REMOVE_DOCUMENT', payload: docId })
      showSuccess('Document deleted', 'Document has been successfully removed.')
    } catch {
      showError('Failed to delete document', 'Please try again.')
    }
  }

  return (
    <div className="flex flex-col h-full p-3 gap-3">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={clsx(
          'rounded-xl border-2 border-dashed p-6 text-center cursor-pointer transition-all min-h-[120px] flex items-center justify-center',
          isDragActive
            ? 'border-brand-400 bg-brand-50 dark:bg-brand-900/20 scale-[1.02]'
            : 'border-gray-300 dark:border-gray-600 hover:border-brand-400 hover:bg-gray-50 dark:hover:bg-gray-800/50',
          uploading && 'opacity-50 cursor-not-allowed'
        )}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-3">
          {uploading ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 size={32} className="text-brand-500 animate-spin" />
              <span className="text-sm font-medium text-brand-600 dark:text-brand-400">
                Processing uploads...
              </span>
            </div>
          ) : isDragActive ? (
            <div className="flex flex-col items-center gap-2">
              <CloudUpload size={32} className="text-brand-500" />
              <span className="text-sm font-medium text-brand-600 dark:text-brand-400">
                Drop your files here
              </span>
            </div>
          ) : (
            <>
              <Upload size={32} className="text-gray-400" />
              <div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Drag and drop your medical documents
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  or <span className="text-brand-500 font-medium">click to browse</span>
                </p>
                <p className="text-xs text-gray-400 mt-2">
                  PDF, DOCX, JPEG, PNG, TIFF, BMP, WEBP · Max 50MB
                </p>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Upload Progress */}
      {Object.keys(uploadProgress).length > 0 && (
        <div className="space-y-2">
          {Object.entries(uploadProgress).map(([fileId, progress]) => (
            <div key={fileId} className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3">
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="font-medium text-gray-700 dark:text-gray-300 truncate">
                  {fileId.split('-')[0]}
                </span>
                <span className="text-gray-500 dark:text-gray-400">
                  {Math.round(progress)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-brand-500 h-2 rounded-full transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* OCR Options */}
      <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-2.5 space-y-2">
        <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          OCR Settings
        </p>
        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-600 dark:text-gray-300">Engine</label>
          <select
            value={ocrEngine}
            onChange={(e) => setOcrEngine(e.target.value as typeof ocrEngine)}
            className="text-xs bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded px-2 py-1 text-gray-700 dark:text-gray-200"
          >
            <option value="auto">Auto (recommended)</option>
            <option value="easyocr">EasyOCR (low-quality scans)</option>
            <option value="tesseract">Tesseract (clean docs)</option>
          </select>
        </div>
        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-600 dark:text-gray-300">Enhance images</label>
          <button
            onClick={() => setEnhanceImages((v) => !v)}
            className={clsx(
              'relative w-9 h-5 rounded-full transition-colors',
              enhanceImages ? 'bg-brand-500' : 'bg-gray-300 dark:bg-gray-600'
            )}
          >
            <span
              className={clsx(
                'absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform',
                enhanceImages && 'translate-x-4'
              )}
            />
          </button>
        </div>
      </div>

      {/* Document list */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {state.documents.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 py-8 text-center">
            <FileText size={32} className="text-gray-300 dark:text-gray-600" />
            <p className="text-xs text-gray-400">No documents yet</p>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between">
              <p className="text-xs text-gray-500">
                {state.documents.length} document{state.documents.length !== 1 ? 's' : ''}
              </p>
              {state.selectedDocumentIds.length > 0 && (
                <p className="text-xs text-brand-500 font-medium">
                  {state.selectedDocumentIds.length} selected for chat
                </p>
              )}
            </div>
            {state.documents.map((doc) => (
              <DocCard
                key={doc.document_id}
                doc={doc}
                selected={state.selectedDocumentIds.includes(doc.document_id)}
                onToggle={() => dispatch({ type: 'TOGGLE_DOCUMENT_SELECTION', payload: doc.document_id })}
                onDelete={() => handleDelete(doc.document_id)}
              />
            ))}
          </>
        )}
      </div>
    </div>
  )
}
