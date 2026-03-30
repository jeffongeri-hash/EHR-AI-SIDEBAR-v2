import React, { useState, useRef, KeyboardEvent } from 'react'
import { Send, Square, Paperclip } from 'lucide-react'
import clsx from 'clsx'

interface ChatInputProps {
  onSend: (text: string) => void
  onStop: () => void
  onAttach?: () => void
  isLoading: boolean
  disabled?: boolean
  selectedDocumentCount?: number
}

export default function ChatInput({
  onSend,
  onStop,
  onAttach,
  isLoading,
  disabled = false,
  selectedDocumentCount = 0,
}: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || isLoading) return
    onSend(trimmed)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleInput = () => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`
  }

  return (
    <div className="border-t border-gray-800 bg-gray-900 p-3">
      {selectedDocumentCount > 0 && (
        <div className="flex items-center gap-1 mb-2 text-xs text-blue-400">
          <Paperclip size={12} />
          <span>{selectedDocumentCount} document{selectedDocumentCount > 1 ? 's' : ''} attached</span>
        </div>
      )}
      <div
        className={clsx(
          'flex items-end gap-2 rounded-2xl border bg-gray-800 px-3 py-2',
          'border-gray-700',
          'focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500/20',
          'transition-all duration-150'
        )}
      >
        {onAttach && (
          <button
            onClick={onAttach}
            className="flex-shrink-0 p-1 text-gray-400 hover:text-brand-500 transition-colors mb-1"
            title="Attach document"
            disabled={disabled}
          >
            <Paperclip size={18} />
          </button>
        )}

        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => { setValue(e.target.value); handleInput() }}
          onKeyDown={handleKeyDown}
          placeholder="Ask about patient records, medications, lab results…"
          rows={1}
          disabled={disabled}
          className={clsx(
            'flex-1 resize-none bg-transparent outline-none text-sm',
            'text-gray-100 placeholder-gray-500',
            'min-h-[24px] max-h-[200px] py-0.5',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        />

        <button
          onClick={isLoading ? onStop : handleSend}
          disabled={!isLoading && (!value.trim() || disabled)}
          className={clsx(
            'flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center transition-all duration-150 mb-0.5',
            isLoading
              ? 'bg-red-500 hover:bg-red-600 text-white'
              : value.trim() && !disabled
                ? 'bg-brand-500 hover:bg-brand-600 text-white shadow-sm'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
          )}
          title={isLoading ? 'Stop generation' : 'Send (Enter)'}
        >
          {isLoading ? <Square size={14} /> : <Send size={14} />}
        </button>
      </div>
      <p className="text-xs text-gray-600 text-center mt-1">
        Press Enter to send · Shift+Enter for new line
      </p>
    </div>
  )
}
