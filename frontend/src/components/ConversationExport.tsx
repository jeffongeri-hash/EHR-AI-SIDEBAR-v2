import React, { useState } from 'react'
import { Download, Search, Calendar, MessageSquare, X, Filter } from 'lucide-react'
import clsx from 'clsx'
import { useToast } from '@/contexts/ToastContext'
import { useApp } from '@/contexts/AppContext'
import type { UIMessage } from '@/types'

interface ConversationExportProps {
  messages: UIMessage[]
  onClose: () => void
}

function ConversationExport({ messages, onClose }: ConversationExportProps) {
  const { showSuccess, showError } = useToast()
  const [format, setFormat] = useState<'json' | 'markdown' | 'txt'>('markdown')
  const [includeSystem, setIncludeSystem] = useState(false)

  const exportConversation = () => {
    try {
      let content = ''
      const filteredMessages = includeSystem 
        ? messages 
        : messages.filter(m => m.role !== 'system')

      if (format === 'json') {
        content = JSON.stringify(filteredMessages, null, 2)
      } else if (format === 'markdown') {
        content = generateMarkdown(filteredMessages)
      } else {
        content = generatePlainText(filteredMessages)
      }

      const blob = new Blob([content], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `conversation-${new Date().toISOString().split('T')[0]}.${format}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      showSuccess('Export successful', `Conversation exported as ${format.toUpperCase()}`)
      onClose()
    } catch (error) {
      showError('Export failed', 'Could not export conversation. Please try again.')
    }
  }

  const generateMarkdown = (msgs: UIMessage[]): string => {
    let md = `# EHR AI Conversation\n\n`
    md += `**Date:** ${new Date().toLocaleDateString()}\n\n`
    
    msgs.forEach((msg, index) => {
      const roleEmoji = msg.role === 'user' ? '👤' : msg.role === 'assistant' ? '🤖' : '⚙️'
      const roleName = msg.role.charAt(0).toUpperCase() + msg.role.slice(1)
      
      md += `## ${roleEmoji} ${roleName} ${index + 1}\n\n`
      md += `${msg.content}\n\n`
      
      if (msg.document_ids?.length) {
        md += `*Referenced documents: ${msg.document_ids.length}*\n\n`
      }
      
      md += `---\n\n`
    })
    
    return md
  }

  const generatePlainText = (msgs: UIMessage[]): string => {
    let text = `EHR AI Conversation - ${new Date().toLocaleDateString()}\n`
    text += `${'='.repeat(50)}\n\n`
    
    msgs.forEach((msg, index) => {
      const rolePrefix = msg.role === 'user' ? 'USER' : msg.role === 'assistant' ? 'ASSISTANT' : 'SYSTEM'
      text += `[${rolePrefix} ${index + 1}]\n`
      text += `${msg.content}\n`
      
      if (msg.document_ids?.length) {
        text += `(Referenced documents: ${msg.document_ids.length})\n`
      }
      
      text += `\n${'-'.repeat(30)}\n\n`
    })
    
    return text
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-md w-full mx-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Export Conversation
          </h3>
          <button
            onClick={onClose}
            className="p-1 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          >
            <X size={20} />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300 block mb-2">
              Export Format
            </label>
            <div className="grid grid-cols-3 gap-2">
              {(['markdown', 'json', 'txt'] as const).map((fmt) => (
                <button
                  key={fmt}
                  onClick={() => setFormat(fmt)}
                  className={clsx(
                    'px-3 py-2 text-sm rounded-lg border transition-colors',
                    format === fmt
                      ? 'bg-brand-50 dark:bg-brand-900/20 border-brand-500 text-brand-700 dark:text-brand-300'
                      : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:border-brand-400'
                  )}
                >
                  {fmt.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="include-system"
              checked={includeSystem}
              onChange={(e) => setIncludeSystem(e.target.checked)}
              className="rounded border-gray-300 text-brand-600 focus:ring-brand-500"
            />
            <label htmlFor="include-system" className="text-sm text-gray-700 dark:text-gray-300">
              Include system messages
            </label>
          </div>

          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
              <MessageSquare size={16} />
              <span>
                {includeSystem ? messages.length : messages.filter(m => m.role !== 'system').length} messages to export
              </span>
            </div>
          </div>
        </div>

        <div className="flex gap-2 mt-6">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={exportConversation}
            className="flex-1 px-4 py-2 text-sm bg-brand-500 text-white rounded-lg hover:bg-brand-600 transition-colors flex items-center justify-center gap-2"
          >
            <Download size={16} />
            Export
          </button>
        </div>
      </div>
    </div>
  )
}

export default ConversationExport