import React, { useState } from 'react'
import { Brain, Trash2, Download, Search } from 'lucide-react'
import MessageList from './MessageList'
import ChatInput from './ChatInput'
import ModelSelector from './ModelSelector'
import ConversationSearch from './ConversationSearch'
import ConversationExport from './ConversationExport'
import { useChat } from '@/hooks/useChat'
import { useApp } from '@/contexts/AppContext'

const QUICK_QUESTIONS = [
  'What are the current medications?',
  'Does the patient have any drug allergies?',
  'What is the most recent HbA1c?',
  'When was the last annual exam?',
  'Summarise the key diagnoses',
  'Extract all lab results',
]

export default function ChatInterface() {
  const { state, dispatch } = useApp()
  const { sendMessage, stopGeneration, clearChat } = useChat()
  const [showSearch, setShowSearch] = useState(false)
  const [showExport, setShowExport] = useState(false)

  const isEmpty = state.messages.length === 0

  const handleMessageSelect = (messageId: string) => {
    // Scroll to message or highlight it
    const element = document.getElementById(`message-${messageId}`)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' })
      element.classList.add('highlight-message')
      setTimeout(() => element.classList.remove('highlight-message'), 2000)
    }
  }

  return (
    <div className="flex flex-col h-full bg-gray-950">
      {/* Header */}
      <div className="flex-shrink-0 flex items-center justify-between px-4 py-3.5 bg-gray-900 border-b border-gray-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center">
            <Brain size={16} className="text-blue-400" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-gray-100 leading-tight">
              Medi-Sync Sidebar
            </h1>
            <p className="text-xs text-gray-500 leading-tight">
              {state.selectedDocumentIds.length > 0
                ? `${state.selectedDocumentIds.length} document${state.selectedDocumentIds.length > 1 ? 's' : ''} attached`
                : 'Ask questions about this document'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <ModelSelector />
          {!isEmpty && (
            <>
              <button
                onClick={() => setShowSearch(!showSearch)}
                title="Search conversation"
                className="p-1.5 text-gray-500 hover:text-brand-400 rounded-lg hover:bg-brand-500/10 transition-colors"
              >
                <Search size={14} />
              </button>
              <button
                onClick={() => setShowExport(true)}
                title="Export conversation"
                className="p-1.5 text-gray-500 hover:text-green-400 rounded-lg hover:bg-green-500/10 transition-colors"
              >
                <Download size={14} />
              </button>
              <button
                onClick={clearChat}
                title="Clear conversation"
                className="p-1.5 text-gray-500 hover:text-red-400 rounded-lg hover:bg-red-500/10 transition-colors"
              >
                <Trash2 size={14} />
              </button>
            </>
          )}
          <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/30">
            AI Powered
          </span>
        </div>
      </div>

      {/* Search Panel */}
      {showSearch && !isEmpty && (
        <ConversationSearch 
          messages={state.messages}
          onMessageSelect={handleMessageSelect}
        />
      )}

      {/* Quick Questions — shown only when no messages */}
      {isEmpty && (
        <div className="flex-shrink-0 px-4 pt-4 pb-1">
          <p className="text-xs font-medium text-gray-500 mb-2.5">Quick Questions:</p>
          <div className="flex flex-wrap gap-2">
            {QUICK_QUESTIONS.map((q) => (
              <button
                key={q}
                onClick={() => sendMessage(q)}
                className="text-xs px-3 py-1.5 rounded-full border border-gray-700 text-gray-300 bg-gray-800/60 hover:bg-gray-700 hover:border-gray-600 hover:text-gray-100 transition-all"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <MessageList messages={state.messages} />

      {/* Input */}
      <ChatInput
        onSend={sendMessage}
        onStop={stopGeneration}
        onAttach={() => dispatch({ type: 'SET_TAB', payload: 'documents' })}
        isLoading={state.isLoading}
        selectedDocumentCount={state.selectedDocumentIds.length}
      />

      {/* Export Modal */}
      {showExport && (
        <ConversationExport 
          messages={state.messages}
          onClose={() => setShowExport(false)}
        />
      )}
    </div>
  )
}
