import React, { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Bot, User, AlertCircle, Copy, Check } from 'lucide-react'
import { useState } from 'react'
import { format } from 'date-fns'
import type { UIMessage } from '@/types'
import clsx from 'clsx'

interface MessageListProps {
  messages: UIMessage[]
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <button
      onClick={copy}
      className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
      title="Copy message"
    >
      {copied ? <Check size={14} className="text-green-500" /> : <Copy size={14} className="text-gray-400" />}
    </button>
  )
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 p-2">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-2 h-2 bg-brand-400 rounded-full animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  )
}

function MessageBubble({ message }: { message: UIMessage }) {
  const isUser = message.role === 'user'
  const isSystem = message.role === 'system'

  if (isSystem) {
    return (
      <div className="flex justify-center my-2">
        <span className="text-xs text-gray-500 bg-gray-800 rounded-full px-3 py-1">
          {message.content}
        </span>
      </div>
    )
  }

  return (
    <div
      id={`message-${message.id}`}
      className={clsx(
        'flex gap-3 group animate-fade-in',
        isUser ? 'flex-row-reverse' : 'flex-row'
      )}
    >
      {/* Avatar */}
      <div
        className={clsx(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium',
          isUser ? 'bg-blue-600' : 'bg-gray-700'
        )}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      {/* Content */}
      <div className={clsx('flex-1 min-w-0', isUser ? 'items-end' : 'items-start', 'flex flex-col gap-1')}>
        <div
          className={clsx(
            'rounded-2xl px-4 py-3 max-w-[85%] relative',
            isUser
              ? 'bg-blue-600 text-white rounded-tr-sm ml-auto'
              : 'bg-gray-800 border border-gray-700 rounded-tl-sm'
          )}
        >
          {message.isStreaming && !message.content ? (
            <TypingIndicator />
          ) : message.error ? (
            <div className="flex items-center gap-2 text-red-500">
              <AlertCircle size={16} />
              <span className="text-sm">{message.content}</span>
            </div>
          ) : (
            <div
              className={clsx(
                'prose prose-sm max-w-none',
                isUser
                  ? 'prose-invert'
                  : 'dark:prose-invert prose-gray'
              )}
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ node, className, children, ...props }) {
                    const isInline = !className
                    return isInline ? (
                      <code
                        className={clsx(
                          'px-1 py-0.5 rounded text-xs font-mono',
                          isUser
                            ? 'bg-brand-600 text-brand-100'
                            : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'
                        )}
                        {...props}
                      >
                        {children}
                      </code>
                    ) : (
                      <code className={clsx('text-xs font-mono', className)} {...props}>
                        {children}
                      </code>
                    )
                  },
                  pre({ children }) {
                    return (
                      <pre className="bg-gray-900 text-gray-100 rounded-lg p-3 overflow-x-auto text-xs">
                        {children}
                      </pre>
                    )
                  },
                  table({ children }) {
                    return (
                      <div className="overflow-x-auto">
                        <table className="min-w-full border-collapse text-xs">{children}</table>
                      </div>
                    )
                  },
                  th({ children }) {
                    return (
                      <th className="border border-gray-300 dark:border-gray-600 px-2 py-1 bg-gray-50 dark:bg-gray-700 font-semibold text-left">
                        {children}
                      </th>
                    )
                  },
                  td({ children }) {
                    return (
                      <td className="border border-gray-300 dark:border-gray-600 px-2 py-1">
                        {children}
                      </td>
                    )
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
              {message.isStreaming && (
                <span className="inline-block w-0.5 h-4 bg-current animate-pulse ml-0.5 align-text-bottom" />
              )}
            </div>
          )}
        </div>

        {/* Meta */}
        <div className={clsx('flex items-center gap-2', isUser ? 'flex-row-reverse' : 'flex-row')}>
          <span className="text-xs text-gray-400">
            {format(new Date(message.timestamp), 'HH:mm')}
          </span>
          {!isUser && <CopyButton text={message.content} />}
          {message.document_ids && message.document_ids.length > 0 && (
            <span className="text-xs text-brand-400">
              📎 {message.document_ids.length} doc{message.document_ids.length > 1 ? 's' : ''}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

export default function MessageList({ messages }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center p-8">
        <div className="w-14 h-14 rounded-2xl bg-gray-800 border border-gray-700 flex items-center justify-center">
          <Bot size={28} className="text-gray-500" />
        </div>
        <div>
          <p className="text-sm font-medium text-gray-400">
            Ask a question about the faxed document
          </p>
          <p className="text-xs text-gray-600 mt-1">or click a quick question above</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-950">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
