import React, { useState, useMemo } from 'react'
import { Search, X, Calendar, User, Bot, Filter, SortDesc, SortAsc } from 'lucide-react'
import clsx from 'clsx'
import type { UIMessage } from '@/types'

interface ConversationSearchProps {
  messages: UIMessage[]
  onMessageSelect?: (messageId: string) => void
  className?: string
}

interface SearchFilters {
  role: 'all' | 'user' | 'assistant'
  sortBy: 'date' | 'relevance'
  sortOrder: 'asc' | 'desc'
}

export default function ConversationSearch({ 
  messages, 
  onMessageSelect,
  className 
}: ConversationSearchProps) {
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState<SearchFilters>({
    role: 'all',
    sortBy: 'date',
    sortOrder: 'desc'
  })
  const [isExpanded, setIsExpanded] = useState(false)

  const filteredAndSortedMessages = useMemo(() => {
    let filtered = messages

    // Filter by role
    if (filters.role !== 'all') {
      filtered = filtered.filter(msg => msg.role === filters.role)
    }

    // Search query
    if (query.trim()) {
      const lowercaseQuery = query.toLowerCase()
      filtered = filtered.filter(msg => 
        msg.content.toLowerCase().includes(lowercaseQuery)
      )
    }

    // Sort
    const sorted = [...filtered].sort((a, b) => {
      if (filters.sortBy === 'date') {
        const dateA = new Date(a.timestamp).getTime()
        const dateB = new Date(b.timestamp).getTime()
        return filters.sortOrder === 'desc' ? dateB - dateA : dateA - dateB
      } else {
        // Sort by relevance (query match position)
        if (!query.trim()) return 0
        const posA = a.content.toLowerCase().indexOf(query.toLowerCase())
        const posB = b.content.toLowerCase().indexOf(query.toLowerCase())
        return filters.sortOrder === 'desc' ? posA - posB : posB - posA
      }
    })

    return sorted
  }, [messages, query, filters])

  const highlightText = (text: string, highlight: string) => {
    if (!highlight.trim()) return text
    
    const parts = text.split(new RegExp(`(${highlight})`, 'gi'))
    return parts.map((part, i) => 
      part.toLowerCase() === highlight.toLowerCase() ? (
        <mark key={i} className="bg-yellow-200 dark:bg-yellow-800 text-yellow-900 dark:text-yellow-200 rounded px-1">
          {part}
        </mark>
      ) : part
    )
  }

  const clearSearch = () => {
    setQuery('')
    setFilters({
      role: 'all',
      sortBy: 'date',
      sortOrder: 'desc'
    })
  }

  return (
    <div className={clsx('bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700', className)}>
      {/* Search Bar */}
      <div className="p-3">
        <div className="relative">
          <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search conversation..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full pl-9 pr-10 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-brand-500 focus:border-brand-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
          />
          {(query || filters.role !== 'all') && (
            <button
              onClick={clearSearch}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <X size={14} />
            </button>
          )}
        </div>

        {/* Filters Toggle */}
        <div className="flex items-center justify-between mt-2">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
          >
            <Filter size={12} />
            <span>Filters</span>
            {isExpanded && <span className="text-brand-500">({Object.values(filters).filter(v => v !== 'all' && v !== 'date' && v !== 'desc').length})</span>}
          </button>
          
          {filteredAndSortedMessages.length > 0 && (
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {filteredAndSortedMessages.length} result{filteredAndSortedMessages.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>
      </div>

      {/* Expanded Filters */}
      {isExpanded && (
        <div className="px-3 pb-3 border-t border-gray-100 dark:border-gray-700 pt-3">
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Message Type
              </label>
              <select
                value={filters.role}
                onChange={(e) => setFilters(prev => ({ ...prev, role: e.target.value as any }))}
                className="w-full text-xs border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              >
                <option value="all">All messages</option>
                <option value="user">User messages</option>
                <option value="assistant">AI responses</option>
              </select>
            </div>
            
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Sort by
              </label>
              <div className="flex gap-1">
                <select
                  value={filters.sortBy}
                  onChange={(e) => setFilters(prev => ({ ...prev, sortBy: e.target.value as any }))}
                  className="flex-1 text-xs border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                >
                  <option value="date">Date</option>
                  <option value="relevance">Relevance</option>
                </select>
                <button
                  onClick={() => setFilters(prev => ({ ...prev, sortOrder: prev.sortOrder === 'desc' ? 'asc' : 'desc' }))}
                  className="px-2 py-1 border border-gray-300 dark:border-gray-600 rounded text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-600"
                  title={`Sort ${filters.sortOrder === 'desc' ? 'ascending' : 'descending'}`}
                >
                  {filters.sortOrder === 'desc' ? <SortDesc size={12} /> : <SortAsc size={12} />}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {query && (
        <div className="max-h-64 overflow-y-auto border-t border-gray-100 dark:border-gray-700">
          {filteredAndSortedMessages.length === 0 ? (
            <div className="p-4 text-center text-sm text-gray-500 dark:text-gray-400">
              No messages found matching "{query}"
            </div>
          ) : (
            <div className="space-y-1 p-2">
              {filteredAndSortedMessages.map((message) => (
                <button
                  key={message.id}
                  onClick={() => onMessageSelect?.(message.id)}
                  className="w-full text-left p-2 rounded hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  <div className="flex items-start gap-2">
                    <div className="flex-shrink-0 mt-1">
                      {message.role === 'user' ? (
                        <User size={14} className="text-blue-500" />
                      ) : (
                        <Bot size={14} className="text-gray-500" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                        {new Date(message.timestamp).toLocaleString()}
                      </div>
                      <div className="text-sm text-gray-900 dark:text-gray-100 line-clamp-2">
                        {highlightText(
                          message.content.length > 100 
                            ? message.content.substring(0, 100) + '...' 
                            : message.content,
                          query
                        )}
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}