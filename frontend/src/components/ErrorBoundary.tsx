/**
 * ErrorBoundary
 * =============
 * Catches unhandled React render errors so a single panel crash cannot
 * blank out the entire application.
 *
 * Usage:
 *   <ErrorBoundary label="Chat">
 *     <ChatInterface />
 *   </ErrorBoundary>
 *
 * When an error is caught the panel shows a minimal recovery UI rather
 * than a white screen.  The full error + stack are logged to the console
 * for developer visibility.
 */

import React, { Component, ErrorInfo, ReactNode } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

interface Props {
  children: ReactNode
  /** Human-readable name shown in the fallback UI ("Chat", "Documents", …) */
  label?: string
}

interface State {
  hasError: boolean
  error: Error | null
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Surface full details in the browser console for developers
    console.error(`[ErrorBoundary: ${this.props.label ?? 'unknown'}] Uncaught error:`, error)
    console.error('Component stack:', info.componentStack)
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (!this.state.hasError) {
      return this.props.children
    }

    const { label = 'Panel' } = this.props
    const message = this.state.error?.message ?? 'An unexpected error occurred.'

    return (
      <div className="flex flex-col items-center justify-center h-full p-6 text-center gap-4">
        <div className="flex items-center justify-center w-12 h-12 rounded-full bg-red-100 dark:bg-red-900/30">
          <AlertTriangle className="w-6 h-6 text-red-500 dark:text-red-400" />
        </div>

        <div>
          <p className="text-sm font-semibold text-gray-800 dark:text-gray-100">
            {label} encountered an error
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 max-w-xs break-words">
            {message}
          </p>
        </div>

        <button
          onClick={this.handleReset}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg
                     bg-blue-600 hover:bg-blue-700 text-white transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Try again
        </button>
      </div>
    )
  }
}
