/**
 * LoginModal
 * ==========
 * Full-screen modal shown when the user is not authenticated.
 * Calls POST /api/auth/login and stores the JWT in memory (not localStorage).
 */

import React, { useState } from 'react'
import { ShieldCheck, Eye, EyeOff, Loader2 } from 'lucide-react'
import { login } from '@/services/auth'
import type { AuthUser } from '@/services/auth'

interface Props {
  onLogin: (user: AuthUser) => void
}

export default function LoginModal({ onLogin }: Props) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const user = await login({ username: username.trim(), password })
      onLogin(user)
    } catch (err: any) {
      setError(err.message ?? 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-gray-950/80 backdrop-blur-sm">
      <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-xl w-full max-w-sm mx-4 p-7">
        {/* Logo / title */}
        <div className="flex flex-col items-center mb-6">
          <div className="flex items-center justify-center w-12 h-12 rounded-full bg-blue-100 dark:bg-blue-900/30 mb-3">
            <ShieldCheck className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          </div>
          <h1 className="text-base font-semibold text-gray-900 dark:text-gray-100">EHR AI Sidebar</h1>
          <p className="text-xs text-gray-400 mt-1">Sign in to access protected health information</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-xs font-medium text-gray-700 dark:text-gray-300 block mb-1">
              Username
            </label>
            <input
              type="text"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="your.username"
              required
              className="w-full text-sm bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700
                         rounded-lg px-3 py-2 text-gray-700 dark:text-gray-200 placeholder-gray-400
                         focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
          </div>

          <div>
            <label className="text-xs font-medium text-gray-700 dark:text-gray-300 block mb-1">
              Password
            </label>
            <div className="relative">
              <input
                type={showPw ? 'text' : 'password'}
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
                className="w-full text-sm bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700
                           rounded-lg px-3 py-2 pr-9 text-gray-700 dark:text-gray-200 placeholder-gray-400
                           focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
              <button
                type="button"
                onClick={() => setShowPw((v) => !v)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                {showPw ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
          </div>

          {error && (
            <p className="text-xs text-red-500 bg-red-50 dark:bg-red-900/20 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading || !username.trim() || !password}
            className="flex items-center justify-center gap-2 w-full bg-blue-600 hover:bg-blue-700
                       disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl
                       px-4 py-2.5 text-sm font-medium transition-colors"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        <p className="text-xs text-center text-gray-400 mt-5">
          All PHI access is recorded in the HIPAA audit log.
        </p>
      </div>
    </div>
  )
}
