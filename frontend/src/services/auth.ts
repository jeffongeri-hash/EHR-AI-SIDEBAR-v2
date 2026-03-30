/**
 * Auth Token Store
 * ================
 * Stores the JWT access token in MODULE MEMORY rather than localStorage.
 *
 * Why this matters:
 *   localStorage is accessible to ANY JavaScript running on the page.
 *   A single XSS vulnerability in any third-party script can silently
 *   read every key in localStorage and exfiltrate it to an attacker.
 *
 *   Memory-stored tokens are NOT accessible outside this module.
 *   The worst an XSS attacker can do is call the exported functions,
 *   which the same attacker could do anyway.
 *
 * Trade-offs:
 *   - Token is lost on page refresh (users must log in once per session)
 *   - This is actually CORRECT HIPAA behaviour — HIPAA requires session
 *     timeout / re-authentication for workstations with PHI access
 *
 * Future improvement:
 *   Add httpOnly, Secure, SameSite=Strict refresh-token cookie managed
 *   by the backend so the session can survive a page refresh without
 *   storing anything sensitive in JavaScript-accessible storage.
 */

const BASE_URL = import.meta.env.VITE_API_URL || '/api'

// ── In-memory token store ─────────────────────────────────────────────────────
// This is a module-level variable — NOT exposed to window, NOT in localStorage.

let _accessToken: string | null = null
let _currentUser: AuthUser | null = null

export interface AuthUser {
  user_id: string
  username: string
  role: string
  expires_in: number
}

export interface LoginCredentials {
  username: string
  password: string
}

export interface LoginResponse {
  access_token: string
  token_type: string
  expires_in: number
  user_id: string
  username: string
  role: string
}

// ── Token accessors ───────────────────────────────────────────────────────────

export function getToken(): string | null {
  return _accessToken
}

export function getCurrentUser(): AuthUser | null {
  return _currentUser
}

export function isAuthenticated(): boolean {
  return _accessToken !== null
}

export function clearAuth(): void {
  _accessToken = null
  _currentUser = null
}

// ── Auth API ──────────────────────────────────────────────────────────────────

export async function login(credentials: LoginCredentials): Promise<AuthUser> {
  const res = await fetch(`${BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail ?? `Login failed (${res.status})`)
  }

  const data: LoginResponse = await res.json()

  // Store in memory only — never in localStorage or sessionStorage
  _accessToken = data.access_token
  _currentUser = {
    user_id: data.user_id,
    username: data.username,
    role: data.role,
    expires_in: data.expires_in,
  }

  return _currentUser
}

export async function register(body: {
  username: string
  email: string
  password: string
  role?: string
}): Promise<{ user_id: string; username: string; role: string }> {
  const res = await fetch(`${BASE_URL}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail ?? `Registration failed (${res.status})`)
  }

  return res.json()
}

export async function fetchCurrentUser(): Promise<AuthUser | null> {
  if (!_accessToken) return null
  const res = await fetch(`${BASE_URL}/auth/me`, {
    headers: { Authorization: `Bearer ${_accessToken}` },
  })
  if (!res.ok) {
    clearAuth()
    return null
  }
  const data = await res.json()
  _currentUser = { ...data, expires_in: 0 }
  return _currentUser
}
