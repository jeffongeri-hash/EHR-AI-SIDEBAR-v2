import React, { createContext, useContext, useReducer, ReactNode } from 'react'
import type { DocumentSummary, ModelInfo, Tab, UIMessage } from '@/types'

interface AppState {
  activeTab: Tab
  conversationId: string | null
  messages: UIMessage[]
  selectedDocumentIds: string[]
  documents: DocumentSummary[]
  availableModels: ModelInfo[]
  selectedProvider: string
  selectedModel: string
  temperature: number
  isLoading: boolean
  isSidebarOpen: boolean
}

type Action =
  | { type: 'SET_TAB'; payload: Tab }
  | { type: 'SET_CONVERSATION_ID'; payload: string }
  | { type: 'ADD_MESSAGE'; payload: UIMessage }
  | { type: 'UPDATE_MESSAGE'; payload: { id: string; updates: Partial<UIMessage> } }
  | { type: 'CLEAR_MESSAGES' }
  | { type: 'SET_DOCUMENTS'; payload: DocumentSummary[] }
  | { type: 'ADD_DOCUMENT'; payload: DocumentSummary }
  | { type: 'REMOVE_DOCUMENT'; payload: string }
  | { type: 'TOGGLE_DOCUMENT_SELECTION'; payload: string }
  | { type: 'SET_MODELS'; payload: ModelInfo[] }
  | { type: 'SET_PROVIDER'; payload: string }
  | { type: 'SET_MODEL'; payload: string }
  | { type: 'SET_TEMPERATURE'; payload: number }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'TOGGLE_SIDEBAR' }

const initialState: AppState = {
  activeTab: 'home',
  conversationId: null,
  messages: [],
  selectedDocumentIds: [],
  documents: [],
  availableModels: [],
  selectedProvider: 'ollama',
  selectedModel: 'llama3.2',
  temperature: 0.7,
  isLoading: false,
  isSidebarOpen: true,
}

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_TAB':
      return { ...state, activeTab: action.payload }
    case 'SET_CONVERSATION_ID':
      return { ...state, conversationId: action.payload }
    case 'ADD_MESSAGE':
      return { ...state, messages: [...state.messages, action.payload] }
    case 'UPDATE_MESSAGE':
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.payload.id ? { ...m, ...action.payload.updates } : m
        ),
      }
    case 'CLEAR_MESSAGES':
      return { ...state, messages: [], conversationId: null }
    case 'SET_DOCUMENTS':
      return { ...state, documents: action.payload }
    case 'ADD_DOCUMENT':
      return { ...state, documents: [action.payload, ...state.documents] }
    case 'REMOVE_DOCUMENT':
      return {
        ...state,
        documents: state.documents.filter((d) => d.document_id !== action.payload),
        selectedDocumentIds: state.selectedDocumentIds.filter((id) => id !== action.payload),
      }
    case 'TOGGLE_DOCUMENT_SELECTION': {
      const id = action.payload
      const selected = state.selectedDocumentIds.includes(id)
        ? state.selectedDocumentIds.filter((d) => d !== id)
        : [...state.selectedDocumentIds, id]
      return { ...state, selectedDocumentIds: selected }
    }
    case 'SET_MODELS':
      return { ...state, availableModels: action.payload }
    case 'SET_PROVIDER':
      return { ...state, selectedProvider: action.payload }
    case 'SET_MODEL':
      return { ...state, selectedModel: action.payload }
    case 'SET_TEMPERATURE':
      return { ...state, temperature: action.payload }
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload }
    case 'TOGGLE_SIDEBAR':
      return { ...state, isSidebarOpen: !state.isSidebarOpen }
    default:
      return state
  }
}

interface AppContextValue {
  state: AppState
  dispatch: React.Dispatch<Action>
}

const AppContext = createContext<AppContextValue | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState)
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}
