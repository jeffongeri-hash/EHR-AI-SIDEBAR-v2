import { useCallback, useRef } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { useToast } from '@/contexts/ToastContext'
import { chatApi } from '@/services/api'
import { useApp } from '@/contexts/AppContext'
import type { UIMessage } from '@/types'

export function useChat() {
  const { state, dispatch } = useApp()
  const { showError, showWarning } = useToast()
  const abortRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim()) {
        showWarning('Empty message', 'Please enter a message before sending.')
        return
      }
      
      if (state.isLoading) {
        showWarning('Please wait', 'A message is currently being processed.')
        return
      }

      // Check if provider is available
      if (state.selectedProvider === 'ollama') {
        showWarning('Ollama not available', 'Please install and start Ollama, or switch to Claude in model settings.')
        return
      }

      // Add user message
      const userMsg: UIMessage = {
        id: uuidv4(),
        role: 'user',
        content: text,
        timestamp: new Date().toISOString(),
        document_ids: state.selectedDocumentIds,
      }
      dispatch({ type: 'ADD_MESSAGE', payload: userMsg })
      dispatch({ type: 'SET_LOADING', payload: true })

      // Placeholder for assistant streaming message
      const assistantId = uuidv4()
      const assistantMsg: UIMessage = {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        isStreaming: true,
      }
      dispatch({ type: 'ADD_MESSAGE', payload: assistantMsg })

      const controller = new AbortController()
      abortRef.current = controller

      try {
        const request = {
          message: text,
          conversation_id: state.conversationId ?? undefined,
          document_ids: state.selectedDocumentIds.length > 0
            ? state.selectedDocumentIds
            : undefined,
          model_provider: state.selectedProvider as 'claude' | 'ollama' | 'huggingface',
          model_name: state.selectedModel,
          temperature: state.temperature,
          max_tokens: 2048,
          stream: true,
        }

        let fullContent = ''
        let convId = state.conversationId

        for await (const chunk of chatApi.streamChatFetch(request, controller.signal)) {
          fullContent += chunk
          dispatch({
            type: 'UPDATE_MESSAGE',
            payload: { id: assistantId, updates: { content: fullContent } },
          })
        }

        // Non-streaming fallback / get conversation ID
        if (!convId) {
          const nonStreamRequest = { ...request, stream: false }
          const resp = await chatApi.send(nonStreamRequest)
          convId = resp.conversation_id
          dispatch({ type: 'SET_CONVERSATION_ID', payload: convId })
          if (!fullContent) {
            dispatch({
              type: 'UPDATE_MESSAGE',
              payload: { id: assistantId, updates: { content: resp.message.content } },
            })
          }
        }

        dispatch({
          type: 'UPDATE_MESSAGE',
          payload: { id: assistantId, updates: { isStreaming: false } },
        })
      } catch (err: unknown) {
        if ((err as Error).name === 'AbortError') return
        const errorMsg = err instanceof Error ? err.message : 'Unknown error'
        showError('Chat error', errorMsg)
        dispatch({
          type: 'UPDATE_MESSAGE',
          payload: {
            id: assistantId,
            updates: { isStreaming: false, error: errorMsg, content: 'Sorry, an error occurred.' },
          },
        })
      } finally {
        dispatch({ type: 'SET_LOADING', payload: false })
        abortRef.current = null
      }
    },
    [state, dispatch, showError, showWarning]
  )

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort()
    dispatch({ type: 'SET_LOADING', payload: false })
  }, [dispatch])

  const clearChat = useCallback(() => {
    if (state.conversationId) {
      chatApi.clearConversation(state.conversationId).catch(() => {})
    }
    dispatch({ type: 'CLEAR_MESSAGES' })
  }, [state.conversationId, dispatch])

  return { sendMessage, stopGeneration, clearChat }
}
