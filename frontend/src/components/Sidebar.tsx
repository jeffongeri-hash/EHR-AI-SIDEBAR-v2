import React from 'react'
import {
  LayoutDashboard,
  FileText,
  Zap,
  Settings,
  ChevronLeft,
  ChevronRight,
  Stethoscope,
} from 'lucide-react'
import clsx from 'clsx'
import { useApp } from '@/contexts/AppContext'
import type { Tab } from '@/types'

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: 'home',      label: 'Home',      icon: <LayoutDashboard size={18} /> },
  { id: 'documents', label: 'Documents', icon: <FileText size={18} /> },
  { id: 'fine-tune', label: 'Fine-Tune', icon: <Zap size={18} /> },
  { id: 'settings',  label: 'Settings',  icon: <Settings size={18} /> },
]

export default function Sidebar() {
  const { state, dispatch } = useApp()
  const open = state.isSidebarOpen

  return (
    <div
      className={clsx(
        'relative flex flex-col border-r border-gray-800 bg-gray-900 transition-all duration-300 flex-shrink-0',
        open ? 'w-48' : 'w-14'
      )}
    >
      {/* Logo */}
      <div className="flex items-center h-14 border-b border-gray-800 px-3 gap-2.5 overflow-hidden">
        <div className="w-8 h-8 bg-blue-600 rounded-xl flex items-center justify-center flex-shrink-0">
          <Stethoscope size={16} className="text-white" />
        </div>
        {open && (
          <span className="text-sm font-semibold text-gray-100 whitespace-nowrap">
            EHR AI
          </span>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 flex flex-col gap-0.5 py-3 px-2 overflow-hidden">
        {TABS.map((tab) => {
          const active = state.activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => dispatch({ type: 'SET_TAB', payload: tab.id })}
              title={!open ? tab.label : undefined}
              className={clsx(
                'flex items-center gap-2.5 rounded-xl px-2.5 py-2 transition-all w-full text-left',
                active
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-500 hover:text-gray-200 hover:bg-gray-800'
              )}
            >
              <span className="flex-shrink-0">{tab.icon}</span>
              {open && (
                <span className="text-sm font-medium whitespace-nowrap overflow-hidden">
                  {tab.label}
                </span>
              )}
            </button>
          )
        })}
      </nav>

      {/* Collapse toggle */}
      <div className="px-2 pb-3">
        <button
          onClick={() => dispatch({ type: 'TOGGLE_SIDEBAR' })}
          className={clsx(
            'flex items-center gap-2 rounded-xl px-2.5 py-2 w-full text-gray-600',
            'hover:text-gray-300 hover:bg-gray-800 transition-all'
          )}
          title={open ? 'Collapse sidebar' : 'Expand sidebar'}
        >
          <span className="flex-shrink-0">
            {open ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
          </span>
          {open && <span className="text-xs font-medium whitespace-nowrap">Collapse</span>}
        </button>
      </div>
    </div>
  )
}
