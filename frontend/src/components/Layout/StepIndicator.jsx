import React from 'react'
import { useProject } from '../../context/ProjectContext'

export default function StepIndicator() {
  const { state } = useProject()
  const completed = state.currentProject?.steps_completed || []
  const pct = Math.round((completed.length / 7) * 100)
  return (
    <div className="p-4 border-b bg-white">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium">Progress: {pct}%</div>
        <div className="w-1/2 bg-gray-200 h-2 rounded overflow-hidden">
          <div className="h-2 bg-indigo-600" style={{ width: `${pct}%` }} />
        </div>
      </div>
    </div>
  )
}
