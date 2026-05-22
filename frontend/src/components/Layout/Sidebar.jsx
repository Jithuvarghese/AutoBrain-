import React from 'react'
import { Link, useParams } from 'react-router-dom'
import { useProject } from '../../context/ProjectContext'

const steps = [
  'Upload Dataset',
  'Preprocessing',
  'Feature Engineering',
  'Target & Sampling',
  'Train Model',
  'Evaluate Model',
  'Documentation',
]

export default function Sidebar() {
  const { state } = useProject()
  const { projectId } = useParams()
  const completed = state.currentProject?.steps_completed || []
  const current = state.currentProject?.current_step || 1

  return (
    <aside className="w-64 bg-white border-r p-4 hidden md:block">
      <div className="space-y-4">
        <div className="text-2xl font-bold">🤖 AutoBrain</div>
        <nav>
          {steps.map((label, i) => {
            const step = i + 1
            const done = completed.includes(step)
            const active = current === step
            const disabled = step > (Math.max(...completed, 0) + 1)
            return (
              <Link key={step} to={`/project/${projectId}/step/${step}`} className={`flex items-center gap-3 p-2 rounded ${done ? 'bg-green-100' : active ? 'bg-indigo-50' : 'hover:bg-gray-50'} ${disabled ? 'opacity-50 pointer-events-none' : ''}`}>
                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-sm ${done ? 'bg-green-500 text-white' : active ? 'ring-2 ring-indigo-400' : 'bg-gray-200'}`}>{done ? '✓' : step}</div>
                <div className="text-sm">{label}</div>
              </Link>
            )
          })}
        </nav>
      </div>
      <div className="mt-auto pt-6">
        <Link to="/" className="text-sm text-gray-600">← All Projects</Link>
      </div>
    </aside>
  )
}
