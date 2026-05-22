import React, { useEffect, useState } from 'react'
import { createProject, listProjects } from '../api/api'

export default function Home() {
  const [projects, setProjects] = useState([])
  const [name, setName] = useState('')

  useEffect(() => {
    listProjects().then((r) => setProjects(r.data || []))
  }, [])

  const handleCreate = async () => {
    if (!name) return
    await createProject(name)
    const res = await listProjects()
    setProjects(res.data || [])
    setName('')
  }

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <div className="text-2xl font-bold">🤖 AutoBrain</div>
          <div className="text-sm text-gray-500">Auto ML, zero code</div>
        </div>
        <div className="flex gap-2">
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="New project name" />
          <button className="btn-primary" onClick={handleCreate}>New Project</button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {projects.length === 0 && <div>No projects yet</div>}
        {projects.map((p) => (
          <div key={p.project_id} className="card">
            <div className="flex justify-between items-center">
              <div>
                <div className="font-semibold">{p.project_name}</div>
                <div className="text-xs text-gray-500">{p.created_at}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
