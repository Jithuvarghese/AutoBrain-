import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { applyFeatureActions, getColumns, getProject, resetFeatures } from '../api/api'
import { useProject } from '../context/ProjectContext'
import { Alert, Badge, MetricCard, SectionHeader, Spinner } from '../components/UI/index'

const ACTION_OPTIONS = [
  { value: 'one_hot_encode', label: 'One-hot encode' },
  { value: 'label_encode', label: 'Label encode' },
  { value: 'min_max_scale', label: 'Min-max scale' },
  { value: 'standard_scale', label: 'Standard scale' },
  { value: 'log_transform', label: 'Log transform' },
  { value: 'drop_column', label: 'Drop column' },
  { value: 'create_interaction', label: 'Create interaction' },
  { value: 'bin_column', label: 'Bin column' },
  { value: 'custom_code', label: 'Custom code' },
]

function makeDraft(columns = []) {
  return {
    type: 'one_hot_encode',
    column: columns[0] || '',
    column2: columns[1] || columns[0] || '',
    params: { bins: 4 },
    code: 'df = df\n',
  }
}

export default function Step3_FeatureEngineering() {
  const { projectId } = useParams()
  const navigate = useNavigate()
  const { dispatch } = useProject()
  const [columns, setColumns] = useState([])
  const [resultColumns, setResultColumns] = useState([])
  const [actions, setActions] = useState([])
  const [draft, setDraft] = useState(makeDraft())
  const [loading, setLoading] = useState(false)
  const [source, setSource] = useState('')

  async function refreshProject() {
    const project = await getProject(projectId)
    dispatch({ type: 'SET_PROJECT', payload: project.data })
  }

  async function loadColumns() {
    try {
      const res = await getColumns(projectId)
      const loadedColumns = res.data?.columns || []
      setSource(res.data?.source || '')
      setColumns(loadedColumns)
      setResultColumns(loadedColumns)
      setDraft((current) => ({ ...makeDraft(loadedColumns), ...current, column: current.column || loadedColumns[0] || '', column2: current.column2 || loadedColumns[1] || loadedColumns[0] || '' }))
    } catch (err) {
      console.error(err)
    }
  }

  useEffect(() => {
    loadColumns()
  }, [projectId])

  useEffect(() => {
    setDraft((current) => ({ ...current, column: columns[0] || '', column2: columns[1] || columns[0] || '' }))
  }, [columns])

  const cards = useMemo(() => [
    { label: 'Columns Available', value: columns.length },
    { label: 'Queued Actions', value: actions.length },
    { label: 'Result Columns', value: resultColumns.length },
  ], [columns.length, actions.length, resultColumns.length])

  const addAction = () => {
    const payload = {
      type: draft.type,
      column: draft.type === 'custom_code' ? null : draft.column || null,
      column2: draft.type === 'create_interaction' ? draft.column2 || null : null,
      params: draft.type === 'bin_column' ? { bins: Number(draft.params?.bins || 4) } : { ...(draft.params || {}) },
      code: draft.type === 'custom_code' ? draft.code || '' : null,
    }
    setActions((prev) => [...prev, payload])
  }

  const applyActions = async () => {
    setLoading(true)
    try {
      const res = await applyFeatureActions(projectId, actions)
      setResultColumns(res.data?.columns || [])
      setActions([])
      await refreshProject()
      await loadColumns()
    } catch (err) {
      alert(err?.response?.data?.detail || 'Feature engineering failed')
    } finally {
      setLoading(false)
    }
  }

  const clearAll = async () => {
    setLoading(true)
    try {
      await resetFeatures(projectId)
      await refreshProject()
      await loadColumns()
      setActions([])
    } catch (err) {
      alert(err?.response?.data?.detail || 'Reset failed')
    } finally {
      setLoading(false)
    }
  }

  const renderActionEditor = () => {
    if (draft.type === 'custom_code') {
      return (
        <div>
          <label className="label">Custom transformation</label>
          <textarea className="input min-h-40 font-mono text-sm" value={draft.code} onChange={(e) => setDraft((prev) => ({ ...prev, code: e.target.value }))} />
        </div>
      )
    }

    if (draft.type === 'create_interaction') {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="label">First column</label>
            <select className="input" value={draft.column} onChange={(e) => setDraft((prev) => ({ ...prev, column: e.target.value }))}>
              {columns.map((column) => <option key={column} value={column}>{column}</option>)}
            </select>
          </div>
          <div>
            <label className="label">Second column</label>
            <select className="input" value={draft.column2} onChange={(e) => setDraft((prev) => ({ ...prev, column2: e.target.value }))}>
              {columns.map((column) => <option key={column} value={column}>{column}</option>)}
            </select>
          </div>
        </div>
      )
    }

    if (draft.type === 'bin_column') {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="label">Column</label>
            <select className="input" value={draft.column} onChange={(e) => setDraft((prev) => ({ ...prev, column: e.target.value }))}>
              {columns.map((column) => <option key={column} value={column}>{column}</option>)}
            </select>
          </div>
          <div>
            <label className="label">Bins</label>
            <input className="input" type="number" min="2" value={draft.params?.bins || 4} onChange={(e) => setDraft((prev) => ({ ...prev, params: { ...prev.params, bins: e.target.value } }))} />
          </div>
        </div>
      )
    }

    return (
      <div>
        <label className="label">Column</label>
        <select className="input" value={draft.column} onChange={(e) => setDraft((prev) => ({ ...prev, column: e.target.value }))}>
          {columns.map((column) => <option key={column} value={column}>{column}</option>)}
        </select>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-6">
      <SectionHeader
        title="Step 3 — Feature Engineering"
        subtitle={`Build model-ready features from ${source || 'the latest dataset'}.`}
        action={
          <div className="flex gap-2">
            <button className="btn-secondary" onClick={loadColumns}>Reload columns</button>
            <button className="btn-danger" onClick={clearAll} disabled={loading}>Reset features</button>
          </div>
        }
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {cards.map((card) => <MetricCard key={card.label} label={card.label} value={card.value} />)}
      </div>

      {columns.length === 0 && <Alert type="error">No columns available yet. Upload or preprocess a dataset first.</Alert>}

      {columns.length === 0 && (
        <div className="card space-y-3 border border-dashed">
          <div className="font-semibold">Nothing to show yet</div>
          <div className="text-sm text-gray-600">
            Step 3 only becomes useful after you upload data and complete Step 2. Right now the project has no columns, so the page has no feature options to display.
          </div>
          <div className="flex flex-wrap gap-2">
            <button className="btn-primary" onClick={() => navigate(`/project/${projectId}/step/1`)}>
              Go to Step 1
            </button>
            <button className="btn-secondary" onClick={() => navigate(`/project/${projectId}/step/2`)}>
              Go to Step 2
            </button>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card space-y-4">
          <div className="font-semibold">Add transformation</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="label">Action</label>
              <select className="input" value={draft.type} onChange={(e) => setDraft((prev) => ({ ...makeDraft(columns), ...prev, type: e.target.value }))}>
                {ACTION_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
              </select>
            </div>
            <div className="flex items-end">
              <button className="btn-secondary w-full" onClick={addAction}>Queue action</button>
            </div>
          </div>

          <div>{renderActionEditor()}</div>

          <div className="flex gap-2">
            <button className="btn-primary" onClick={applyActions} disabled={loading || actions.length === 0}>
              {loading ? <span className="inline-flex items-center gap-2"><Spinner size="sm" />Applying</span> : 'Apply queued actions'}
            </button>
          </div>

          <div>
            <div className="text-sm font-semibold mb-2">Queued actions</div>
            {actions.length === 0 ? (
              <div className="text-sm text-gray-500">No feature engineering actions queued.</div>
            ) : (
              <div className="space-y-2">
                {actions.map((action, index) => (
                  <div key={`${action.type}-${index}`} className="border rounded-lg px-3 py-2 text-sm flex items-center justify-between gap-3">
                    <div className="space-y-1">
                      <div className="font-medium">{action.type}</div>
                      <div className="text-gray-500">
                        {action.column && <span className="mr-2">Column: {action.column}</span>}
                        {action.column2 && <span className="mr-2">Column 2: {action.column2}</span>}
                        {action.code && <span className="mr-2">Custom code</span>}
                        {action.params?.bins && <span>Bins: {action.params.bins}</span>}
                      </div>
                    </div>
                    <button className="text-red-600" onClick={() => setActions((prev) => prev.filter((_, itemIndex) => itemIndex !== index))}>Remove</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="card space-y-4">
          <div className="font-semibold">Current columns</div>
          {resultColumns.length === 0 ? (
            <div className="text-sm text-gray-500">No column summary available yet.</div>
          ) : (
            <div className="space-y-3 max-h-[520px] overflow-auto pr-1">
              {resultColumns.map((column) => (
                <div key={column.name} className="border rounded-lg p-3">
                  <div className="flex items-center justify-between gap-3">
                    <div className="font-medium">{column.name}</div>
                    <Badge>{column.type}</Badge>
                  </div>
                  <div className="text-sm text-gray-500 mt-1">{column.dtype} • {column.unique_values} unique values • {column.null_count} nulls</div>
                  {column.sample_values?.length > 0 && <div className="text-sm mt-2">Examples: {column.sample_values.join(', ')}</div>}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
