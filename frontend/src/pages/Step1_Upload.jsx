import React, { useCallback, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { uploadDataset, getPreview, getProject } from '../api/api'
import { useProject } from '../context/ProjectContext'
import { Spinner, MetricCard, DataTable } from '../components/UI/index'

export default function Step1_Upload() {
  const { projectId } = useParams()
  const { dispatch } = useProject()
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [uploadInfo, setUploadInfo] = useState(null)
  const navigate = useNavigate()

  const fetchPreview = useCallback(async () => {
    try {
      const res = await getPreview(projectId)
      setUploadInfo(res.data)
    } catch (err) {
      console.error(err)
    }
  }, [projectId])

  const handleFiles = async (files) => {
    if (!files || files.length === 0) return
    const file = files[0]
    setLoading(true)
    try {
      await uploadDataset(projectId, file)
      // backend updates state.json; fetch full project
      const full = await getProject(projectId)
      dispatch({ type: 'SET_PROJECT', payload: full.data })
      await fetchPreview()
    } catch (err) {
      console.error(err)
      alert('Upload failed')
    } finally {
      setLoading(false)
    }
  }

  const onDrop = async (e) => {
    e.preventDefault()
    setDragging(false)
    await handleFiles(e.dataTransfer.files)
  }

  const onFileChange = async (e) => {
    await handleFiles(e.target.files)
  }

  return (
    <div className="p-4">
      <h2 className="text-xl font-semibold mb-4">Step 1 — Upload Dataset</h2>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`card border-dashed border-2 p-8 text-center ${dragging ? 'border-indigo-500 scale-101' : ''}`}
      >
        {loading ? (
          <div className="flex flex-col items-center">
            <Spinner />
            <div className="mt-2">Processing...</div>
          </div>
        ) : (
          <>
            <div className="text-lg font-medium">Drag & drop your CSV or Excel file here</div>
            <div className="text-sm text-gray-500 mt-2">or</div>
            <div className="mt-3">
              <input type="file" accept=".csv,.xlsx,.xls" onChange={onFileChange} />
            </div>
          </>
        )}
      </div>

      {uploadInfo && (
        <div className="mt-6 space-y-4">
          <div className="grid grid-cols-4 gap-4">
            <MetricCard label="Total Rows" value={uploadInfo.rows} />
            <MetricCard label="Total Columns" value={uploadInfo.columns} />
            <MetricCard label="Duplicate Rows" value={uploadInfo.duplicate_count} />
            <MetricCard label="Columns with Nulls" value={Object.values(uploadInfo.null_counts || {}).filter(v => v>0).length} />
          </div>

          <div>
            <h3 className="font-semibold mb-2">Column Overview</h3>
            <DataTable columns={["name","dtype","non_null_count","null_count","null_pct"]} data={uploadInfo.column_info || []} />
          </div>

          <div>
            <h3 className="font-semibold mb-2">Data Preview</h3>
            <DataTable columns={uploadInfo.column_names || []} data={(uploadInfo.preview || []).slice(0,10)} />
          </div>

          <div className="flex justify-end">
            <button className="btn-primary" onClick={() => navigate(`/project/${projectId}/step/2`)}>Next: Preprocessing →</button>
          </div>
        </div>
      )}
    </div>
  )
}
