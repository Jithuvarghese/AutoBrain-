import axios from 'axios'

const api = axios.create({ baseURL: 'http://localhost:8000' })

// Projects
export const listProjects = () => api.get('/api/projects')
export const createProject = (name) => api.post('/api/projects', { name })
export const getProject = (id) => api.get(`/api/projects/${id}`)
export const deleteProject = (id) => api.delete(`/api/projects/${id}`)
export const renameProject = (id, name) => api.put(`/api/projects/${id}/name`, { name })

// Upload
export const uploadDataset = (projectId, file) => {
  const form = new FormData(); form.append('file', file)
  return api.post(`/api/upload/${projectId}`, form)
}
export const getPreview = (projectId) => api.get(`/api/upload/${projectId}/preview`)

// Preprocessing
export const getPreprocessingSuggestions = (projectId) => api.get(`/api/preprocessing/${projectId}/suggestions`)
export const applyAutoPreprocessing = (projectId) => api.post(`/api/preprocessing/${projectId}/auto`)
export const applyManualPreprocessing = (projectId, actions) => api.post(`/api/preprocessing/${projectId}/manual`, { actions })

// Feature Engineering
export const getColumns = (projectId) => api.get(`/api/features/${projectId}/columns`)
export const applyFeatureActions = (projectId, actions) => api.post(`/api/features/${projectId}/apply`, { actions })
export const resetFeatures = (projectId) => api.post(`/api/features/${projectId}/reset`)

// Sampling
export const configureSampling = (projectId, config) => api.post(`/api/sampling/${projectId}/configure`, config)
export const getSamplingInfo = (projectId) => api.get(`/api/sampling/${projectId}/info`)

// Training
export const getAlgorithms = (projectId) => api.get(`/api/training/${projectId}/algorithms`)
export const trainModel = (projectId, config) => api.post(`/api/training/${projectId}/train`, config)

// Evaluation
export const evaluateModel = (projectId) => api.post(`/api/evaluation/${projectId}/evaluate`)
export const getEvaluationResults = (projectId) => api.get(`/api/evaluation/${projectId}/results`)

// Documentation
export const generateDocumentation = (projectId) => api.get(`/api/documentation/${projectId}/generate`)

export default api
