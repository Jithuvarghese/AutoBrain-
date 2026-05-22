import React, { createContext, useContext, useReducer } from 'react'

const initialState = {
  currentProject: null,
  loading: false,
  error: null,
}

const ProjectContext = createContext(null)

function reducer(state, action) {
  switch (action.type) {
    case 'SET_PROJECT':
      return { ...state, currentProject: action.payload }
    case 'SET_LOADING':
      return { ...state, loading: action.payload }
    case 'SET_ERROR':
      return { ...state, error: action.payload }
    case 'UPDATE_STATE':
      return { ...state, currentProject: action.payload }
    case 'CLEAR_PROJECT':
      return { ...state, currentProject: null }
    default:
      return state
  }
}

export function ProjectProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState)
  return <ProjectContext.Provider value={{ state, dispatch }}>{children}</ProjectContext.Provider>
}

export function useProject() {
  const ctx = useContext(ProjectContext)
  if (!ctx) throw new Error('useProject must be used within ProjectProvider')
  return ctx
}
