import { BrowserRouter, Routes, Route, Outlet, useParams } from 'react-router-dom'
import { ProjectProvider, useProject } from './context/ProjectContext'
import Sidebar from './components/Layout/Sidebar'
import StepIndicator from './components/Layout/StepIndicator'
import Home from './pages/Home'
import Step1_Upload from './pages/Step1_Upload'
import Step2_Preprocessing from './pages/Step2_Preprocessing'
import Step3_FeatureEngineering from './pages/Step3_FeatureEngineering'
import Step4_Sampling from './pages/Step4_Sampling'
import Step5_Training from './pages/Step5_Training'
import Step6_Evaluation from './pages/Step6_Evaluation'
import Step7_Documentation from './pages/Step7_Documentation'

function ProjectLayout() {
  const { projectId } = useParams()
  const { dispatch } = useProject()
  // on mount: load project through API (left as TODO)
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <StepIndicator />
        <main className="flex-1 overflow-y-auto bg-gray-50 p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <ProjectProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/project/:projectId" element={<ProjectLayout />}>
            <Route path="step/1" element={<Step1_Upload />} />
            <Route path="step/2" element={<Step2_Preprocessing />} />
            <Route path="step/3" element={<Step3_FeatureEngineering />} />
            <Route path="step/4" element={<Step4_Sampling />} />
            <Route path="step/5" element={<Step5_Training />} />
            <Route path="step/6" element={<Step6_Evaluation />} />
            <Route path="step/7" element={<Step7_Documentation />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ProjectProvider>
  )
}
