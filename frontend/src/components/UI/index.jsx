import React from 'react'

export function Spinner({ size = 'md' }) {
  const dims = size === 'sm' ? 4 : size === 'lg' ? 12 : 6
  return <div className={`animate-spin rounded-full border-2 border-t-indigo-600 w-${dims} h-${dims}`} />
}

export function Badge({ children, color = 'indigo' }) {
  const bg = color === 'green' ? 'bg-green-100 text-green-800' : 'bg-indigo-100 text-indigo-800'
  return <span className={`px-2 py-1 text-xs rounded ${bg}`}>{children}</span>
}

export function Alert({ type = 'info', children }) {
  const bg = type === 'error' ? 'bg-red-100 text-red-800' : 'bg-blue-100 text-blue-800'
  return <div className={`${bg} p-3 rounded`}>{children}</div>
}

export function MetricCard({ label, value, subtitle }) {
  return (
    <div className="card">
      <div className="text-sm text-gray-500">{label}</div>
      <div className="text-2xl font-bold">{value}</div>
      {subtitle && <div className="text-xs text-gray-400">{subtitle}</div>}
    </div>
  )
}

export function DataTable({ columns = [], data = [], maxHeight = '400px' }) {
  return (
    <div style={{ maxHeight, overflow: 'auto' }} className="border rounded">
      <table className="min-w-full table-fixed">
        <thead className="sticky top-0 bg-white">
          <tr>
            {columns.map((c) => (
              <th className="p-2 text-left" key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className="hover:bg-gray-50">
              {columns.map((c) => (
                <td className="p-2 text-sm" key={c}>{row[c] ?? <i>null</i>}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Modal({ open, onClose, title, children }) {
  if (!open) return null
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center">
      <div className="bg-white rounded p-4 w-1/2">
        <div className="flex justify-between items-center mb-4">
          <div className="font-bold">{title}</div>
          <button onClick={onClose}>✕</button>
        </div>
        {children}
      </div>
    </div>
  )
}

export function SectionHeader({ title, subtitle, action }) {
  return (
    <div className="flex items-center justify-between mb-4">
      <div>
        <div className="text-lg font-semibold">{title}</div>
        {subtitle && <div className="text-sm text-gray-500">{subtitle}</div>}
      </div>
      <div>{action}</div>
    </div>
  )
}

export default { Spinner, Badge, Alert, MetricCard, DataTable, Modal, SectionHeader }
