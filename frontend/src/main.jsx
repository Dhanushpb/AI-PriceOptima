import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './style.css'

// Mount React app to #root
const container = document.getElementById('root')
const root = createRoot(container)
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
