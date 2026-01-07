import React, { useState } from 'react'
import './App.css'

/*
  AI Price Optimization Dashboard
  - Implements a form with default values for 18 features
  - Sends POST /predict to the backend and displays recommended price
  - Uses inline styles and `useState` for state handling
*/

export default function App() {
  const [form, setForm] = useState({
    cost: 25,
    demand: 25,
    inventory: 69,
    competitor_price: 256
  })
  const [loading, setLoading] = useState(false)
  const [recommended, setRecommended] = useState(null)
  const [error, setError] = useState(null)
  const [modelUp, setModelUp] = useState(true)

  const handleChange = (e) => {
    const { name, value } = e.target
    setForm((f) => ({ ...f, [name]: value }))
  }

  // Send POST request to FastAPI backend (only when all fields filled)
  const handlePredict = async () => {
    setError(null)
    setRecommended(null)

    // Validate required fields
    const required = ['cost', 'demand', 'inventory', 'competitor_price']
    const missing = required.filter((k) => form[k] === '' || form[k] === null || form[k] === undefined)
    if (missing.length > 0) {
      setError('Please enter values for cost, demand, inventory and competitor price.')
      return
    }

    setLoading(true)


    // Build payload: convert types
    const payload = {
      cost: Number(form.cost),
      demand: Number(form.demand),
      inventory: parseInt(form.inventory, 10),
      competitor_price: Number(form.competitor_price)
    }

    try {
      const res = await fetch('http://127.0.0.1:8000/predict_price', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Server ${res.status}: ${text}`)
      }

      const data = await res.json()
      setRecommended(data.recommended_price)
      setModelUp(true)
    } catch (err) {
      setError(err.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>AI PriceOptima</h1>
      <p className="subtitle">Real-Time Dynamic Pricing Recommendation</p>

      <div className="grid">
        <div className="card">
          <h3>Input Panel</h3>
          <p className="small">Enter four key pricing factors. Backend auto-generates remaining features.</p>

          <label>Cost</label>
          <input name="cost" type="number" value={form.cost} onChange={handleChange} />

          <label>Demand</label>
          <input name="demand" type="number" value={form.demand} onChange={handleChange} />

          <label>Inventory</label>
          <input name="inventory" type="number" value={form.inventory} onChange={handleChange} />

          <label>Competitor Price</label>
          <input name="competitor_price" type="number" value={form.competitor_price} onChange={handleChange} />

          <div style={{ display: 'flex', gap: 10, marginTop: 12 }}>
            <button onClick={handlePredict} disabled={loading} className="primary">
              {loading ? 'Predicting…' : 'Predict Price'}
            </button>
            <button
              onClick={() => {
                setForm({ cost: 25, demand: 25, inventory: 69, competitor_price: 256 })
                setRecommended(null)
                setError(null)
              }}
            >
              Reset
            </button>
          </div>
        </div>

        <div className="card output">
          <div className="output-header">
            <h3>Recommended Price</h3>
            <div className="model-status">Model Status: {modelUp ? '✓' : '✗'}</div>
          </div>

          <div className="price">{recommended ? `₹ ${Number(recommended).toFixed(2)}` : '--'}</div>

          <p className="note">This price is generated using AI-based demand and inventory analysis.</p>

          {error && <div className="error">Error: {error}</div>}
        </div>
      </div>
    </div>
  )
}
