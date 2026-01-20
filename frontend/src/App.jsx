import { useState, useEffect } from 'react'
import './App.css'

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [numRows, setNumRows] = useState(1000)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [uploadFile, setUploadFile] = useState(null)
  const [epochs, setEpochs] = useState(100)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [isTraining, setIsTraining] = useState(false)
  const [showColumns, setShowColumns] = useState(false)
  const [healthCheck, setHealthCheck] = useState(null)

  const fetchHealthCheck = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`)
      if (response.ok) {
        const data = await response.json()
        setHealthCheck(data)
      }
    } catch (err) {
      console.error('Failed to fetch health check:', err)
    }
  }

  const fetchModelInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/model-info`)
      if (response.ok) {
        const data = await response.json()
        if (data.success && data.data) {
          setModelInfo(data.data)
        } else {
          setModelInfo(null)
        }
      }
    } catch (err) {
      console.error('Failed to fetch model info:', err)
      setModelInfo(null)
    }
  }

  const checkTrainingStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/training-status`)
      if (response.ok) {
        const status = await response.json()
        console.log('Training status:', status)
        setTrainingStatus(status)
        setIsTraining(status.is_training)
        
        if (!status.is_training && status.progress === 100) {
          fetchModelInfo()
        }
      }
    } catch (err) {
      console.error('Failed to fetch training status:', err)
    }
  }

  useEffect(() => {
    fetchHealthCheck()
    fetchModelInfo()
    checkTrainingStatus()
    
    const interval = setInterval(checkTrainingStatus, 2000)
    return () => clearInterval(interval)
  }, [])

  const handleGenerate = async () => {
    setError(null)
    setSuccess(null)
    setLoading(true)

    try {
      if (numRows < 1 || numRows > 100000) {
        throw new Error('Number of rows must be between 1 and 100,000')
      }

      const response = await fetch(
        `${API_BASE_URL}/generate?n=${numRows}`,
        {
          method: 'POST',
          headers: {
            'Accept': 'text/csv',
          },
        }
      )

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to generate data')
      }

      const blob = await response.blob()
      
      const contentDisposition = response.headers.get('Content-Disposition')
      let filename = 'synthetic_data.csv'
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/)
        if (filenameMatch) {
          filename = filenameMatch[1]
        }
      }

      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)

      setSuccess(`Successfully generated ${numRows} rows! Check your downloads.`)

    } catch (err) {
      setError(err.message || 'An unexpected error occurred')
      console.error('Generation error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !loading) {
      handleGenerate()
    }
  }

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file && file.name.endsWith('.csv')) {
      setUploadFile(file)
      setError(null)
    } else {
      setError('Please select a valid CSV file')
      setUploadFile(null)
    }
  }

  const handleUploadAndTrain = async () => {
    console.log('handleUploadAndTrain called', { uploadFile, epochs, loading, isTraining })
    
    if (!uploadFile) {
      setError('Please select a CSV file first')
      return
    }

    setError(null)
    setSuccess(null)
    setLoading(true)
    setIsTraining(true)

    try {
      const formData = new FormData()
      formData.append('file', uploadFile)

      console.log('Sending request to:', `${API_BASE_URL}/upload-train?epochs=${epochs}`)
      
      const response = await fetch(
        `${API_BASE_URL}/upload-train?epochs=${epochs}`,
        {
          method: 'POST',
          body: formData,
        }
      )

      console.log('Response status:', response.status)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to upload and train')
      }

      const result = await response.json()
      console.log('Training result:', result)
      
      setSuccess(`Training started! Dataset: ${result.rows} rows, ${result.columns.length} columns`)
      setUploadFile(null)
      const fileInput = document.getElementById('fileInput')
      if (fileInput) fileInput.value = ''
      
    } catch (err) {
      setError(err.message || 'An unexpected error occurred')
      console.error('Upload error:', err)
      setIsTraining(false)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="dashboard">
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <div className="header-icon">
              <span className="material-symbols-outlined">neurology</span>
            </div>
            <div>
              <h1 className="header-title">SynthoML Dashboard</h1>
              <p className="header-subtitle">Training & Generation Pipeline</p>
            </div>
          </div>
          <div className="header-right">
            <div className="status-badge">
              <span className="status-dot"></span>
              <span className="status-text">System Online</span>
            </div>
          </div>
        </div>
      </header>

      <main className="main-content">
        {success && <div className="alert alert-success">✅ {success}</div>}
        {error && <div className="alert alert-error">❌ {error}</div>}
        {trainingStatus?.is_training && (
          <div className="alert alert-info">
            <div className="training-progress">
              <div className="training-progress-header">
                <strong>⚡ Training in Progress</strong>
                <span className="training-epoch">
                  Epoch {trainingStatus.current_epoch} / {trainingStatus.total_epochs}
                </span>
              </div>
              <div className="progress-bar">
                <div 
                  className="progress-bar-fill" 
                  style={{ width: `${trainingStatus.progress}%` }}
                ></div>
              </div>
              <p className="training-message">{trainingStatus.message}</p>
            </div>
          </div>
        )}

        <div className="grid">
          <div className="card">
            <div className="card-header">
              <div className="card-header-left">
                <div className="card-icon upload-icon">
                  <span className="material-symbols-outlined">cloud_upload</span>
                </div>
                <div>
                  <h2 className="card-title">Train Model</h2>
                  <p className="card-subtitle">Upload dataset and configure training</p>
                </div>
              </div>
            </div>

            <div className="card-body">
              <div className="upload-zone">
                <input
                  id="fileInput"
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  disabled={loading || isTraining}
                  className="file-input"
                />
                <label htmlFor="fileInput" className="upload-label">
                  <span className="material-symbols-outlined upload-icon">folder_open</span>
                  <p className="upload-text">
                    <span className="upload-highlight">Click to upload</span> or drag and drop
                  </p>
                  <p className="upload-hint">CSV files only (MAX. 100MB)</p>
                </label>
              </div>

              {uploadFile && (
                <div className="file-chip">
                  <div className="file-chip-content">
                    <div className="file-chip-icon">
                      <span className="material-symbols-outlined">description</span>
                    </div>
                    <div className="file-chip-info">
                      <span className="file-chip-name">{uploadFile.name}</span>
                      <span className="file-chip-size">
                        {(uploadFile.size / 1024).toFixed(2)} KB
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      setUploadFile(null)
                      document.getElementById('fileInput').value = ''
                    }}
                    className="file-chip-remove"
                  >
                    ✕
                  </button>
                </div>
              )}

              <hr className="divider" />

              <div className="form-group">
                <label className="form-label">
                  Training Epochs
                  <span className="form-label-hint">(10 - 1000)</span>
                </label>
                <div className="input-wrapper">
                  <input
                    type="number"
                    min="10"
                    max="1000"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value) || 100)}
                    disabled={loading || isTraining}
                    className="input"
                    placeholder="100"
                  />
                  <span className="material-symbols-outlined input-icon">restart_alt</span>
                </div>
                <p className="form-hint">Higher epochs may improve accuracy but take longer.</p>
              </div>
            </div>

            <div className="card-footer">
              <button
                onClick={handleUploadAndTrain}
                disabled={!uploadFile || loading || isTraining}
                className="button button-primary"
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    <span>Training...</span>
                  </>
                ) : (
                  <>
                    <span className="material-symbols-outlined button-icon">rocket_launch</span>
                    <span>Start Training</span>
                  </>
                )}
              </button>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div className="card-header-left">
                <div className="card-icon generate-icon">
                  <span className="material-symbols-outlined">casino</span>
                </div>
                <div>
                  <h2 className="card-title">Generate Data</h2>
                  <p className="card-subtitle">Create synthetic records</p>
                </div>
              </div>
              {modelInfo && (
                <div className="model-ready-badge">
                  <span className="material-symbols-outlined badge-icon">check_circle</span>
                  <span>Model Ready</span>
                </div>
              )}
            </div>

            <div className="card-body">
              {modelInfo ? (
                <>
                  <div className="stats-grid">
                    <div className="stat-card">
                      <span className="stat-label">Library</span>
                      <div className="stat-value">
                        <span className="material-symbols-outlined stat-icon">deployed_code</span>
                        <span>{modelInfo.library.toUpperCase()}</span>
                      </div>
                    </div>
                    <div className="stat-card">
                      <span className="stat-label">Columns Detected</span>
                      <div className="stat-value">
                        <span className="material-symbols-outlined stat-icon">view_column</span>
                        <span>{modelInfo.columns.length} Fields</span>
                      </div>
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">
                      Number of Rows to Generate
                      <span className="form-label-hint">(1 - 100k)</span>
                    </label>
                    <div className="input-wrapper">
                      <input
                        type="number"
                        min="1"
                        max="100000"
                        value={numRows}
                        onChange={(e) => setNumRows(parseInt(e.target.value) || 0)}
                        onKeyPress={handleKeyPress}
                        disabled={loading || isTraining}
                        className="input"
                        placeholder="1000"
                      />
                      <span className="material-symbols-outlined input-icon">123</span>
                    </div>
                  </div>

                  <div className="columns-details">
                    <button
                      onClick={() => setShowColumns(!showColumns)}
                      className="columns-toggle"
                    >
                      <div className="columns-toggle-left">
                        <span className="material-symbols-outlined columns-icon">table_chart</span>
                        <span>View Schema Columns</span>
                      </div>
                      <span className={`toggle-arrow ${showColumns ? 'open' : ''}`}>▼</span>
                    </button>
                    {showColumns && (
                      <div className="columns-list">
                        {modelInfo.columns.map((col, i) => (
                          <span key={i} className="column-tag">{col}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="empty-state">
                  <span className="material-symbols-outlined empty-icon">model_training</span>
                  <p className="empty-title">Model Not Ready</p>
                  <p className="empty-text">Train a model on the left to unlock generation.</p>
                </div>
              )}
            </div>

            <div className="card-footer">
              <button
                onClick={handleGenerate}
                disabled={loading || numRows < 1 || !modelInfo || isTraining}
                className="button button-success"
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <span className="material-symbols-outlined button-icon">download</span>
                    <span>Generate & Download</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </main>

      <div className="bg-decoration">
        <div className="decoration-1"></div>
        <div className="decoration-2"></div>
      </div>
    </div>
  )
}

export default App
