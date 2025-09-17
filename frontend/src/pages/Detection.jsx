import React, { useState, useRef, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Alert,
  CircularProgress,
  Chip,
  Paper,
  Stack
} from '@mui/material'
import {
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  Search as SearchIcon,
  FlightTakeoff as DroneIcon,
  PlayArrow as PlayIcon
} from '@mui/icons-material'
import { motion } from 'framer-motion'
import axios from 'axios'

const Detection = () => {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [detectionResults, setDetectionResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showDemo, setShowDemo] = useState(false)
  
  const fileInputRef = useRef(null)

  // Load demo detection on component mount
  useEffect(() => {
    loadDemoDetection()
  }, [])

  const loadDemoDetection = async () => {
    try {
      const response = await fetch('/demo_detection_results.json')
      if (response.ok) {
        const demoResults = await response.json()
        setShowDemo(true)
        setPreviewUrl('/demo_detection.jpg')
        setDetectionResults(demoResults)
      }
    } catch (error) {
      console.log('Demo detection not available')
    }
  }

  const tryDemoDetection = () => {
    setShowDemo(true)
    setPreviewUrl('/demo_detection.jpg')
    loadDemoDetection()
    setSelectedFile(null)
    setError(null)
  }
  
  const handleFileSelect = (event) => {
    const file = event.target.files[0]
    if (file) {
      if (file.type.startsWith('image/')) {
        setSelectedFile(file)
        const url = URL.createObjectURL(file)
        setPreviewUrl(url)
        setError(null)
        setShowDemo(false)
        setDetectionResults(null)
      } else {
        setError('Please select a valid image file (JPG, PNG, etc.)')
      }
    }
  }
  
  const clearSelection = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setDetectionResults(null)
    setError(null)
    setShowDemo(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }
  
  return (
    <Box>
      {/* Drone-themed Header */}
      <motion.div
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <Box sx={{ 
          mb: 4, 
          p: 3,
          background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%)',
          borderRadius: 3,
          border: '1px solid rgba(59, 130, 246, 0.2)'
        }}>
          <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
            <motion.div
              animate={{ 
                rotate: [0, 5, -5, 0],
                scale: [1, 1.1, 1] 
              }}
              transition={{ 
                duration: 3, 
                repeat: Infinity,
                repeatType: "reverse" 
              }}
            >
              <DroneIcon sx={{ 
                fontSize: 48, 
                color: '#3b82f6'
              }} />
            </motion.div>
            <Box>
              <Typography variant="h3" component="h1" sx={{ 
                fontWeight: 800, 
                background: 'linear-gradient(45deg, #3b82f6, #10b981)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 0.5
              }}>
                🚁 Drone Rock Detection
              </Typography>
            </Box>
          </Stack>
          
          <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
            🎯 Advanced aerial surveillance system for real-time rockfall detection
          </Typography>
          
          <Button
            variant="contained"
            size="large"
            onClick={tryDemoDetection}
            startIcon={<PlayIcon />}
            sx={{
              background: 'linear-gradient(45deg, #3b82f6, #10b981)',
              color: 'white',
              fontWeight: 700,
              px: 4,
              py: 1.5
            }}
          >
            🚀 Try Live Demo Detection
          </Button>
        </Box>
      </motion.div>
      
      <Grid container spacing={3}>
        {/* Upload Section */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ 
            background: 'rgba(15, 23, 42, 0.8)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: 3
          }}>
            <CardContent>
              <Typography variant="h6" component="div" sx={{ mb: 3, fontWeight: 700 }}>
                📸 Drone Image Upload
              </Typography>
              
              <Paper
                variant="outlined"
                sx={{
                  p: 4,
                  textAlign: 'center',
                  cursor: 'pointer',
                  borderStyle: 'dashed',
                  borderWidth: 2,
                  borderColor: showDemo ? '#10b981' : '#475569',
                  backgroundColor: showDemo ? 'rgba(16, 185, 129, 0.1)' : 'rgba(15, 23, 42, 0.5)',
                  borderRadius: 3
                }}
                onClick={() => !showDemo && fileInputRef.current?.click()}
              >
                {showDemo ? (
                  <div>
                    <Typography variant="h6" sx={{ mb: 1, color: '#10b981', fontWeight: 700 }}>
                      🎯 Drone Surveillance Active
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Real-time aerial rock detection in progress
                    </Typography>
                  </div>
                ) : (
                  <div>
                    <UploadIcon sx={{ fontSize: 48, color: '#64748b', mb: 2 }} />
                    <Typography variant="h6" sx={{ mb: 1 }}>
                      📤 Upload Drone Imagery
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Drop aerial images here or click to upload
                    </Typography>
                  </div>
                )}
                
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                />
              </Paper>
              
              {/* Preview */}
              {previewUrl && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                    {showDemo ? "🎯 Live Drone Feed with AI Detection" : "📸 Uploaded Image"}
                  </Typography>
                  
                  <Box sx={{ 
                    position: 'relative',
                    border: showDemo ? '3px solid #10b981' : '2px solid #475569',
                    borderRadius: 2,
                    overflow: 'hidden'
                  }}>
                    <img
                      src={previewUrl}
                      alt="Preview"
                      style={{
                        width: '100%',
                        maxHeight: '400px',
                        objectFit: 'contain'
                      }}
                    />
                    
                    {showDemo && detectionResults && (
                      <Box sx={{
                        position: 'absolute',
                        top: 8,
                        left: 8,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        color: 'white',
                        px: 2,
                        py: 1,
                        borderRadius: 1
                      }}>
                        <Typography variant="caption" sx={{ fontWeight: 700 }}>
                          🔴 LIVE DETECTION ACTIVE
                        </Typography>
                      </Box>
                    )}
                  </Box>
                  
                  <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                    {showDemo && (
                      <Button
                        variant="contained"
                        onClick={() => {
                          setShowDemo(false)
                          setDetectionResults(null)
                          setPreviewUrl(null)
                        }}
                        startIcon={<UploadIcon />}
                        sx={{ 
                          flex: 1,
                          background: 'linear-gradient(45deg, #10b981, #059669)',
                          fontWeight: 600
                        }}
                      >
                        📤 Upload Your Own Image
                      </Button>
                    )}
                    
                    <Button
                      variant="outlined"
                      onClick={clearSelection}
                      startIcon={<DeleteIcon />}
                    >
                      Clear
                    </Button>
                  </Box>
                </Box>
              )}
              
              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {error}
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Results Section */}
        <Grid item xs={12} lg={6}>
          <Card sx={{ 
            height: 'fit-content',
            background: 'rgba(15, 23, 42, 0.8)',
            border: detectionResults ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: 3
          }}>
            <CardContent>
              <Typography variant="h6" component="div" sx={{ fontWeight: 700, mb: 3 }}>
                🎯 AI Detection Results
              </Typography>
              
              {detectionResults ? (
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Paper sx={{ 
                      p: 3, 
                      textAlign: 'center', 
                      background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.2))',
                      border: '1px solid rgba(59, 130, 246, 0.3)'
                    }}>
                      <Typography variant="h3" sx={{ 
                        fontWeight: 800, 
                        color: '#3b82f6'
                      }}>
                        {detectionResults.total_detections}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
                        🪨 Rocks Detected
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6}>
                    <Paper sx={{ 
                      p: 3, 
                      textAlign: 'center', 
                      background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.2))',
                      border: '1px solid rgba(16, 185, 129, 0.3)'
                    }}>
                      <Typography variant="h3" sx={{ 
                        fontWeight: 800, 
                        color: '#10b981'
                      }}>
                        {detectionResults.detections && detectionResults.detections.length > 0 
                          ? (detectionResults.detections.reduce((sum, det) => sum + det.confidence, 0) / detectionResults.detections.length * 100).toFixed(1)
                          : 0}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
                        🎯 Detection Confidence
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
              ) : (
                <Box sx={{ textAlign: 'center', py: 6, opacity: 0.7 }}>
                  <DroneIcon sx={{ fontSize: 64, color: '#64748b', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
                    🚁 Drone Ready for Analysis
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Upload an aerial image or try the demo to see AI-powered rock detection
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Detection
