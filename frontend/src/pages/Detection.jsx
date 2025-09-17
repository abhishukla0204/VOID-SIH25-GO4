import React, { useState, useRef } from 'react'
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
  List,
  ListItem,
  ListItemText,
  Divider,
  ImageList,
  ImageListItem,
  ImageListItemBar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material'
import {
  CloudUpload as UploadIcon,
  PhotoCamera as CameraIcon,
  Visibility as ViewIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
  Search as SearchIcon
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'

const Detection = () => {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [detectionResults, setDetectionResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [detectionHistory, setDetectionHistory] = useState([])
  const [selectedDetection, setSelectedDetection] = useState(null)
  const [dialogOpen, setDialogOpen] = useState(false)
  
  const fileInputRef = useRef(null)
  
  const handleFileSelect = (event) => {
    const file = event.target.files[0]
    if (file) {
      if (file.type.startsWith('image/')) {
        setSelectedFile(file)
        const url = URL.createObjectURL(file)
        setPreviewUrl(url)
        setError(null)
      } else {
        setError('Please select a valid image file (JPG, PNG, etc.)')
      }
    }
  }
  
  const handleDetection = async () => {
    if (!selectedFile) {
      setError('Please select an image first')
      return
    }
    
    setLoading(true)
    setError(null)
    
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      
      const response = await axios.post(
        'http://localhost:8000/api/detect-rocks',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      )
      
      const results = response.data
      setDetectionResults(results)
      
      // Add to history
      const newDetection = {
        id: Date.now(),
        filename: selectedFile.name,
        timestamp: new Date().toLocaleString(),
        total_detections: results.total_detections,
        confidence: results.detections.length > 0 
          ? (results.detections.reduce((sum, det) => sum + det.confidence, 0) / results.detections.length).toFixed(2)
          : 0,
        image_url: previewUrl,
        results: results
      }
      
      setDetectionHistory(prev => [newDetection, ...prev.slice(0, 9)]) // Keep last 10
      
    } catch (err) {
      setError('Failed to detect rocks. Please try again.')
      console.error('Detection error:', err)
    } finally {
      setLoading(false)
    }
  }
  
  const handleDragOver = (event) => {
    event.preventDefault()
  }
  
  const handleDrop = (event) => {
    event.preventDefault()
    const files = Array.from(event.dataTransfer.files)
    if (files.length > 0) {
      const file = files[0]
      if (file.type.startsWith('image/')) {
        setSelectedFile(file)
        const url = URL.createObjectURL(file)
        setPreviewUrl(url)
        setError(null)
      }
    }
  }
  
  const clearSelection = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setDetectionResults(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }
  
  const downloadResults = () => {
    if (detectionResults) {
      const dataStr = JSON.stringify(detectionResults, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      const url = URL.createObjectURL(dataBlob)
      const link = document.createElement('a')
      link.href = url
      link.download = `detection_results_${Date.now()}.json`
      link.click()
      URL.revokeObjectURL(url)
    }
  }
  
  const viewDetectionDetails = (detection) => {
    setSelectedDetection(detection)
    setDialogOpen(true)
  }
  
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#10b981' // High confidence - green
    if (confidence >= 0.6) return '#f59e0b' // Medium confidence - yellow
    return '#ef4444' // Low confidence - red
  }
  
  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 700, mb: 1 }}>
          Rock Detection
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Upload images to detect and analyze rocks using AI-powered detection
        </Typography>
      </Box>
      
      <Grid container spacing={3}>
        {/* Upload Section */}
        <Grid item xs={12} lg={6}>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card className="glass-card">
              <CardContent>
                <Typography variant="h6" component="div" sx={{ mb: 3 }}>
                  Image Upload
                </Typography>
                
                {/* Upload Area */}
                <Paper
                  variant="outlined"
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  sx={{
                    p: 4,
                    textAlign: 'center',
                    cursor: 'pointer',
                    borderStyle: 'dashed',
                    borderWidth: 2,
                    borderColor: '#475569',
                    backgroundColor: 'rgba(15, 23, 42, 0.5)',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      borderColor: '#3b82f6',
                      backgroundColor: 'rgba(59, 130, 246, 0.1)'
                    }
                  }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <UploadIcon sx={{ fontSize: 48, color: '#64748b', mb: 2 }} />
                  <Typography variant="h6" sx={{ mb: 1 }}>
                    Drop image here or click to upload
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Supports JPG, PNG, GIF formats
                  </Typography>
                  
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
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.2 }}
                  >
                    <Box sx={{ mt: 3 }}>
                      <img
                        src={previewUrl}
                        alt="Preview"
                        style={{
                          width: '100%',
                          maxHeight: '300px',
                          objectFit: 'contain',
                          borderRadius: '8px'
                        }}
                      />
                      <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                        <Button
                          variant="contained"
                          onClick={handleDetection}
                          disabled={loading}
                          startIcon={loading ? <CircularProgress size={20} /> : <SearchIcon />}
                          sx={{ flex: 1 }}
                        >
                          {loading ? 'Detecting...' : 'Detect Rocks'}
                        </Button>
                        <Button
                          variant="outlined"
                          onClick={clearSelection}
                          startIcon={<DeleteIcon />}
                        >
                          Clear
                        </Button>
                      </Box>
                    </Box>
                  </motion.div>
                )}
                
                {/* Error Display */}
                {error && (
                  <Alert severity="error" sx={{ mt: 2 }}>
                    {error}
                  </Alert>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        {/* Results Section */}
        <Grid item xs={12} lg={6}>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card className="glass-card" sx={{ height: 'fit-content' }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography variant="h6" component="div">
                    Detection Results
                  </Typography>
                  {detectionResults && (
                    <Button
                      size="small"
                      onClick={downloadResults}
                      startIcon={<DownloadIcon />}
                    >
                      Download
                    </Button>
                  )}
                </Box>
                
                {detectionResults ? (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                  >
                    <Box sx={{ mb: 3 }}>
                      <Grid container spacing={2} sx={{ mb: 2 }}>
                        <Grid item xs={6}>
                          <Paper sx={{ p: 2, textAlign: 'center', backgroundColor: 'rgba(59, 130, 246, 0.1)' }}>
                            <Typography variant="h4" sx={{ fontWeight: 700, color: '#3b82f6' }}>
                              {detectionResults.total_detections}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Rocks Detected
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={6}>
                          <Paper sx={{ p: 2, textAlign: 'center', backgroundColor: 'rgba(16, 185, 129, 0.1)' }}>
                            <Typography variant="h4" sx={{ fontWeight: 700, color: '#10b981' }}>
                              {detectionResults.detections.length > 0 
                                ? (detectionResults.detections.reduce((sum, det) => sum + det.confidence, 0) / detectionResults.detections.length * 100).toFixed(1)
                                : 0}%
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Avg Confidence
                            </Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                      
                      {detectionResults.detections.length > 0 && (
                        <Box>
                          <Typography variant="subtitle2" sx={{ mb: 2 }}>
                            Individual Detections:
                          </Typography>
                          <List>
                            {detectionResults.detections.map((detection, index) => (
                              <React.Fragment key={index}>
                                <ListItem>
                                  <ListItemText
                                    primary={`Rock ${index + 1}`}
                                    secondary={
                                      <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                                        <Chip
                                          label={`${(detection.confidence * 100).toFixed(1)}%`}
                                          size="small"
                                          sx={{
                                            backgroundColor: getConfidenceColor(detection.confidence),
                                            color: 'white'
                                          }}
                                        />
                                        <Chip
                                          label={`${detection.bbox[2] - detection.bbox[0]}×${detection.bbox[3] - detection.bbox[1]}px`}
                                          size="small"
                                          variant="outlined"
                                        />
                                      </Box>
                                    }
                                  />
                                </ListItem>
                                {index < detectionResults.detections.length - 1 && (
                                  <Divider variant="inset" component="li" sx={{ borderColor: '#334155' }} />
                                )}
                              </React.Fragment>
                            ))}
                          </List>
                        </Box>
                      )}
                    </Box>
                  </motion.div>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <CameraIcon sx={{ fontSize: 64, color: '#64748b', mb: 2 }} />
                    <Typography variant="body1" color="text.secondary">
                      Upload an image to see detection results
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        {/* Detection History */}
        {detectionHistory.length > 0 && (
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Card className="glass-card">
                <CardContent>
                  <Typography variant="h6" component="div" sx={{ mb: 3 }}>
                    Recent Detections
                  </Typography>
                  
                  <ImageList
                    sx={{ width: '100%', height: 400 }}
                    cols={4}
                    rowHeight={200}
                    gap={8}
                  >
                    <AnimatePresence>
                      {detectionHistory.map((detection, index) => (
                        <motion.div
                          key={detection.id}
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.8 }}
                          transition={{ delay: index * 0.1 }}
                        >
                          <ImageListItem>
                            <img
                              src={detection.image_url}
                              alt={detection.filename}
                              loading="lazy"
                              style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover',
                                borderRadius: '8px'
                              }}
                            />
                            <ImageListItemBar
                              title={detection.filename}
                              subtitle={`${detection.total_detections} rocks • ${detection.confidence}% avg`}
                              actionIcon={
                                <Button
                                  size="small"
                                  onClick={() => viewDetectionDetails(detection)}
                                  sx={{ color: 'white' }}
                                >
                                  <ViewIcon />
                                </Button>
                              }
                              sx={{
                                background: 'linear-gradient(to top, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.3) 70%, rgba(0,0,0,0) 100%)'
                              }}
                            />
                          </ImageListItem>
                        </motion.div>
                      ))}
                    </AnimatePresence>
                  </ImageList>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        )}
      </Grid>
      
      {/* Detection Details Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Detection Details
        </DialogTitle>
        <DialogContent>
          {selectedDetection && (
            <Box>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <img
                    src={selectedDetection.image_url}
                    alt={selectedDetection.filename}
                    style={{
                      width: '100%',
                      maxHeight: '300px',
                      objectFit: 'contain',
                      borderRadius: '8px'
                    }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" sx={{ mb: 2 }}>
                    {selectedDetection.filename}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Processed: {selectedDetection.timestamp}
                  </Typography>
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2">
                      Total Detections: {selectedDetection.total_detections}
                    </Typography>
                    <Typography variant="subtitle2">
                      Average Confidence: {selectedDetection.confidence}%
                    </Typography>
                  </Box>
                  
                  {selectedDetection.results.detections.length > 0 && (
                    <Box>
                      <Typography variant="subtitle2" sx={{ mb: 1 }}>
                        Detection Details:
                      </Typography>
                      {selectedDetection.results.detections.map((det, index) => (
                        <Box key={index} sx={{ mb: 1 }}>
                          <Typography variant="body2">
                            Rock {index + 1}: {(det.confidence * 100).toFixed(1)}% confidence
                          </Typography>
                        </Box>
                      ))}
                    </Box>
                  )}
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default Detection