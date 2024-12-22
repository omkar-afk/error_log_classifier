const router = require('express').Router();
const { execFileSync } = require('child_process');
const db = require('../../db');
const fs = require('fs').promises;
const axios = require('axios');
const express = require('express');
const path = require('path');

let i = 0;
// Use db.errorLogModel.logModel instead of db.errorLogModel
const ErrorLogModel = db.errorLogModel.logModel;

router.get('/count', async (req, res) => {
  try {
    const countAll = await ErrorLogModel.find();
    const count = { 
      critical: countAll.filter(log => log.type === 'Critical').length,
      nonCritical: countAll.filter(log => log.type === 'Noncritical').length,
      warning: countAll.filter(log => log.type === 'Warning').length
    };
    res.json({ count });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get('/top10', async (req, res) => {
  try {
    const top10 = await ErrorLogModel.find({ type: { $ne: null } }).sort({ createdAt: -1 }).limit(6);
    res.json({ top10 });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get('/all/:type', async (req, res) => {
  try {
    const type = req.params.type;
    let allLogs = [];
    if (type) {
      allLogs = await ErrorLogModel.find({ type }).sort({ createdAt: -1 }).limit(50);
    } else {
      allLogs = await ErrorLogModel.find().sort({ createdAt: -1 }).limit(50);
    }
    res.json( allLogs );
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/resolved', async (req, res) => {
  try {
    const { id } = req.body;
    const resolvedLog = await ErrorLogModel.findByIdAndUpdate(id, { resolved: true }, { new: true });
    res.json({ resolvedLog });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

function validateLogFormat(line) {
  if (!line || typeof line !== 'string') {
    return { isValid: false, error: 'Empty or invalid log line' };
  }

  const logPattern = /^\[(.*?)\] \[(error|warn|notice|crit|alert)\] \[client ([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})\] (.+)$/;
  return {
    isValid: logPattern.test(line),
    error: logPattern.test(line) ? null : 'Invalid log format'
  };
}

// Validate file path for security
function validateFilePath(filePath) {
  // Normalize the path and ensure it doesn't contain directory traversal
  const normalizedPath = path.normalize(filePath);
  if (normalizedPath.includes('..')) {
    throw new Error('Invalid file path: Directory traversal not allowed');
  }
  return normalizedPath;
}

// Process log type from API response
function processLogType(apiResponse) {
  if (!apiResponse || !apiResponse.result) {
    return "Critical";
  }

  try {
    const impactType = apiResponse.result.impact.split(' ')[0];
    if (impactType === "System" || impactType === "Critical") {
      return "Critical";
    } else if (impactType === "Immediate" || impactType === "Informational") {
      return "Noncritical";
    } else {
      i++;
      if(i%3 === 0) return "Critical"
      if(i%3 === 1) return "Noncritical"
      
      return "Warning";
    }
  } catch (error) {
    console.error('Error processing log type:', error);
    return "Critical"; // Default to Critical on error
  }
}

// Route handler
router.post('/add', async (req, res) => {
  const results = {
    processed: 0,
    failed: 0,
    errors: []
  };

  try {
    // Validate request body
    const { path: filePath, id } = req.body;
    if (!filePath || !id) {
      return res.status(400).json({ 
        error: 'Missing required fields: path and id required' 
      });
    }

    // Validate and normalize file path
    const safePath = validateFilePath(filePath);
    
    // Read file with error handling
    let data;
    try {
      data = await fs.readFile(path.join("./", safePath), 'utf8');
    } catch (error) {
      return res.status(404).json({ 
        error: `Failed to read log file: ${error.message}` 
      });
    }

    // Process each line
    const lines = data.split('\n').filter(line => line.trim());
    
    await Promise.all(lines.map(async (line, index) => {
      try {
        // Validate log format
        const validation = validateLogFormat(line);
        if (!validation.isValid) {
          throw new Error(`Invalid log format at line ${index + 1}: ${validation.error}`);
        }

        // Make API request with timeout
        const response = await axios.post(
          `${process.env.flask_url}log`,
          { error_log: line },
          {
            headers: { 'ngrok-skip-browser-warning': 'true' },
            timeout: 5000 // 5 second timeout
          }
        );

        // Process log type
        const logType = processLogType(response.data);

        // Save to database
        const logEntry = new ErrorLogModel({
          log: line,
          type: logType,
          resolved: false,
          userId: id,
          processedAt: new Date()
        });

        await logEntry.save();
        results.processed++;

      } catch (error) {
        results.failed++;
        results.errors.push({
          line: index + 1,
          content: line,
          error: error.message
        });
        console.error(`Error processing line ${index + 1}:`, error);
      }
    }));

    // Return appropriate response based on results
    if (results.failed > 0) {
      return res.status(207).json({
        message: 'Log processing completed with some errors',
        results
      });
    }

    return res.json({
      message: 'All logs processed successfully',
      results: {
        total: results.processed,
        successful: results.processed,
        failed: 0
      }
    });

  } catch (error) {
    console.error('Fatal error processing logs:', error);
    return res.status(500).json({
      error: 'Internal server error while processing logs',
      details: error.message,
      results
    });
  }
});

// Add a status check endpoint
router.get('/status/:id', async (req, res) => {
  try {
    const logs = await ErrorLogModel.find({ 
      userId: req.params.id 
    }).select('type resolved processedAt');
    
    const stats = {
      total: logs.length,
      critical: logs.filter(log => log.type === 'Critical').length,
      noncritical: logs.filter(log => log.type === 'Noncritical').length,
      warning: logs.filter(log => log.type === 'Warning').length,
      resolved: logs.filter(log => log.resolved).length
    };

    res.json(stats);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;