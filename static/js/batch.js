// Batch Analysis Page JavaScript
// Handles batch exoplanet analysis functionality

document.addEventListener('DOMContentLoaded', function() {
    initializeBatchPage();
});

// Initialize batch page
function initializeBatchPage() {
    setupFileUpload();
    setupBatchAnalysis();
    setupDownloadHandler();
}

// Setup file upload functionality
function setupFileUpload() {
    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Drop zone click handler
    if (dropZone) {
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
        
        // Drag and drop handlers
        dropZone.addEventListener('dragover', handleDragOver);
        dropZone.addEventListener('dragleave', handleDragLeave);
        dropZone.addEventListener('drop', handleDrop);
    }
    
    // Analyze button handler
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', handleBatchAnalysis);
    }
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processUploadedFile(file);
    }
}

// Handle drag over
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.add('dragover');
}

// Handle drag leave
function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('dragover');
}

// Handle drop
function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processUploadedFile(files[0]);
    }
}

// Process uploaded file
function processUploadedFile(file) {
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        ExoplanetApp.showNotification('Please upload a CSV file.', 'error');
        return;
    }
    
    // Validate file size (16MB limit)
    const maxSize = 16 * 1024 * 1024; // 16MB
    if (file.size > maxSize) {
        ExoplanetApp.showNotification('File size too large. Please upload a file smaller than 16MB.', 'error');
        return;
    }
    
    // Store current file
    window.currentBatchFile = file;
    
    // Show file info
    showFileInfo(file);
    
    // Preview file content
    previewFileContent(file);
    
    // Enable analyze button
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = false;
    }
}

// Show file information
function showFileInfo(file) {
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    
    if (fileInfo && fileName) {
        fileName.textContent = file.name;
        fileInfo.classList.remove('hidden');
    }
}

// Preview file content
function previewFileContent(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const csv = e.target.result;
        const lines = csv.split('\n');
        const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
        
        // Check required columns
        const requiredColumns = [
            'orbital_period', 'planetary_radius', 'transit_depth', 'transit_duration',
            'impact_parameter', 'stellar_temperature', 'stellar_radius', 'stellar_gravity',
            'equilibrium_temp', 'insolation', 'planetary_mass', 'semi_major_axis',
            'orbital_eccentricity', 'stellar_mass', 'stellar_metallicity', 'system_distance'
        ];
        
        const missingColumns = requiredColumns.filter(col => !headers.includes(col));
        
        if (missingColumns.length > 0) {
            ExoplanetApp.showNotification(
                `Missing required columns: ${missingColumns.join(', ')}`, 
                'warning'
            );
        } else {
            ExoplanetApp.showNotification('All required columns found!', 'success');
        }
        
        // Display sample data preview
        displaySampleData(lines, headers);
    };
    
    reader.readAsText(file);
}

// Display sample data
function displaySampleData(lines, headers) {
    const samplePreview = document.getElementById('sample-preview');
    const sampleTable = document.getElementById('sample-table');
    
    if (samplePreview && sampleTable) {
        samplePreview.classList.remove('hidden');
        
        // Create table HTML
        let tableHTML = `
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            ${headers.map(header => 
                                `<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                    ${header}
                                </th>`
                            ).join('')}
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
        `;
        
        // Add sample rows (first 5 rows)
        for (let i = 1; i <= Math.min(5, lines.length - 1); i++) {
            if (lines[i].trim()) {
                const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''));
                tableHTML += '<tr>';
                values.forEach(value => {
                    tableHTML += `<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${value}</td>`;
                });
                tableHTML += '</tr>';
            }
        }
        
        tableHTML += `
                    </tbody>
                </table>
            </div>
            <div class="mt-4 text-sm text-gray-600">
                Showing first 5 rows of ${lines.length - 1} total rows
            </div>
        `;
        
        sampleTable.innerHTML = tableHTML;
    }
}

// Handle batch analysis
async function handleBatchAnalysis() {
    if (!window.currentBatchFile) {
        ExoplanetApp.showNotification('Please select a file first.', 'error');
        return;
    }
    
    try {
        // Show progress modal
        showProgressModal();
        
        // Create form data
        const formData = new FormData();
        formData.append('file', window.currentBatchFile);
        
        // Make API request
        const response = await fetch('/api/batch', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // Hide progress modal
        hideProgressModal();
        
        if (response.ok) {
            displayBatchResults(result);
        } else {
            ExoplanetApp.showNotification(result.error || 'Batch analysis failed.', 'error');
        }
    } catch (error) {
        console.error('Batch analysis error:', error);
        hideProgressModal();
        ExoplanetApp.showNotification('Error processing batch data.', 'error');
    }
}

// Show progress modal
function showProgressModal() {
    const modal = document.getElementById('progress-modal');
    if (modal) {
        modal.classList.remove('hidden');
        modal.classList.add('flex');
        
        // Start progress animation
        animateProgress();
    }
}

// Hide progress modal
function hideProgressModal() {
    const modal = document.getElementById('progress-modal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }
}

// Animate progress bar
function animateProgress() {
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    if (progressBar && progressText) {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 90) progress = 90;
            
            progressBar.style.width = progress + '%';
            progressText.textContent = Math.round(progress) + '% complete';
            
            if (progress >= 90) {
                clearInterval(interval);
            }
        }, 200);
        
        // Store interval for cleanup
        window.progressInterval = interval;
    }
}

// Display batch results
function displayBatchResults(result) {
    const resultsSection = document.getElementById('results-section');
    const summaryStats = document.getElementById('summary-stats');
    const resultsTable = document.getElementById('results-table');
    
    if (resultsSection) {
        resultsSection.classList.remove('hidden');
        
        // Calculate summary statistics
        const exoplanets = result.results.filter(r => r.prediction && !r.error).length;
        const validResults = result.results.filter(r => !r.error);
        const avgConfidence = validResults.length > 0 ? 
            validResults.reduce((sum, r) => sum + r.confidence, 0) / validResults.length : 0;
        const agreements = validResults.filter(r => r.base_model_agreement).length;
        const errors = result.results.filter(r => r.error).length;
        
        // Display summary statistics
        if (summaryStats) {
            summaryStats.innerHTML = `
                <div class="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <div class="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                        <i class="fas fa-globe text-blue-600 text-xl"></i>
                    </div>
                    <h4 class="font-semibold text-gray-900 mb-2">Exoplanets Detected</h4>
                    <p class="text-2xl font-bold text-blue-600">${exoplanets}</p>
                    <p class="text-sm text-gray-600">out of ${validResults.length}</p>
                </div>
                <div class="text-center p-4 bg-green-50 rounded-lg border border-green-200">
                    <div class="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                        <i class="fas fa-chart-line text-green-600 text-xl"></i>
                    </div>
                    <h4 class="font-semibold text-gray-900 mb-2">Average Confidence</h4>
                    <p class="text-2xl font-bold text-green-600">${(avgConfidence * 100).toFixed(1)}%</p>
                    <p class="text-sm text-gray-600">Overall confidence</p>
                </div>
                <div class="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
                    <div class="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
                        <i class="fas fa-handshake text-purple-600 text-xl"></i>
                    </div>
                    <h4 class="font-semibold text-gray-900 mb-2">Model Agreements</h4>
                    <p class="text-2xl font-bold text-purple-600">${agreements}</p>
                    <p class="text-sm text-gray-600">out of ${validResults.length}</p>
                </div>
                <div class="text-center p-4 bg-orange-50 rounded-lg border border-orange-200">
                    <div class="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-3">
                        <i class="fas fa-exclamation-triangle text-orange-600 text-xl"></i>
                    </div>
                    <h4 class="font-semibold text-gray-900 mb-2">Processing Errors</h4>
                    <p class="text-2xl font-bold text-orange-600">${errors}</p>
                    <p class="text-sm text-gray-600">${errors > 0 ? 'Check results table' : 'All processed'}</p>
                </div>
            `;
        }
        
        // Display results table
        if (resultsTable) {
            displayResultsTable(result.results);
        }
        
        // Store results for download
        window.batchResults = result.results;
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// Display results table
function displayResultsTable(results) {
    const resultsTable = document.getElementById('results-table');
    
    if (resultsTable) {
        let tableHTML = `
            <div class="overflow-x-auto shadow-sm rounded-lg border border-gray-200">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Index</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Prediction</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">XGBoost</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Random Forest</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Agreement</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
        `;
        
        results.forEach(result => {
            if (result.error) {
                // Error row
                tableHTML += `
                    <tr class="bg-red-50">
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${result.index}</td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500" colspan="5">Error</td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-red-600">
                            <i class="fas fa-exclamation-circle mr-1"></i>
                            ${result.error}
                        </td>
                    </tr>
                `;
            } else {
                // Success row
                const predictionClass = result.prediction ? 'text-green-600' : 'text-red-600';
                const agreementClass = result.base_model_agreement ? 'text-green-600' : 'text-red-600';
                
                tableHTML += `
                    <tr class="hover:bg-gray-50">
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${result.index}</td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm ${predictionClass}">
                            <div class="flex items-center">
                                <i class="fas ${result.prediction ? 'fa-globe' : 'fa-times-circle'} mr-2"></i>
                                ${result.prediction ? 'Exoplanet' : 'Not Exoplanet'}
                            </div>
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            <div class="flex items-center">
                                <div class="w-16 bg-gray-200 rounded-full h-2 mr-2">
                                    <div class="bg-blue-600 h-2 rounded-full" style="width: ${result.confidence * 100}%"></div>
                                </div>
                                ${(result.confidence * 100).toFixed(1)}%
                            </div>
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${(result.xgb_probability * 100).toFixed(1)}%</td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${(result.rf_probability * 100).toFixed(1)}%</td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm ${agreementClass}">
                            <i class="fas ${result.base_model_agreement ? 'fa-check-circle' : 'fa-times-circle'} mr-1"></i>
                            ${result.base_model_agreement ? 'Yes' : 'No'}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-green-600">
                            <i class="fas fa-check-circle mr-1"></i>
                            Success
                        </td>
                    </tr>
                `;
            }
        });
        
        tableHTML += `
                    </tbody>
                </table>
            </div>
        `;
        
        resultsTable.innerHTML = tableHTML;
    }
}

// Setup download handler
function setupDownloadHandler() {
    const downloadBtn = document.getElementById('download-results');
    
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadResults);
    }
}

// Download results as CSV
function downloadResults() {
    if (!window.batchResults) {
        ExoplanetApp.showNotification('No results to download.', 'error');
        return;
    }
    
    try {
        // Prepare CSV data
        const headers = [
            'Index', 'Prediction', 'Confidence', 'XGBoost_Probability', 
            'Random_Forest_Probability', 'Base_Model_Agreement', 'Status'
        ];
        
        const csvRows = [headers.join(',')];
        
        window.batchResults.forEach(result => {
            if (result.error) {
                csvRows.push([
                    result.index,
                    'Error',
                    '',
                    '',
                    '',
                    '',
                    result.error
                ].join(','));
            } else {
                csvRows.push([
                    result.index,
                    result.prediction ? 'Exoplanet' : 'Not Exoplanet',
                    (result.confidence * 100).toFixed(2),
                    (result.xgb_probability * 100).toFixed(2),
                    (result.rf_probability * 100).toFixed(2),
                    result.base_model_agreement ? 'Yes' : 'No',
                    'Success'
                ].join(','));
            }
        });
        
        // Create and download file
        const csvContent = csvRows.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        
        if (link.download !== undefined) {
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', `exoplanet_predictions_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            ExoplanetApp.showNotification('Results downloaded successfully!', 'success');
        } else {
            ExoplanetApp.showNotification('Download not supported in this browser.', 'error');
        }
    } catch (error) {
        console.error('Download error:', error);
        ExoplanetApp.showNotification('Error downloading results.', 'error');
    }
}

// Remove file
function removeFile() {
    window.currentBatchFile = null;
    window.batchResults = null;
    
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const analyzeBtn = document.getElementById('analyze-btn');
    const samplePreview = document.getElementById('sample-preview');
    const resultsSection = document.getElementById('results-section');
    
    if (fileInput) fileInput.value = '';
    if (fileInfo) fileInfo.classList.add('hidden');
    if (analyzeBtn) analyzeBtn.disabled = true;
    if (samplePreview) samplePreview.classList.add('hidden');
    if (resultsSection) resultsSection.classList.add('hidden');
}

// Setup remove file button
document.addEventListener('DOMContentLoaded', function() {
    const removeFileBtn = document.getElementById('remove-file');
    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', removeFile);
    }
});

// Export functions for global access
window.BatchPage = {
    removeFile,
    downloadResults,
    displayBatchResults
};
