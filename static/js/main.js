// Main JavaScript for Exoplanet Detection Platform
// Modern, minimalist interactions and functionality

// Global variables
let currentFile = null;
let isProcessing = false;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    setupAnimations();
});

// Initialize the application
function initializeApp() {
    // Initialize dark theme
    initializeDarkTheme();
    
    // Add loading states
    addLoadingStates();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Setup responsive navigation
    setupResponsiveNavigation();
    
    // Initialize charts if on statistics page
    if (window.location.pathname.includes('statistics')) {
        initializeCharts();
    }
}

// Setup event listeners
function setupEventListeners() {
    // File upload handlers
    setupFileUpload();
    
    // Form submission handlers
    setupFormHandlers();
    
    // Button click handlers
    setupButtonHandlers();
    
    // Keyboard shortcuts
    setupKeyboardShortcuts();
}

// Setup animations
function setupAnimations() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    document.querySelectorAll('.feature-card, .stat-card, .chart-container').forEach(el => {
        observer.observe(el);
    });
}

// Add loading states to buttons and forms
function addLoadingStates() {
    // This function is now handled by individual event handlers
    // to avoid conflicts with specific button behaviors
}

// Initialize tooltips
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
}

// Show tooltip
function showTooltip(event) {
    const element = event.target;
    const tooltipText = element.getAttribute('data-tooltip');
    
    if (tooltipText) {
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip-content';
        tooltip.textContent = tooltipText;
        tooltip.style.cssText = `
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
            z-index: 1000;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s ease;
        `;
        
        document.body.appendChild(tooltip);
        
        // Position tooltip
        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
        tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
        
        // Fade in
        setTimeout(() => {
            tooltip.style.opacity = '1';
        }, 10);
        
        element.tooltipElement = tooltip;
    }
}

// Hide tooltip
function hideTooltip(event) {
    const element = event.target;
    if (element.tooltipElement) {
        element.tooltipElement.remove();
        element.tooltipElement = null;
    }
}

// Setup responsive navigation
function setupResponsiveNavigation() {
    const nav = document.querySelector('nav');
    if (!nav) return;
    
    // Add mobile menu toggle if needed
    const mobileMenuButton = document.createElement('button');
    mobileMenuButton.className = 'md:hidden p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100';
    mobileMenuButton.innerHTML = '<i class="fas fa-bars"></i>';
    
    // Insert before navigation items
    const navContainer = nav.querySelector('.flex.justify-between');
    if (navContainer) {
        const navItems = navContainer.querySelector('.flex.items-center.space-x-8');
        if (navItems) {
            navContainer.insertBefore(mobileMenuButton, navItems);
        }
    }
}

// Setup file upload functionality
function setupFileUpload() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        input.addEventListener('change', handleFileSelect);
    });
    
    // Drag and drop functionality
    const dropZones = document.querySelectorAll('.drop-zone');
    
    dropZones.forEach(zone => {
        zone.addEventListener('dragover', handleDragOver);
        zone.addEventListener('dragleave', handleDragLeave);
        zone.addEventListener('drop', handleDrop);
    });
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

// Handle drag over
function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
}

// Handle drag leave
function handleDragLeave(event) {
    event.currentTarget.classList.remove('dragover');
}

// Handle drop
function handleDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

// Process uploaded file
function processFile(file) {
    currentFile = file;
    
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showNotification('Please upload a CSV file.', 'error');
        return;
    }
    
    // Show file info
    showFileInfo(file);
    
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

// Setup form handlers
function setupFormHandlers() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', handleFormSubmit);
    });
}

// Handle form submission
function handleFormSubmit(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    
    // Show loading state
    const submitBtn = form.querySelector('button[type="submit"]');
    if (submitBtn) {
        showLoadingState(submitBtn);
    }
    
    // Process form based on action
    if (form.id === 'prediction-form') {
        handlePrediction(formData);
    } else if (form.id === 'batch-form') {
        handleBatchAnalysis(formData);
    }
}

// Setup button handlers
function setupButtonHandlers() {
    // Template buttons
    const templateBtns = document.querySelectorAll('.template-btn');
    templateBtns.forEach(btn => {
        btn.addEventListener('click', handleTemplateSelect);
    });
    
    // Random generation button
    const randomBtn = document.getElementById('generate-random');
    if (randomBtn) {
        randomBtn.addEventListener('click', generateRandomParams);
    }
    
    // Remove file button
    const removeFileBtn = document.getElementById('remove-file');
    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', removeFile);
    }
}

// Handle template selection
function handleTemplateSelect(event) {
    const template = event.currentTarget.getAttribute('data-template');
    
    // Remove active class from all buttons
    document.querySelectorAll('.template-btn').forEach(btn => {
        btn.classList.remove('bg-blue-100', 'border-blue-500');
    });
    
    // Add active class to selected button
    event.currentTarget.classList.add('bg-blue-100', 'border-blue-500');
    
    // Load template data
    loadTemplateData(template);
}

// Load template data
async function loadTemplateData(templateName) {
    try {
        const response = await fetch(`/api/template/${templateName}`);
        const data = await response.json();
        
        if (response.ok) {
            displayTemplateParams(data);
        } else {
            showNotification('Failed to load template data.', 'error');
        }
    } catch (error) {
        console.error('Error loading template:', error);
        showNotification('Error loading template data.', 'error');
    }
}

// Display template parameters
function displayTemplateParams(data) {
    const preview = document.getElementById('template-preview');
    const params = document.getElementById('template-params');
    
    if (preview && params) {
        preview.classList.remove('hidden');
        
        params.innerHTML = '';
        
        Object.entries(data).forEach(([key, value]) => {
            const paramDiv = document.createElement('div');
            paramDiv.className = 'flex justify-between items-center p-2 bg-gray-50 rounded';
            paramDiv.innerHTML = `
                <span class="text-sm font-medium text-gray-700">${formatParameterName(key)}</span>
                <span class="text-sm text-gray-600">${value}</span>
            `;
            params.appendChild(paramDiv);
        });
    }
}

// Format parameter names
function formatParameterName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// Generate random parameters
async function generateRandomParams() {
    try {
        const response = await fetch('/api/random');
        const data = await response.json();
        
        if (response.ok) {
            displayRandomParams(data);
        } else {
            showNotification('Failed to generate random parameters.', 'error');
        }
    } catch (error) {
        console.error('Error generating random params:', error);
        showNotification('Error generating random parameters.', 'error');
    }
}

// Display random parameters
function displayRandomParams(data) {
    const preview = document.getElementById('random-preview');
    const params = document.getElementById('random-params');
    
    if (preview && params) {
        preview.classList.remove('hidden');
        
        params.innerHTML = '';
        
        Object.entries(data).forEach(([key, value]) => {
            const paramDiv = document.createElement('div');
            paramDiv.className = 'flex justify-between items-center p-2 bg-gray-50 rounded';
            paramDiv.innerHTML = `
                <span class="text-sm font-medium text-gray-700">${formatParameterName(key)}</span>
                <span class="text-sm text-gray-600">${value.toFixed(2)}</span>
            `;
            params.appendChild(paramDiv);
        });
    }
}

// Handle prediction
async function handlePrediction(formData) {
    try {
        // Convert form data to JSON
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = parseFloat(value) || value;
        }
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayPredictionResult(result);
        } else {
            showNotification(result.error || 'Prediction failed.', 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification('Error making prediction.', 'error');
    }
}

// Display prediction result
function displayPredictionResult(result) {
    const resultsSection = document.getElementById('results-section');
    const predictionResult = document.getElementById('prediction-result');
    
    if (resultsSection && predictionResult) {
        resultsSection.classList.remove('hidden');
        
        const isExoplanet = result.prediction;
        const confidence = result.confidence;
        const confidenceClass = confidence > 0.8 ? 'confidence-high' : 
                               confidence > 0.6 ? 'confidence-medium' : 'confidence-low';
        
        predictionResult.innerHTML = `
            <div class="prediction-result ${isExoplanet ? 'exoplanet' : 'not-exoplanet'}">
                <div class="text-4xl mb-4">
                    ${isExoplanet ? 'EXOPLANET' : 'NOT EXOPLANET'}
                </div>
                <h3 class="text-2xl font-bold mb-2">
                    ${isExoplanet ? 'Exoplanet Detected!' : 'Not an Exoplanet'}
                </h3>
                <p class="text-lg">
                    Confidence: <span class="${confidenceClass}">${(confidence * 100).toFixed(1)}%</span>
                </p>
            </div>
        `;
        
        // Display base model analysis
        displayBaseModelAnalysis(result);
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// Display base model analysis
function displayBaseModelAnalysis(result) {
    const baseModelDetails = document.getElementById('base-model-details');
    
    if (baseModelDetails) {
        baseModelDetails.innerHTML = `
            <div class="text-center p-4 bg-blue-50 rounded-lg">
                <h4 class="font-semibold text-gray-900 mb-2">XGBoost</h4>
                <p class="text-2xl font-bold text-blue-600">${(result.xgb_probability * 100).toFixed(1)}%</p>
            </div>
            <div class="text-center p-4 bg-green-50 rounded-lg">
                <h4 class="font-semibold text-gray-900 mb-2">Random Forest</h4>
                <p class="text-2xl font-bold text-green-600">${(result.rf_probability * 100).toFixed(1)}%</p>
            </div>
            <div class="text-center p-4 bg-purple-50 rounded-lg">
                <h4 class="font-semibold text-gray-900 mb-2">Agreement</h4>
                <p class="text-2xl font-bold ${result.base_model_agreement ? 'text-green-600' : 'text-red-600'}">
                    ${result.base_model_agreement ? '✓' : '✗'}
                </p>
            </div>
        `;
    }
}

// Handle batch analysis
async function handleBatchAnalysis(formData) {
    if (!currentFile) {
        showNotification('Please select a file first.', 'error');
        return;
    }
    
    try {
        const formData = new FormData();
        formData.append('file', currentFile);
        
        // Show progress modal
        showProgressModal();
        
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
            showNotification(result.error || 'Batch analysis failed.', 'error');
        }
    } catch (error) {
        console.error('Batch analysis error:', error);
        hideProgressModal();
        showNotification('Error processing batch data.', 'error');
    }
}

// Display batch results
function displayBatchResults(result) {
    const resultsSection = document.getElementById('results-section');
    const summaryStats = document.getElementById('summary-stats');
    const resultsTable = document.getElementById('results-table');
    
    if (resultsSection) {
        resultsSection.classList.remove('hidden');
        
        // Display summary statistics
        if (summaryStats) {
            const exoplanets = result.results.filter(r => r.prediction).length;
            const avgConfidence = result.results.reduce((sum, r) => sum + r.confidence, 0) / result.results.length;
            const agreements = result.results.filter(r => r.base_model_agreement).length;
            
            summaryStats.innerHTML = `
                <div class="text-center p-4 bg-blue-50 rounded-lg">
                    <h4 class="font-semibold text-gray-900 mb-2">Exoplanets Detected</h4>
                    <p class="text-2xl font-bold text-blue-600">${exoplanets}</p>
                </div>
                <div class="text-center p-4 bg-green-50 rounded-lg">
                    <h4 class="font-semibold text-gray-900 mb-2">Average Confidence</h4>
                    <p class="text-2xl font-bold text-green-600">${(avgConfidence * 100).toFixed(1)}%</p>
                </div>
                <div class="text-center p-4 bg-purple-50 rounded-lg">
                    <h4 class="font-semibold text-gray-900 mb-2">Model Agreements</h4>
                    <p class="text-2xl font-bold text-purple-600">${agreements}/${result.total_samples}</p>
                </div>
                <div class="text-center p-4 bg-orange-50 rounded-lg">
                    <h4 class="font-semibold text-gray-900 mb-2">Agreement Rate</h4>
                    <p class="text-2xl font-bold text-orange-600">${((agreements / result.total_samples) * 100).toFixed(1)}%</p>
                </div>
            `;
        }
        
        // Display results table
        if (resultsTable) {
            displayResultsTable(result.results);
        }
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// Display results table
function displayResultsTable(results) {
    const resultsTable = document.getElementById('results-table');
    
    if (resultsTable) {
        let tableHTML = `
            <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                    <tr>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Index</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Prediction</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">XGBoost</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Random Forest</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Agreement</th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
        `;
        
        results.forEach(result => {
            const predictionClass = result.prediction ? 'text-green-600' : 'text-red-600';
            const agreementClass = result.base_model_agreement ? 'text-green-600' : 'text-red-600';
            
            tableHTML += `
                <tr>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${result.index}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm ${predictionClass}">
                        ${result.prediction ? 'Exoplanet' : 'Not Exoplanet'}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${(result.confidence * 100).toFixed(1)}%</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${(result.xgb_probability * 100).toFixed(1)}%</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${(result.rf_probability * 100).toFixed(1)}%</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm ${agreementClass}">
                        ${result.base_model_agreement ? '✓' : '✗'}
                    </td>
                </tr>
            `;
        });
        
        tableHTML += `
                </tbody>
            </table>
        `;
        
        resultsTable.innerHTML = tableHTML;
    }
}

// Show progress modal
function showProgressModal() {
    const modal = document.getElementById('progress-modal');
    if (modal) {
        modal.classList.remove('hidden');
        modal.classList.add('flex');
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

// Show loading state
function showLoadingState(button) {
    button.disabled = true;
    button.innerHTML = '<span class="spinner mr-2"></span>Processing...';
}

// Remove file
function removeFile() {
    currentFile = null;
    
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    if (fileInput) fileInput.value = '';
    if (fileInfo) fileInfo.classList.add('hidden');
    if (analyzeBtn) analyzeBtn.disabled = true;
}

// Setup keyboard shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // Ctrl/Cmd + Enter for form submission
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            const activeForm = document.querySelector('form:not([style*="display: none"])');
            if (activeForm) {
                activeForm.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to close modals
        if (event.key === 'Escape') {
            hideProgressModal();
        }
    });
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${
        type === 'error' ? 'bg-red-100 text-red-800 border border-red-200' :
        type === 'success' ? 'bg-green-100 text-green-800 border border-green-200' :
        'bg-blue-100 text-blue-800 border border-blue-200'
    }`;
    
    notification.innerHTML = `
        <div class="flex items-center space-x-2">
            <i class="fas ${
                type === 'error' ? 'fa-exclamation-circle' :
                type === 'success' ? 'fa-check-circle' :
                'fa-info-circle'
            }"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-2 text-gray-500 hover:text-gray-700">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Initialize charts (placeholder for statistics page)
function initializeCharts() {
    // Chart initialization will be handled by statistics.js
    console.log('Charts will be initialized by statistics.js');
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Dark theme functionality
function initializeDarkTheme() {
    console.log('Initializing dark theme...');
    
    // Check for saved theme preference or default to light mode
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    console.log('Saved theme:', savedTheme);
    console.log('Prefers dark:', prefersDark);
    
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        enableDarkTheme();
    } else {
        enableLightTheme();
    }
    
    // Setup theme toggle button
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        console.log('Theme toggle button found, adding event listener');
        themeToggle.addEventListener('click', toggleTheme);
    } else {
        console.error('Theme toggle button not found!');
    }
    
    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!localStorage.getItem('theme')) {
            if (e.matches) {
                enableDarkTheme();
            } else {
                enableLightTheme();
            }
        }
    });
}

function toggleTheme() {
    console.log('Toggle theme clicked');
    const isDark = document.documentElement.classList.contains('dark');
    console.log('Currently dark mode:', isDark);
    
    if (isDark) {
        enableLightTheme();
        localStorage.setItem('theme', 'light');
        console.log('Switched to light theme');
    } else {
        enableDarkTheme();
        localStorage.setItem('theme', 'dark');
        console.log('Switched to dark theme');
    }
}

function enableDarkTheme() {
    console.log('Enabling dark theme');
    document.documentElement.classList.add('dark');
    document.body.classList.add('dark');
    updateThemeIcon('light');
    
    // Update any existing charts to dark theme
    if (typeof updateChartsTheme === 'function') {
        updateChartsTheme('dark');
    }
}

function enableLightTheme() {
    console.log('Enabling light theme');
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
    updateThemeIcon('dark');
    
    // Update any existing charts to light theme
    if (typeof updateChartsTheme === 'function') {
        updateChartsTheme('light');
    }
}

function updateThemeIcon(theme) {
    const themeIcon = document.getElementById('theme-icon');
    const themeIconMobile = document.getElementById('theme-icon-mobile');
    
    const icons = [themeIcon, themeIconMobile].filter(icon => icon);
    
    if (icons.length > 0) {
        icons.forEach(icon => {
            if (theme === 'dark') {
                icon.className = 'fas fa-moon text-gray-700 dark:text-white text-sm';
            } else {
                icon.className = 'fas fa-sun text-gray-700 dark:text-white text-sm';
            }
        });
        console.log('Updated theme icons to:', theme === 'dark' ? 'moon' : 'sun');
    } else {
        console.error('Theme icons not found!');
    }
}

// Update notification function to support dark theme
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 transition-colors duration-300 ${
        type === 'error' ? 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 border border-red-200 dark:border-red-700' :
        type === 'success' ? 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 border border-green-200 dark:border-green-700' :
        'bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 border border-blue-200 dark:border-blue-700'
    }`;
    
    notification.innerHTML = `
        <div class="flex items-center space-x-2">
            <i class="fas ${
                type === 'error' ? 'fa-exclamation-circle' :
                type === 'success' ? 'fa-check-circle' :
                'fa-info-circle'
            }"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Export functions for use in other scripts
window.ExoplanetApp = {
    showNotification,
    displayPredictionResult,
    displayBatchResults,
    showLoadingState,
    hideProgressModal,
    toggleTheme,
    enableDarkTheme,
    enableLightTheme
};
