// Prediction Page JavaScript
// Handles single exoplanet prediction functionality

document.addEventListener('DOMContentLoaded', function() {
    initializePredictionPage();
});

// Initialize prediction page
function initializePredictionPage() {
    setupInputMethodButtons();
    setupTemplateHandlers();
    setupRandomHandlers();
    setupFormHandlers();
}

// Setup input method buttons
function setupInputMethodButtons() {
    const methodButtons = document.querySelectorAll('.input-method-btn');
    
    methodButtons.forEach(button => {
        button.addEventListener('click', function() {
            const method = this.getAttribute('data-method');
            switchInputMethod(method);
        });
    });
}

// Switch input method
function switchInputMethod(method) {
    // Remove active class from all buttons
    document.querySelectorAll('.input-method-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Add active class to clicked button
    document.querySelector(`[data-method="${method}"]`).classList.add('active');
    
    // Hide all forms
    document.querySelectorAll('.input-form').forEach(form => {
        form.classList.add('hidden');
    });
    
    // Show selected form
    const targetForm = document.getElementById(`${method}-form`);
    if (targetForm) {
        targetForm.classList.remove('hidden');
    }
    
    // Reset results section
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsSection.classList.add('hidden');
    }
}

// Setup template handlers
function setupTemplateHandlers() {
    const templateButtons = document.querySelectorAll('.template-btn');
    const predictTemplateBtn = document.getElementById('predict-template');
    
    templateButtons.forEach(button => {
        button.addEventListener('click', function() {
            const template = this.getAttribute('data-template');
            selectTemplate(template);
        });
    });
    
    if (predictTemplateBtn) {
        predictTemplateBtn.addEventListener('click', handleTemplatePrediction);
    }
}

// Select template
function selectTemplate(templateName) {
    // Remove active class from all template buttons
    document.querySelectorAll('.template-btn').forEach(btn => {
        btn.classList.remove('bg-blue-100', 'border-blue-500');
    });
    
    // Add active class to selected button
    document.querySelector(`[data-template="${templateName}"]`).classList.add('bg-blue-100', 'border-blue-500');
    
    // Load and display template data
    loadTemplateData(templateName);
}

// Load template data
async function loadTemplateData(templateName) {
    try {
        const response = await fetch(`/api/template/${templateName}`);
        const data = await response.json();
        
        if (response.ok) {
            displayTemplatePreview(data);
            // Store template data for prediction
            window.currentTemplateData = data;
        } else {
            ExoplanetApp.showNotification('Failed to load template data.', 'error');
        }
    } catch (error) {
        console.error('Error loading template:', error);
        ExoplanetApp.showNotification('Error loading template data.', 'error');
    }
}

// Display template preview
function displayTemplatePreview(data) {
    const preview = document.getElementById('template-preview');
    const params = document.getElementById('template-params');
    
    if (preview && params) {
        preview.classList.remove('hidden');
        
        params.innerHTML = '';
        
        // Group parameters into two columns
        const entries = Object.entries(data);
        const midPoint = Math.ceil(entries.length / 2);
        
        const leftColumn = entries.slice(0, midPoint);
        const rightColumn = entries.slice(midPoint);
        
        const leftDiv = document.createElement('div');
        const rightDiv = document.createElement('div');
        
        leftDiv.className = 'space-y-2';
        rightDiv.className = 'space-y-2';
        
        leftColumn.forEach(([key, value]) => {
            const paramDiv = document.createElement('div');
            paramDiv.className = 'flex justify-between items-center p-2 bg-gray-50 rounded text-sm';
            paramDiv.innerHTML = `
                <span class="font-medium text-gray-700">${formatParameterName(key)}</span>
                <span class="text-gray-600">${value}</span>
            `;
            leftDiv.appendChild(paramDiv);
        });
        
        rightColumn.forEach(([key, value]) => {
            const paramDiv = document.createElement('div');
            paramDiv.className = 'flex justify-between items-center p-2 bg-gray-50 rounded text-sm';
            paramDiv.innerHTML = `
                <span class="font-medium text-gray-700">${formatParameterName(key)}</span>
                <span class="text-gray-600">${value}</span>
            `;
            rightDiv.appendChild(paramDiv);
        });
        
        params.appendChild(leftDiv);
        params.appendChild(rightDiv);
    }
}

// Setup random handlers
function setupRandomHandlers() {
    const generateBtn = document.getElementById('generate-random');
    const predictRandomBtn = document.getElementById('predict-random');
    
    if (generateBtn) {
        generateBtn.addEventListener('click', generateRandomParameters);
    }
    
    if (predictRandomBtn) {
        predictRandomBtn.addEventListener('click', handleRandomPrediction);
    }
}

// Generate random parameters
async function generateRandomParameters() {
    const generateBtn = document.getElementById('generate-random');
    const originalText = generateBtn.innerHTML;
    
    try {
        // Show loading state
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<span class="spinner mr-2"></span>Processing...';
        
        const response = await fetch('/api/random');
        const data = await response.json();
        
        if (response.ok) {
            displayRandomPreview(data);
            // Store random data for prediction
            window.currentRandomData = data;
            ExoplanetApp.showNotification('Random parameters generated successfully!', 'success');
        } else {
            ExoplanetApp.showNotification('Failed to generate random parameters.', 'error');
        }
    } catch (error) {
        console.error('Error generating random params:', error);
        ExoplanetApp.showNotification('Error generating random parameters.', 'error');
    } finally {
        // Reset button state
        generateBtn.disabled = false;
        generateBtn.innerHTML = originalText;
    }
}

// Display random preview
function displayRandomPreview(data) {
    const preview = document.getElementById('random-preview');
    const params = document.getElementById('random-params');
    
    if (preview && params) {
        preview.classList.remove('hidden');
        
        params.innerHTML = '';
        
        // Group parameters into two columns
        const entries = Object.entries(data);
        const midPoint = Math.ceil(entries.length / 2);
        
        const leftColumn = entries.slice(0, midPoint);
        const rightColumn = entries.slice(midPoint);
        
        const leftDiv = document.createElement('div');
        const rightDiv = document.createElement('div');
        
        leftDiv.className = 'space-y-2';
        rightDiv.className = 'space-y-2';
        
        leftColumn.forEach(([key, value]) => {
            const paramDiv = document.createElement('div');
            paramDiv.className = 'flex justify-between items-center p-2 bg-gray-50 rounded text-sm';
            paramDiv.innerHTML = `
                <span class="font-medium text-gray-700">${formatParameterName(key)}</span>
                <span class="text-gray-600">${value.toFixed(2)}</span>
            `;
            leftDiv.appendChild(paramDiv);
        });
        
        rightColumn.forEach(([key, value]) => {
            const paramDiv = document.createElement('div');
            paramDiv.className = 'flex justify-between items-center p-2 bg-gray-50 rounded text-sm';
            paramDiv.innerHTML = `
                <span class="font-medium text-gray-700">${formatParameterName(key)}</span>
                <span class="text-gray-600">${value.toFixed(2)}</span>
            `;
            rightDiv.appendChild(paramDiv);
        });
        
        params.appendChild(leftDiv);
        params.appendChild(rightDiv);
    }
}

// Setup form handlers
function setupFormHandlers() {
    const predictionForm = document.getElementById('prediction-form');
    
    if (predictionForm) {
        predictionForm.addEventListener('submit', handleManualPrediction);
    }
}

// Handle manual prediction
async function handleManualPrediction(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const data = {};
    
    // Convert form data to object
    for (let [key, value] of formData.entries()) {
        data[key] = parseFloat(value) || value;
    }
    
    // Calculate derived parameters
    calculateDerivedParameters(data);
    
    // Make prediction
    await makePrediction(data, 'predict-manual');
}

// Handle template prediction
async function handleTemplatePrediction() {
    if (!window.currentTemplateData) {
        ExoplanetApp.showNotification('Please select a template first.', 'error');
        return;
    }
    
    // Calculate derived parameters
    const data = { ...window.currentTemplateData };
    calculateDerivedParameters(data);
    
    // Make prediction
    await makePrediction(data, 'predict-template');
}

// Handle random prediction
async function handleRandomPrediction() {
    if (!window.currentRandomData) {
        ExoplanetApp.showNotification('Please generate random parameters first.', 'error');
        return;
    }
    
    // Calculate derived parameters
    const data = { ...window.currentRandomData };
    calculateDerivedParameters(data);
    
    // Make prediction
    await makePrediction(data, 'predict-random');
}

// Calculate derived parameters
function calculateDerivedParameters(data) {
    // Planetary density
    if (data.planetary_radius > 0) {
        data.planetary_density = data.planetary_mass / (data.planetary_radius ** 3);
    } else {
        data.planetary_density = 0;
    }
    
    // Transit depth radius ratio
    if (data.planetary_radius > 0) {
        data.transit_depth_radius_ratio = data.transit_depth / (data.planetary_radius ** 2);
    } else {
        data.transit_depth_radius_ratio = 0;
    }
    
    // Orbital velocity
    if (data.orbital_period > 0) {
        data.orbital_velocity = (2 * Math.PI * data.semi_major_axis) / data.orbital_period;
    } else {
        data.orbital_velocity = 0;
    }
    
    // Stellar luminosity proxy
    data.stellar_luminosity_proxy = ((data.stellar_temperature / 5778) ** 4) * (data.stellar_radius ** 2);
}

// Make prediction
async function makePrediction(data, buttonId = null) {
    let predictButton = null;
    let originalText = '';
    
    try {
        // Find the specific button that was clicked
        if (buttonId) {
            predictButton = document.getElementById(buttonId);
        } else {
            predictButton = document.querySelector('.btn-primary:not([disabled])');
        }
        
        if (predictButton) {
            originalText = predictButton.innerHTML;
            predictButton.disabled = true;
            predictButton.innerHTML = '<span class="spinner mr-2"></span>Processing...';
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
            ExoplanetApp.showNotification('Prediction completed successfully!', 'success');
        } else {
            ExoplanetApp.showNotification(result.error || 'Prediction failed.', 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        ExoplanetApp.showNotification('Error making prediction.', 'error');
    } finally {
        // Reset button state
        if (predictButton) {
            predictButton.disabled = false;
            predictButton.innerHTML = originalText;
        }
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
        
        // Determine confidence level and styling
        let confidenceClass, confidenceText;
        if (confidence > 0.8) {
            confidenceClass = 'confidence-high';
            confidenceText = 'High';
        } else if (confidence > 0.6) {
            confidenceClass = 'confidence-medium';
            confidenceText = 'Medium';
        } else {
            confidenceClass = 'confidence-low';
            confidenceText = 'Low';
        }
        
        predictionResult.innerHTML = `
            <div class="prediction-result ${isExoplanet ? 'exoplanet' : 'not-exoplanet'}">
                <div class="text-6xl mb-4">
                    ${isExoplanet ? 'EXOPLANET' : 'NOT EXOPLANET'}
                </div>
                <h3 class="text-3xl font-bold mb-2 text-gray-900 dark:text-white transition-colors duration-300">
                    ${isExoplanet ? 'Exoplanet Detected!' : 'Not an Exoplanet'}
                </h3>
                <p class="text-xl mb-2 text-gray-700 dark:text-gray-300 transition-colors duration-300">
                    Confidence: <span class="${confidenceClass}">${(confidence * 100).toFixed(1)}%</span>
                </p>
                <p class="text-sm text-gray-600 dark:text-gray-400 transition-colors duration-300">
                    ${confidenceText} confidence prediction
                </p>
            </div>
        `;
        
        // Display base model analysis
        displayBaseModelAnalysis(result);
        
        // Display architecture explanation
        displayArchitectureExplanation(result);
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// Display base model analysis
function displayBaseModelAnalysis(result) {
    const baseModelDetails = document.getElementById('base-model-details');
    
    if (baseModelDetails) {
        baseModelDetails.innerHTML = `
            <div class="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800 transition-colors duration-300">
                <div class="w-12 h-12 bg-blue-100 dark:bg-blue-800 rounded-full flex items-center justify-center mx-auto mb-3 transition-colors duration-300">
                    <i class="fas fa-tree text-blue-600 dark:text-blue-400 text-xl transition-colors duration-300"></i>
                </div>
                <h4 class="font-semibold text-gray-900 dark:text-white mb-2 transition-colors duration-300">XGBoost</h4>
                <p class="text-2xl font-bold text-blue-600 dark:text-blue-400 transition-colors duration-300">${(result.xgb_probability * 100).toFixed(1)}%</p>
                <p class="text-sm text-gray-600 dark:text-gray-300 transition-colors duration-300">Probability</p>
            </div>
            <div class="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800 transition-colors duration-300">
                <div class="w-12 h-12 bg-green-100 dark:bg-green-800 rounded-full flex items-center justify-center mx-auto mb-3 transition-colors duration-300">
                    <i class="fas fa-tree text-green-600 dark:text-green-400 text-xl transition-colors duration-300"></i>
                </div>
                <h4 class="font-semibold text-gray-900 dark:text-white mb-2 transition-colors duration-300">Random Forest</h4>
                <p class="text-2xl font-bold text-green-600 dark:text-green-400 transition-colors duration-300">${(result.rf_probability * 100).toFixed(1)}%</p>
                <p class="text-sm text-gray-600 dark:text-gray-300 transition-colors duration-300">Probability</p>
            </div>
            <div class="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800 transition-colors duration-300">
                <div class="w-12 h-12 bg-purple-100 dark:bg-purple-800 rounded-full flex items-center justify-center mx-auto mb-3 transition-colors duration-300">
                    <i class="fas fa-handshake text-purple-600 dark:text-purple-400 text-xl transition-colors duration-300"></i>
                </div>
                <h4 class="font-semibold text-gray-900 dark:text-white mb-2 transition-colors duration-300">Agreement</h4>
                <p class="text-2xl font-bold ${result.base_model_agreement ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'} transition-colors duration-300">
                    ${result.base_model_agreement ? '✓' : '✗'}
                </p>
                <p class="text-sm text-gray-600 dark:text-gray-300 transition-colors duration-300">
                    ${result.base_model_agreement ? 'Models agree' : 'Models disagree'}
                </p>
            </div>
        `;
    }
}

// Display architecture explanation
function displayArchitectureExplanation(result) {
    const explanation = document.getElementById('architecture-explanation');
    
    if (explanation) {
        const agreementStatus = result.base_model_agreement ? 
            'Base models agree - prediction is more reliable' : 
            'Base models disagree - neural network must decide between conflicting predictions';
        
        const agreementClass = result.base_model_agreement ? 'success' : 'warning';
        
        explanation.innerHTML = `
            <div class="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6 transition-colors duration-300">
                <h4 class="text-lg font-semibold text-gray-900 dark:text-white mb-4 transition-colors duration-300">Architecture Flow</h4>
                <div class="space-y-3 text-sm text-gray-700 dark:text-gray-300 transition-colors duration-300">
                    <div class="flex items-center space-x-4">
                        <span class="bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 px-3 py-1 rounded-full font-medium transition-colors duration-300">Raw Features</span>
                        <i class="fas fa-arrow-right text-blue-500 dark:text-blue-400 transition-colors duration-300"></i>
                        <span class="bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 px-3 py-1 rounded-full font-medium transition-colors duration-300">XGBoost</span>
                        <i class="fas fa-arrow-right text-blue-500 dark:text-blue-400 transition-colors duration-300"></i>
                        <span class="bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 px-3 py-1 rounded-full font-medium transition-colors duration-300">Probability</span>
                    </div>
                    <div class="flex items-center space-x-4">
                        <span class="bg-green-100 dark:bg-green-800 text-green-800 dark:text-green-200 px-3 py-1 rounded-full font-medium transition-colors duration-300">Raw Features</span>
                        <i class="fas fa-arrow-right text-green-500 dark:text-green-400 transition-colors duration-300"></i>
                        <span class="bg-green-100 dark:bg-green-800 text-green-800 dark:text-green-200 px-3 py-1 rounded-full font-medium transition-colors duration-300">Random Forest</span>
                        <i class="fas fa-arrow-right text-green-500 dark:text-green-400 transition-colors duration-300"></i>
                        <span class="bg-green-100 dark:bg-green-800 text-green-800 dark:text-green-200 px-3 py-1 rounded-full font-medium transition-colors duration-300">Probability</span>
                    </div>
                    <div class="flex items-center space-x-4">
                        <span class="bg-purple-100 dark:bg-purple-800 text-purple-800 dark:text-purple-200 px-3 py-1 rounded-full font-medium transition-colors duration-300">[XGBoost, Random Forest]</span>
                        <i class="fas fa-arrow-right text-purple-500 dark:text-purple-400 transition-colors duration-300"></i>
                        <span class="bg-purple-100 dark:bg-purple-800 text-purple-800 dark:text-purple-200 px-3 py-1 rounded-full font-medium transition-colors duration-300">Neural Network</span>
                        <i class="fas fa-arrow-right text-purple-500 dark:text-purple-400 transition-colors duration-300"></i>
                        <span class="bg-purple-100 dark:bg-purple-800 text-purple-800 dark:text-purple-200 px-3 py-1 rounded-full font-medium transition-colors duration-300">Final Decision</span>
                    </div>
                </div>
                <div class="mt-4 p-3 bg-${agreementClass === 'success' ? 'green' : 'yellow'}-100 dark:bg-${agreementClass === 'success' ? 'green' : 'yellow'}-900/20 border border-${agreementClass === 'success' ? 'green' : 'yellow'}-200 dark:border-${agreementClass === 'success' ? 'green' : 'yellow'}-800 rounded transition-colors duration-300">
                    <p class="text-sm text-${agreementClass === 'success' ? 'green' : 'yellow'}-800 dark:text-${agreementClass === 'success' ? 'green' : 'yellow'}-200 transition-colors duration-300">
                        ${agreementStatus}
                    </p>
                </div>
            </div>
        `;
    }
}

// Format parameter names
function formatParameterName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// Reset prediction form
function resetPredictionForm() {
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsSection.classList.add('hidden');
    }
    
    // Reset to manual input
    switchInputMethod('manual');
    
    // Clear template and random data
    window.currentTemplateData = null;
    window.currentRandomData = null;
    
    // Reset forms
    const templatePreview = document.getElementById('template-preview');
    const randomPreview = document.getElementById('random-preview');
    
    if (templatePreview) templatePreview.classList.add('hidden');
    if (randomPreview) randomPreview.classList.add('hidden');
}

// Export functions for global access
window.PredictionPage = {
    resetPredictionForm,
    switchInputMethod,
    loadTemplateData,
    generateRandomParameters,
    makePrediction
};
