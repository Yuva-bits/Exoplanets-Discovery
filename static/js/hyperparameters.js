// Hyperparameter Tuning JavaScript
// Handles neural network hyperparameter experimentation

document.addEventListener('DOMContentLoaded', function() {
    initializeHyperparameterPage();
});

// Initialize hyperparameter page
function initializeHyperparameterPage() {
    console.log('Initializing hyperparameter tuning page...');
    
    setupSliderControls();
    setupPresetButtons();
    setupFormSubmission();
    loadPresets();
    
    console.log('Hyperparameter tuning page initialized');
}

// Setup slider controls with live value updates
function setupSliderControls() {
    const sliders = document.querySelectorAll('input[type="range"]');
    
    sliders.forEach(slider => {
        // Update display value on input
        slider.addEventListener('input', function() {
            const valueDisplay = document.getElementById(this.id + '_value');
            if (valueDisplay) {
                let displayValue = this.value;
                
                // Format display values
                if (this.id === 'learning_rate') {
                    displayValue = parseFloat(displayValue).toFixed(4);
                } else if (this.id === 'dropout_rate') {
                    displayValue = parseFloat(displayValue).toFixed(2);
                }
                
                valueDisplay.textContent = displayValue;
            }
            
            // Update slider visual state
            updateSliderVisualState(this);
        });
        
        // Initialize visual state
        updateSliderVisualState(slider);
    });
}

// Update slider visual state
function updateSliderVisualState(slider) {
    const percentage = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
    slider.style.background = `linear-gradient(to right, #3B82F6 0%, #3B82F6 ${percentage}%, #E5E7EB ${percentage}%, #E5E7EB 100%)`;
}

// Setup preset buttons
function setupPresetButtons() {
    const presetButtons = document.querySelectorAll('.preset-btn');
    
    presetButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all preset buttons
            presetButtons.forEach(btn => {
                btn.classList.remove('ring-2', 'ring-blue-500', 'bg-blue-50', 'dark:bg-blue-900/20');
                btn.classList.add('bg-gray-100', 'dark:bg-gray-700');
            });
            
            // Add active class to clicked button
            this.classList.remove('bg-gray-100', 'dark:bg-gray-700');
            this.classList.add('ring-2', 'ring-blue-500', 'bg-blue-50', 'dark:bg-blue-900/20');
            
            // Apply preset configuration
            const presetId = this.id.replace('preset-', '');
            applyPreset(presetId);
        });
    });
}

// Load presets from API
async function loadPresets() {
    try {
        const response = await fetch('/api/hyperparameters/presets');
        const data = await response.json();
        
        if (data.success) {
            window.hyperparameterPresets = data.presets;
            console.log('Hyperparameter presets loaded');
        }
    } catch (error) {
        console.error('Error loading presets:', error);
    }
}

// Apply preset configuration
function applyPreset(presetId) {
    if (!window.hyperparameterPresets || !window.hyperparameterPresets[presetId]) {
        console.error('Preset not found:', presetId);
        return;
    }
    
    const preset = window.hyperparameterPresets[presetId];
    console.log('Applying preset:', preset.name);
    
    // Update form values
    const form = document.getElementById('hyperparameter-form');
    const formData = new FormData(form);
    
    // Set all form values
    Object.keys(preset).forEach(key => {
        if (key !== 'name' && key !== 'description') {
            const element = document.getElementById(key);
            if (element) {
                element.value = preset[key];
                
                // Trigger input event to update display values and visual state
                const event = new Event('input', { bubbles: true });
                element.dispatchEvent(event);
            }
        }
    });
    
    // Show preset description
    showPresetDescription(preset.name, preset.description);
}

// Show preset description
function showPresetDescription(name, description) {
    // Remove existing description
    const existingDesc = document.querySelector('.preset-description');
    if (existingDesc) {
        existingDesc.remove();
    }
    
    // Create new description
    const descriptionDiv = document.createElement('div');
    descriptionDiv.className = 'preset-description mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800';
    descriptionDiv.innerHTML = `
        <div class="flex items-center">
            <i class="fas fa-info-circle text-blue-600 dark:text-blue-400 mr-2"></i>
            <div>
                <div class="text-sm font-medium text-blue-900 dark:text-blue-200">${name} Configuration</div>
                <div class="text-xs text-blue-700 dark:text-blue-300">${description}</div>
            </div>
        </div>
    `;
    
    // Insert after preset buttons
    const presetContainer = document.querySelector('.grid.grid-cols-2.gap-3');
    presetContainer.parentNode.insertBefore(descriptionDiv, presetContainer.nextSibling);
}

// Setup form submission
function setupFormSubmission() {
    const form = document.getElementById('hyperparameter-form');
    const submitBtn = document.getElementById('run-experiment-btn');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Disable submit button and show loading
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Running Experiment...';
        
        try {
            // Collect form data
            const formData = new FormData(form);
            const hyperparameters = {};
            
            for (let [key, value] of formData.entries()) {
                // Convert numeric values
                if (['hidden_layers', 'neurons_per_layer', 'batch_size', 'epochs'].includes(key)) {
                    hyperparameters[key] = parseInt(value);
                } else if (['dropout_rate', 'learning_rate'].includes(key)) {
                    hyperparameters[key] = parseFloat(value);
                } else {
                    hyperparameters[key] = value;
                }
            }
            
            console.log('Running experiment with hyperparameters:', hyperparameters);
            
            // Run experiment
            const results = await runExperiment(hyperparameters);
            
            // Display results
            displayExperimentResults(results);
            
        } catch (error) {
            console.error('Experiment failed:', error);
            showExperimentError(error.message);
        } finally {
            // Re-enable submit button
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-play mr-2"></i>Run Experiment';
        }
    });
}

// Run hyperparameter experiment
async function runExperiment(hyperparameters) {
    const response = await fetch('/api/hyperparameters/experiment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(hyperparameters)
    });
    
    const data = await response.json();
    
    if (!data.success) {
        throw new Error(data.error || 'Experiment failed');
    }
    
    return data.results;
}

// Display experiment results
function displayExperimentResults(results) {
    const resultsContainer = document.getElementById('experiment-results');
    
    // Calculate performance improvement
    const baseAccuracy = 0.9189;
    const baseAUC = 0.9541;
    const accuracyImprovement = ((results.accuracy - baseAccuracy) / baseAccuracy * 100).toFixed(2);
    const aucImprovement = ((results.auc - baseAUC) / baseAUC * 100).toFixed(2);
    
    resultsContainer.innerHTML = `
        <!-- Performance Metrics -->
        <div class="space-y-4">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white">Performance Metrics</h3>
            
            <div class="grid grid-cols-2 gap-4">
                <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-blue-600 dark:text-blue-400 text-sm font-medium">Accuracy</p>
                            <p class="text-2xl font-bold text-blue-800 dark:text-blue-200">${(results.accuracy * 100).toFixed(2)}%</p>
                            <p class="text-xs text-blue-600 dark:text-blue-400">${accuracyImprovement > 0 ? '+' : ''}${accuracyImprovement}% vs baseline</p>
                        </div>
                        <i class="fas fa-bullseye text-blue-500 text-xl"></i>
                    </div>
                </div>
                
                <div class="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border border-green-200 dark:border-green-800">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-green-600 dark:text-green-400 text-sm font-medium">AUC Score</p>
                            <p class="text-2xl font-bold text-green-800 dark:text-green-200">${(results.auc * 100).toFixed(2)}%</p>
                            <p class="text-xs text-green-600 dark:text-green-400">${aucImprovement > 0 ? '+' : ''}${aucImprovement}% vs baseline</p>
                        </div>
                        <i class="fas fa-chart-line text-green-500 text-xl"></i>
                    </div>
                </div>
                
                <div class="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border border-purple-200 dark:border-purple-800">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-purple-600 dark:text-purple-400 text-sm font-medium">F1-Score</p>
                            <p class="text-2xl font-bold text-purple-800 dark:text-purple-200">${(results.f1_score * 100).toFixed(2)}%</p>
                            <p class="text-xs text-purple-600 dark:text-purple-400">Balanced metric</p>
                        </div>
                        <i class="fas fa-balance-scale text-purple-500 text-xl"></i>
                    </div>
                </div>
                
                <div class="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 border border-orange-200 dark:border-orange-800">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-orange-600 dark:text-orange-400 text-sm font-medium">Training Time</p>
                            <p class="text-2xl font-bold text-orange-800 dark:text-orange-200">${results.training_time}s</p>
                            <p class="text-xs text-orange-600 dark:text-orange-400">Simulated time</p>
                        </div>
                        <i class="fas fa-clock text-orange-500 text-xl"></i>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Additional Metrics -->
        <div class="space-y-4">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white">Additional Metrics</h3>
            
            <div class="grid grid-cols-2 gap-4">
                <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-600 dark:text-gray-400">Precision</span>
                        <span class="font-semibold text-gray-900 dark:text-white">${(results.precision * 100).toFixed(2)}%</span>
                    </div>
                </div>
                <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-600 dark:text-gray-400">Recall</span>
                        <span class="font-semibold text-gray-900 dark:text-white">${(results.recall * 100).toFixed(2)}%</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Configuration Summary -->
        <div class="space-y-4">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white">Configuration Used</h3>
            
            <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div class="grid grid-cols-2 gap-4 text-sm">
                    <div class="space-y-2">
                        <div class="flex justify-between">
                            <span class="text-gray-600 dark:text-gray-400">Hidden Layers:</span>
                            <span class="font-medium text-gray-900 dark:text-white">${results.hyperparameters.hidden_layers}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600 dark:text-gray-400">Neurons per Layer:</span>
                            <span class="font-medium text-gray-900 dark:text-white">${results.hyperparameters.neurons_per_layer}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600 dark:text-gray-400">Dropout Rate:</span>
                            <span class="font-medium text-gray-900 dark:text-white">${results.hyperparameters.dropout_rate}</span>
                        </div>
                    </div>
                    <div class="space-y-2">
                        <div class="flex justify-between">
                            <span class="text-gray-600 dark:text-gray-400">Learning Rate:</span>
                            <span class="font-medium text-gray-900 dark:text-white">${results.hyperparameters.learning_rate}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600 dark:text-gray-400">Batch Size:</span>
                            <span class="font-medium text-gray-900 dark:text-white">${results.hyperparameters.batch_size}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600 dark:text-gray-400">Epochs:</span>
                            <span class="font-medium text-gray-900 dark:text-white">${results.hyperparameters.epochs}</span>
                        </div>
                    </div>
                </div>
                <div class="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
                    <div class="grid grid-cols-2 gap-4 text-sm">
                        <div class="flex justify-between">
                            <span class="text-gray-600 dark:text-gray-400">Activation:</span>
                            <span class="font-medium text-gray-900 dark:text-white">${results.hyperparameters.activation.toUpperCase()}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600 dark:text-gray-400">Optimizer:</span>
                            <span class="font-medium text-gray-900 dark:text-white">${results.hyperparameters.optimizer.toUpperCase()}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Performance Analysis -->
        <div class="space-y-4">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white">Performance Analysis</h3>
            
            <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div class="space-y-3 text-sm text-gray-600 dark:text-gray-400">
                    ${generatePerformanceAnalysis(results, accuracyImprovement, aucImprovement)}
                </div>
            </div>
        </div>
    `;
}

// Generate performance analysis
function generatePerformanceAnalysis(results, accuracyImprovement, aucImprovement) {
    let analysis = [];
    
    // Accuracy analysis
    if (accuracyImprovement > 2) {
        analysis.push('<p><strong class="text-green-600">✓ Excellent accuracy improvement!</strong> This configuration significantly outperforms the baseline.</p>');
    } else if (accuracyImprovement > 0) {
        analysis.push('<p><strong class="text-blue-600">✓ Good accuracy improvement.</strong> This configuration shows modest gains over the baseline.</p>');
    } else if (accuracyImprovement > -2) {
        analysis.push('<p><strong class="text-yellow-600">Similar accuracy to baseline.</strong> Consider adjusting hyperparameters for better performance.</p>');
    } else {
        analysis.push('<p><strong class="text-red-600">✗ Lower accuracy than baseline.</strong> This configuration may be overfitting or underfitting.</p>');
    }
    
    // AUC analysis
    if (aucImprovement > 1) {
        analysis.push('<p><strong class="text-green-600">✓ Strong discrimination ability!</strong> The model effectively separates exoplanets from false positives.</p>');
    } else if (aucImprovement > -1) {
        analysis.push('<p><strong class="text-blue-600">✓ Good discrimination ability.</strong> The model maintains solid separation performance.</p>');
    } else {
        analysis.push('<p><strong class="text-yellow-600">Reduced discrimination ability.</strong> Consider optimizing for better class separation.</p>');
    }
    
    // Training efficiency analysis
    if (results.training_time < 30) {
        analysis.push('<p><strong class="text-green-600">✓ Fast training!</strong> This configuration trains efficiently.</p>');
    } else if (results.training_time < 60) {
        analysis.push('<p><strong class="text-blue-600">✓ Moderate training time.</strong> Good balance between performance and efficiency.</p>');
    } else {
        analysis.push('<p><strong class="text-orange-600">Longer training time.</strong> Consider reducing model complexity for faster training.</p>');
    }
    
    // Precision-Recall balance
    const precisionRecallRatio = results.precision / results.recall;
    if (precisionRecallRatio > 1.2) {
        analysis.push('<p><strong class="text-blue-600">High precision model.</strong> Fewer false positives, but may miss some exoplanets.</p>');
    } else if (precisionRecallRatio < 0.8) {
        analysis.push('<p><strong class="text-green-600">High recall model.</strong> Finds most exoplanets, but may have more false positives.</p>');
    } else {
        analysis.push('<p><strong class="text-purple-600">Balanced model.</strong> Good balance between precision and recall.</p>');
    }
    
    return analysis.join('');
}

// Show experiment error
function showExperimentError(message) {
    const resultsContainer = document.getElementById('experiment-results');
    
    resultsContainer.innerHTML = `
        <div class="text-center py-12">
            <i class="fas fa-exclamation-triangle text-red-500 text-4xl mb-4"></i>
            <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">Experiment Failed</h3>
            <p class="text-red-600 dark:text-red-400 mb-4">${message}</p>
            <button onclick="location.reload()" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-300">
                Try Again
            </button>
        </div>
    `;
}
