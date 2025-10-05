// Statistics Page JavaScript
// Handles model statistics and performance visualizations

// Global chart instances to prevent duplicates
let precisionRecallChartInstance = null;
let isLoading = false;

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded - Initializing statistics page');
    initializeStatisticsPage();
});

// Handle page visibility changes to refresh visualizations
document.addEventListener('visibilitychange', function() {
    if (!document.hidden && !isLoading) {
        console.log('Page became visible - refreshing visualizations');
        setTimeout(() => {
            initializeStatisticsPage();
        }, 500); // Small delay to ensure page is fully visible
    }
});

// Initialize statistics page
function initializeStatisticsPage() {
    if (isLoading) {
        console.log('Statistics page is already loading, skipping...');
        return;
    }
    
    isLoading = true;
    console.log('Initializing statistics page...');
    
    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.error('Chart.js is not loaded!');
        // Try to load Chart.js dynamically
        loadChartJS();
    } else {
        console.log('Chart.js is loaded successfully');
        // Proceed with initialization
        proceedWithInitialization();
    }
}

// Clear all visualizations before reloading
function clearVisualizations() {
    console.log('Clearing all visualizations...');
    
    // Clear chart instances
    
    if (precisionRecallChartInstance) {
        precisionRecallChartInstance.destroy();
        precisionRecallChartInstance = null;
    }
    
    // Clear visualization containers
    const containers = ['correlation-matrix', 'high-impact-features', 'roc-curves'];
    containers.forEach(id => {
        const container = document.getElementById(id);
        if (container) {
            container.innerHTML = '<div class="text-center text-gray-500">Loading...</div>';
        }
    });
}

// Proceed with initialization after Chart.js is confirmed loaded
function proceedWithInitialization() {
    try {
        clearVisualizations();
        loadStatisticsData();
        setupCharts();
        setupInteractiveElements();
        loadAdvancedVisualizations();
        console.log('Statistics page initialization complete');
    } catch (error) {
        console.error('Error during statistics page initialization:', error);
    } finally {
        isLoading = false;
    }
}

// Load Chart.js dynamically if not already loaded
function loadChartJS() {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
    script.onload = function() {
        console.log('Chart.js loaded dynamically');
        proceedWithInitialization();
    };
    script.onerror = function() {
        console.error('Failed to load Chart.js');
        isLoading = false;
    };
    document.head.appendChild(script);
}

// Load statistics data
async function loadStatisticsData() {
    try {
        const response = await fetch('/api/statistics');
        const data = await response.json();
        
        if (response.ok) {
            if (data.default) {
                // Use default data
                displayDefaultStatistics(data.data);
            } else {
                // Use loaded data
                displayStatistics(data.data);
            }
        } else {
            console.error('Error loading statistics:', data.error);
            ExoplanetApp.showNotification('Failed to load statistics data.', 'error');
        }
    } catch (error) {
        console.error('Statistics loading error:', error);
        ExoplanetApp.showNotification('Error loading statistics data.', 'error');
    }
}

// Display default statistics
function displayDefaultStatistics(data) {
    // Update performance overview cards
    updatePerformanceCards(data);
    
    // Create charts with default data
    // Don't call createFeatureImportanceChart as it conflicts with API images
}

// Display loaded statistics
function displayStatistics(data) {
    // Update performance overview cards
    updatePerformanceCards(data);
    
    // Create charts with loaded data
    // Don't call createFeatureImportanceChart as it conflicts with API images
}

// Update performance cards
function updatePerformanceCards(data) {
    // This function can be used to dynamically update performance cards
    // if needed in the future
    console.log('Performance cards updated with data:', data);
}

// Setup charts
function setupCharts() {
    // Initialize Chart.js if not already loaded
    if (typeof Chart === 'undefined') {
        console.error('Chart.js not loaded');
        return;
    }
    
    // Set default Chart.js configuration
    Chart.defaults.font.family = 'Glacial Indifference, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    Chart.defaults.color = '#6B7280';
    Chart.defaults.plugins.legend.display = false;
    
    console.log('Chart.js configuration set up successfully');
}


// Create feature importance chart - DISABLED (conflicts with API images)
function createFeatureImportanceChart(data) {
    // This function is disabled because it conflicts with the API image loading
    // The key-features element is used to display images from /api/key-features
    console.log('Feature importance chart creation disabled - using API images instead');
}

// Setup interactive elements
function setupInteractiveElements() {
    // Add hover effects to performance cards
    setupPerformanceCardInteractions();
    
    // Add click handlers for expandable sections
    setupExpandableSections();
    
    // Setup tooltips for metrics
    setupMetricTooltips();
}

// Setup performance card interactions
function setupPerformanceCardInteractions() {
    const statCards = document.querySelectorAll('.stat-card');
    
    statCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-4px)';
            this.style.transition = 'transform 0.2s ease';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

// Setup expandable sections
function setupExpandableSections() {
    // Add click handlers for any expandable content
    const expandableElements = document.querySelectorAll('[data-expandable]');
    
    expandableElements.forEach(element => {
        element.addEventListener('click', function() {
            const target = document.querySelector(this.getAttribute('data-target'));
            if (target) {
                target.classList.toggle('hidden');
                
                const icon = this.querySelector('i');
                if (icon) {
                    icon.classList.toggle('fa-chevron-down');
                    icon.classList.toggle('fa-chevron-up');
                }
            }
        });
    });
}

// Setup metric tooltips
function setupMetricTooltips() {
    const metrics = document.querySelectorAll('.metric-card, .stat-card');
    
    metrics.forEach(metric => {
        metric.addEventListener('mouseenter', function() {
            // Add subtle glow effect
            this.style.boxShadow = '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)';
        });
        
        metric.addEventListener('mouseleave', function() {
            // Remove glow effect
            this.style.boxShadow = '';
        });
    });
}

// Create confidence distribution chart (if needed)
function createConfidenceChart() {
    const ctx = document.getElementById('confidence-chart');
    if (!ctx) return;
    
    const confidenceData = {
        ranges: ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%', '<50%'],
        percentages: [45.2, 28.7, 15.3, 7.8, 2.1, 0.9]
    };
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: confidenceData.ranges,
            datasets: [{
                data: confidenceData.percentages,
                backgroundColor: [
                    '#10B981', '#3B82F6', '#F59E0B', '#EF4444', '#8B5CF6', '#6B7280'
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#374151',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed + '%';
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// Create model comparison radar chart (if needed)
function createRadarChart() {
    const ctx = document.getElementById('radar-chart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Speed'],
            datasets: [
                {
                    label: 'Hierarchical Ensemble',
                    data: [83.90, 74.52, 70.86, 72.64, 91.93, 85],
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderColor: '#3B82F6',
                    borderWidth: 2,
                    pointBackgroundColor: '#3B82F6',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#3B82F6'
                },
                {
                    label: 'XGBoost',
                    data: [83.24, 74.70, 67.21, 70.76, 90.56, 90],
                    backgroundColor: 'rgba(16, 185, 129, 0.2)',
                    borderColor: '#10B981',
                    borderWidth: 2,
                    pointBackgroundColor: '#10B981',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#10B981'
                },
                {
                    label: 'Random Forest',
                    data: [81.59, 73.06, 61.75, 66.93, 89.42, 75],
                    backgroundColor: 'rgba(139, 92, 246, 0.2)',
                    borderColor: '#8B5CF6',
                    borderWidth: 2,
                    pointBackgroundColor: '#8B5CF6',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#8B5CF6'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        font: {
                            size: 11
                        },
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        color: '#F3F4F6'
                    },
                    pointLabels: {
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// Load advanced visualizations
async function loadAdvancedVisualizations() {
    try {
        console.log('Starting to load advanced visualizations...');
        
        // Check if all required elements exist
        const elements = ['correlation-matrix', 'high-impact-features', 'roc-curves'];
        elements.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                console.log(`Element ${id} found`);
            } else {
                console.error(`Element ${id} not found!`);
            }
        });
        
        // Load visualizations sequentially to prevent conflicts
        await loadCorrelationMatrix();
        await new Promise(resolve => setTimeout(resolve, 100)); // Small delay
        
        await loadHighImpactFeatures();
        await new Promise(resolve => setTimeout(resolve, 100)); // Small delay
        
        await loadROCCurves();
        
        // Create precision-recall chart
        createPrecisionRecallChart();
        
        console.log('All visualizations loaded successfully');
    } catch (error) {
        console.error('Error loading visualizations:', error);
    }
}

// Load correlation matrix
async function loadCorrelationMatrix() {
    try {
        console.log('Loading correlation matrix...');
        const response = await fetch('/api/correlation-matrix');
        const data = await response.json();
        
        console.log('Correlation matrix response:', data);
        
        if (response.ok && data.success) {
            const container = document.getElementById('correlation-matrix');
            if (container) {
                let statsHtml = '';
                // Update correlation statistics in the new styled cards
                if (data.correlation_stats) {
                    const maxCorrelationEl = document.getElementById('max-correlation');
                    const minCorrelationEl = document.getElementById('min-correlation');
                    const avgCorrelationEl = document.getElementById('avg-correlation');
                    
                    if (maxCorrelationEl) maxCorrelationEl.textContent = (data.correlation_stats.max_correlation || 0).toFixed(3);
                    if (minCorrelationEl) minCorrelationEl.textContent = (data.correlation_stats.min_correlation || 0).toFixed(3);
                    if (avgCorrelationEl) avgCorrelationEl.textContent = (data.correlation_stats.avg_correlation || 0).toFixed(3);
                }
                
                container.innerHTML = `
                    <img src="${data.image}" alt="Correlation Matrix" class="w-full h-auto rounded-lg shadow-sm">
                    ${statsHtml}
                `;
                console.log('Correlation matrix loaded successfully');
            } else {
                console.error('Correlation matrix container not found');
            }
        } else {
            throw new Error(data.error || 'Failed to load correlation matrix');
        }
    } catch (error) {
        console.error('Error loading correlation matrix:', error);
        const container = document.getElementById('correlation-matrix');
        if (container) {
            container.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-exclamation-triangle text-red-500 text-2xl mb-2"></i>
                    <p class="text-red-600">Failed to load correlation matrix</p>
                </div>
            `;
        }
    }
}

// Create precision-recall chart
function createPrecisionRecallChart() {
    const ctx = document.getElementById('precision-recall-chart');
    if (!ctx) {
        console.error('Precision-recall chart container not found');
        return;
    }
    
    // Destroy existing chart instance if it exists
    if (precisionRecallChartInstance) {
        console.log('Destroying existing precision-recall chart');
        precisionRecallChartInstance.destroy();
        precisionRecallChartInstance = null;
    }
    
    console.log('Creating precision-recall chart');
    
    precisionRecallChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Hierarchical Ensemble', 'XGBoost', 'Random Forest'],
            datasets: [
                {
                    label: 'Precision',
                    data: [85.4, 78.9, 76.5],
                    backgroundColor: '#14B8A6',
                    borderColor: '#0D9488',
                    borderWidth: 1,
                    borderRadius: 8,
                    borderSkipped: false,
                },
                {
                    label: 'Recall',
                    data: [62.3, 58.8, 54.3],
                    backgroundColor: '#3B82F6',
                    borderColor: '#2563EB',
                    borderWidth: 1,
                    borderRadius: 8,
                    borderSkipped: false,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#374151',
                    borderWidth: 1,
                    cornerRadius: 8,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(107, 114, 128, 0.1)',
                        drawBorder: false
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11,
                            weight: '500'
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
}

// Load key features overview - High Impact Bar Graph
async function loadKeyFeatures() {
    try {
        console.log('Loading key features...');
        const response = await fetch('/api/key-features');
        const data = await response.json();
        
        console.log('Key features response:', data);
        
        if (response.ok && data.success) {
            const container = document.getElementById('key-features');
            if (container) {
                // Display the enhanced bar graph
                container.innerHTML = `
                    <div class="w-full">
                        <img src="${data.image}" alt="High-Impact Features Bar Graph" class="w-full h-auto rounded-lg shadow-sm border border-gray-200 dark:border-gray-600">
                    </div>
                `;
                
                // Add dynamic feature insights if available
                if (data.high_impact_features && typeof data.high_impact_features === 'object') {
                    try {
                        const criticalFeatures = Object.values(data.high_impact_features)
                            .filter(f => f && f.category === 'Critical' && typeof f.impact === 'number')
                            .sort((a, b) => b.impact - a.impact);
                        
                        if (criticalFeatures.length > 0) {
                            container.innerHTML += `
                                <div class="mt-6 bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 p-4 rounded-lg border border-red-200/50 dark:border-red-800/50">
                                    <h4 class="text-sm font-bold text-red-800 dark:text-red-300 mb-2 flex items-center">
                                        <i class="fas fa-exclamation-triangle mr-2"></i>
                                        Critical Impact Features (>20%)
                                    </h4>
                                    <div class="grid grid-cols-1 gap-2">
                                        ${criticalFeatures.map((feature, index) => `
                                            <div class="flex justify-between items-center text-xs">
                                                <span class="text-gray-700 dark:text-gray-300 font-medium">${feature.name || 'Unknown Feature'}</span>
                                                <span class="font-bold text-red-700 dark:text-red-400">${feature.impact.toFixed(1)}%</span>
                                            </div>
                                        `).join('')}
                                    </div>
                                    <div class="mt-3 text-center">
                                        <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300">
                                            <i class="fas fa-chart-bar mr-1"></i>
                                            Top ${criticalFeatures.length} Features Drive Primary Detection
                                        </span>
                                    </div>
                                </div>
                            `;
                        }
                    } catch (error) {
                        console.error('Error processing high impact features:', error);
                    }
                }
                console.log('Key features loaded successfully');
            } else {
                console.error('Key features container not found');
            }
        } else {
            throw new Error(data.error || 'Failed to load key features');
        }
    } catch (error) {
        console.error('Error loading key features:', error);
        const container = document.getElementById('key-features');
        if (container) {
            container.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-exclamation-triangle text-red-500 text-2xl mb-2"></i>
                    <p class="text-red-600 dark:text-red-400">Failed to load key features bar graph</p>
                    <button onclick="loadKeyFeatures()" class="mt-2 px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600">
                        Retry
                    </button>
                </div>
            `;
        }
    }
}

// Load feature importance chart
async function loadFeatureImportanceChart() {
    try {
        const response = await fetch('/api/feature-importance');
        const data = await response.json();
        
        if (response.ok && data.success) {
            const container = document.getElementById('feature-importance');
            if (container) {
                container.innerHTML = `
                    <img src="${data.image}" alt="Feature Importance" class="w-full h-auto rounded-lg shadow-sm">
                    <div class="mt-4 bg-gray-50 p-4 rounded-lg">
                        <h4 class="text-sm font-semibold text-gray-900 mb-2">Top Features:</h4>
                        <div class="grid grid-cols-2 gap-2 text-xs text-gray-600">
                            <div>• Orbital Period (koi_period)</div>
                            <div>• Planetary Radius (koi_prad)</div>
                            <div>• Equilibrium Temperature (koi_teq)</div>
                            <div>• Transit Depth (koi_depth)</div>
                        </div>
                    </div>
                `;
            }
        } else {
            throw new Error(data.error || 'Failed to load feature importance');
        }
    } catch (error) {
        console.error('Error loading feature importance:', error);
        const container = document.getElementById('feature-importance');
        if (container) {
            container.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-exclamation-triangle text-red-500 text-2xl mb-2"></i>
                    <p class="text-red-600">Failed to load feature importance</p>
                </div>
            `;
        }
    }
}

// Load high-impact features (simplified version for the high-impact-features container)
async function loadHighImpactFeatures() {
    try {
        console.log('Loading high-impact features...');
        const response = await fetch('/api/key-features');
        const data = await response.json();
        
        console.log('High-impact features response:', data);
        
        if (response.ok && data.success) {
            const container = document.getElementById('high-impact-features');
            if (container) {
                // Display the high-impact features bar graph
                container.innerHTML = `
                    <div class="w-full">
                        <img src="${data.image}" alt="High-Impact Features Bar Graph" class="w-full h-auto rounded-lg shadow-sm border border-gray-200 dark:border-gray-600">
                    </div>
                `;
                
                // Add dynamic feature insights if available
                if (data.high_impact_features) {
                    const criticalFeatures = Object.values(data.high_impact_features)
                        .filter(f => f.category === 'Critical')
                        .sort((a, b) => b.impact - a.impact);
                    
                    if (criticalFeatures.length > 0) {
                        container.innerHTML += `
                            <div class="mt-6 bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 p-4 rounded-lg border border-red-200/50 dark:border-red-800/50">
                                <h4 class="text-sm font-bold text-red-800 dark:text-red-300 mb-2 flex items-center">
                                    <i class="fas fa-exclamation-triangle mr-2"></i>
                                    Critical Impact Features (>20%)
                                </h4>
                                <div class="grid grid-cols-1 gap-2">
                                    ${criticalFeatures.map((feature, index) => `
                                        <div class="flex justify-between items-center text-xs">
                                            <span class="text-gray-700 dark:text-gray-300 font-medium">${feature.name}</span>
                                            <span class="font-bold text-red-700 dark:text-red-400">${feature.impact.toFixed(1)}%</span>
                                        </div>
                                    `).join('')}
                                </div>
                                <div class="mt-3 text-center">
                                    <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300">
                                        <i class="fas fa-chart-bar mr-1"></i>
                                        Top ${criticalFeatures.length} Features Drive Primary Detection
                                    </span>
                                </div>
                            </div>
                        `;
                    }
                }
                console.log('High-impact features loaded successfully');
            } else {
                console.error('High-impact features container not found');
            }
        } else {
            throw new Error(data.error || 'Failed to load high-impact features');
        }
    } catch (error) {
        console.error('Error loading high-impact features:', error);
        const container = document.getElementById('high-impact-features');
        if (container) {
            container.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-exclamation-triangle text-red-500 text-2xl mb-2"></i>
                    <p class="text-red-600 dark:text-red-400">Failed to load high-impact features bar graph</p>
                    <button onclick="loadHighImpactFeatures()" class="mt-2 px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600">
                        Retry
                    </button>
                </div>
            `;
        }
    }
}

// Load ROC curves
async function loadROCCurves() {
    try {
        const response = await fetch('/api/roc-curves');
        const data = await response.json();
        
        if (response.ok && data.success) {
            const container = document.getElementById('roc-curves');
            if (container) {
                container.innerHTML = `
                    <img src="${data.image}" alt="ROC Curves" class="w-full h-auto rounded-lg shadow-sm">
                `;
            }
        } else {
            throw new Error(data.error || 'Failed to load ROC curves');
        }
    } catch (error) {
        console.error('Error loading ROC curves:', error);
        const container = document.getElementById('roc-curves');
        if (container) {
            container.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-exclamation-triangle text-red-500 text-2xl mb-2"></i>
                    <p class="text-red-600 dark:text-red-400">Failed to load ROC curves</p>
                    <button onclick="loadROCCurves()" class="mt-2 px-3 py-1 bg-green-500 text-white rounded text-sm hover:bg-green-600">
                        Retry
                    </button>
                </div>
            `;
        }
    }
}

// Export functions for global access
window.StatisticsPage = {
    loadStatisticsData,
    createFeatureImportanceChart,
    createConfidenceChart,
    createRadarChart,
    loadAdvancedVisualizations,
    loadCorrelationMatrix,
    loadKeyFeatures,
    loadHighImpactFeatures,
    loadROCCurves
};
