/**
 * GPU Inference Pipeline — shared utilities
 */

// Chart.js global defaults for the dark theme
function chartDefaults(yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { family: "'JetBrains Mono', monospace", size: 11 } }
      },
      tooltip: {
        backgroundColor: 'rgba(15,19,32,0.95)',
        borderColor: 'rgba(255,255,255,0.08)',
        borderWidth: 1,
        titleColor: '#e2e8f0',
        bodyColor: '#94a3b8',
        padding: 10,
        cornerRadius: 8,
      }
    },
    scales: {
      x: {
        ticks: { color: '#6b7280', font: { family: "'JetBrains Mono', monospace", size: 10 } },
        grid:  { color: 'rgba(255,255,255,0.04)' },
      },
      y: {
        ticks: { color: '#6b7280', font: { family: "'JetBrains Mono', monospace", size: 10 } },
        grid:  { color: 'rgba(255,255,255,0.04)' },
        title: yLabel ? { display: true, text: yLabel, color: '#6b7280', font: { size: 10 } } : undefined,
      }
    }
  };
}
