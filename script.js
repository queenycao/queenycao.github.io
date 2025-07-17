// Global variables
let modelSession = null;
const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

// Initialize ONNX model
async function initModel() {
  try {
    console.log("Loading model...");
    const session = await ort.InferenceSession.create(
      "https://drive.google.com/uc?export=download&id=1a9JaFJroxtcewaWQLxbG4cnuooKTXkyv",
      { executionProviders: ['wasm'] }  // Force WebAssembly backend
    );
    console.log("✅ Model loaded!", session);
    return session;
  } catch (e) {
    console.error("❌ Model loading failed:", e);
    throw e;  // Rethrow to see the full error in console
  }
}

// Usage
initModel().then(session => {
  // Now use `session` for predictions
}).catch(console.error);

// Generate heatmap
async function generateHeatmap() {
    if (!modelSession) {
        alert("Model still loading. Please wait...");
        return;
    }

    const dayIndex = parseInt(document.getElementById('day-select').value);
    const canvas = document.getElementById('heatmap');
    const ctx = canvas.getContext('2d');
    
    // 1. Create inputs
    const noise = Array.from({length: 100}, () => Math.random() * 2 - 1);
    const condition = Array(7).fill(0);
    condition[dayIndex] = 1;

    // 2. Run model
    try {
        document.getElementById('loading').textContent = "Generating...";
        const inputs = {
            noise: new ort.Tensor('float32', noise, [1, 100]),
            condition: new ort.Tensor('float32', condition, [1, 7])
        };
        
        const results = await modelSession.run(inputs);
        const output = results.output.data;
        
        // 3. Render heatmap
        renderHeatmap(output, ctx);
        document.getElementById('loading').textContent = `Generated ${DAYS[dayIndex]} heatmap!`;
    } catch (error) {
        console.error("Generation failed:", error);
        document.getElementById('loading').textContent = "Error during generation.";
    }
}

// Render heatmap to canvas
function renderHeatmap(data, ctx) {
    const size = 64; // Assuming 64x64 output
    const scale = 4; // Scale up for better visibility
    
    // Normalize data
    const maxVal = Math.max(...data);
    const minVal = Math.min(...data);
    
    // Clear canvas
    ctx.clearRect(0, 0, size*scale, size*scale);
    
    // Draw heatmap
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const val = data[y * size + x];
            const intensity = Math.floor(((val - minVal) / (maxVal - minVal)) * 255);
            
            // Use red gradient (adjust as needed)
            ctx.fillStyle = `rgb(${intensity}, 50, 50)`;
            ctx.fillRect(x*scale, y*scale, scale, scale);
        }
    }
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', () => {
    initModel();
    document.getElementById('generate-btn').addEventListener('click', generateHeatmap);
});
