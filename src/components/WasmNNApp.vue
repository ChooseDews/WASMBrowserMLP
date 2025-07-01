<script setup>
import { ref, onMounted, computed, nextTick } from 'vue'
import init, { MLP, get_training_data, plot_results, plot_mask, set_seed, plot_loss } from '../../rust_lib/pkg/rust_lib.js';
import About from './About.vue';
import ActivationVisualizer from './ActivationVisualizer.vue';

const predictions = ref([])
const trainingLog = ref("")
const accuracy = ref(0)
const learningRate = ref(0.03);
const epochs = ref(1500);
const trainingDataAmount = ref(2000);
const layerSizesInput = ref("2,5,10,5,1");
const maskInput = ref("circle(0.5,-0.25,0) & circle(0.6,0.25,0) & !rec(-0.25,-0.25,0.25,0.25) & circle(1,1,1) & !circle(0.75,1,1) & circle(0.5,1,1)");
const batchSize = ref(18);
const lossFunction = ref("bce");
const loading = ref(true);
const isTraining = ref(false);
const shouldStopTraining = ref(false); //for user to stop training
const modelSummary = ref(null);
const trainingTimeMs = ref(null);
const currentEpoch = ref(0);
const totalEpochs = ref(0);
const trainingProgress = ref(0);
const currentLoss = ref(0);
const randomSeed = ref(42);
const hoverActivations = ref(null);

let mlp = null;
let trainingData = null;
let testData = null;

const paramKeys = [
  'learningRate',
  'epochs',
  'trainingDataAmount',
  'layerSizesInput',
  'maskInput',
  'batchSize',
  'lossFunction',
  'randomSeed',
];

function saveParamsToUrl() {
  const params = new URLSearchParams(window.location.search);
  params.set('learningRate', learningRate.value);
  params.set('epochs', epochs.value);
  params.set('trainingDataAmount', trainingDataAmount.value);
  params.set('layerSizesInput', layerSizesInput.value);
  params.set('maskInput', maskInput.value);
  params.set('batchSize', batchSize.value);
  params.set('lossFunction', lossFunction.value);
  params.set('randomSeed', randomSeed.value);
  const newUrl = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState({}, '', newUrl);
}

function loadParamsFromUrl() {
  const params = new URLSearchParams(window.location.search);
  if (params.has('learningRate')) learningRate.value = parseFloat(params.get('learningRate'));
  if (params.has('epochs')) epochs.value = parseInt(params.get('epochs'));
  if (params.has('trainingDataAmount')) trainingDataAmount.value = parseInt(params.get('trainingDataAmount'));
  if (params.has('layerSizesInput')) layerSizesInput.value = params.get('layerSizesInput');
  if (params.has('maskInput')) maskInput.value = params.get('maskInput');
  if (params.has('batchSize')) batchSize.value = parseInt(params.get('batchSize'));
  if (params.has('lossFunction')) lossFunction.value = params.get('lossFunction');
  if (params.has('randomSeed')) randomSeed.value = parseInt(params.get('randomSeed'));
}

const parsedLayerSizes = computed(() => {
  return layerSizesInput.value
    .split(',')
    .map(s => parseInt(s.trim(), 10))
    .filter(n => !isNaN(n));
});

const stopTraining = () => {
  shouldStopTraining.value = true;
  trainingLog.value += "\nTraining stopped by user.\n";
};

const startTraining = async () => {
  loading.value = true;
  isTraining.value = true;
  shouldStopTraining.value = false;
  trainingLog.value = "";
  predictions.value = [];
  accuracy.value = 0;
  modelSummary.value = null;
  trainingTimeMs.value = null;
  currentEpoch.value = 0;
  totalEpochs.value = epochs.value;
  trainingProgress.value = 0;
  currentLoss.value = 0;

  saveParamsToUrl();
  await nextTick();

  try {
    const layerSizes = layerSizesInput.value.split(',').map(s => parseInt(s.trim(), 10));
    if (layerSizes.some(isNaN) || layerSizes.length < 2) {
      alert("Please enter valid comma-separated layer sizes (e.g., 2,20,1).");
      loading.value = false;
      return;
    }

    set_seed(randomSeed.value);

    trainingData = get_training_data(trainingDataAmount.value, maskInput.value);
    testData = get_training_data(trainingDataAmount.value, maskInput.value);

    mlp = new MLP(layerSizes);
    mlp.set_loss_function(lossFunction.value);

    const lossCanvas = document.getElementById("loss-chart");
    if (lossCanvas) {
      lossCanvas.width = lossCanvas.clientWidth;
      lossCanvas.height = lossCanvas.clientHeight;
    }

    const losses = [];
    const totalEpochs = epochs.value;
    let chunkSize = 20;
    let target_update_interval = 1 * 200; // 1 second



    const startTime = Date.now();
    let last_update = Date.now();

    plotMask();
    for (let epochStart = 0; epochStart < totalEpochs; epochStart += chunkSize) {

      last_update = Date.now();
      if (shouldStopTraining.value) {
        break;
      }

      await new Promise(resolve => setTimeout(resolve, 0));

      const epochsToTrain = Math.min(chunkSize, totalEpochs - epochStart);

      const chunkLosses = mlp.train_epochs(trainingData, epochStart, epochsToTrain, learningRate.value, batchSize.value);

      //update chunk size based on time taken to train this chunk
      let dt = Date.now() - last_update;
      chunkSize = Math.max(1, Math.floor(chunkSize * target_update_interval / dt));


      //handle result plotting
      losses.push(...chunkLosses);
      const latestLoss = chunkLosses[chunkLosses.length - 1];
      const currentEpochNum = epochStart + epochsToTrain;
      currentEpoch.value = currentEpochNum;
      trainingProgress.value = (currentEpochNum / totalEpochs) * 100;
      currentLoss.value = latestLoss;
      trainingLog.value += `Epochs: ${epochStart + 1}-${currentEpochNum}/${totalEpochs}, Loss: ${latestLoss.toFixed(4)}\n`;
      const lossesFloat64Array = Float64Array.from(losses);
      plot_loss("loss-chart", lossesFloat64Array);
      accuracy.value = mlp.calculate_accuracy(testData);
      modelSummary.value = mlp.get_summary(testData);
      const resultCanvas = document.getElementById("result-chart");
      if (resultCanvas) {
        resultCanvas.width = resultCanvas.clientWidth;
        resultCanvas.height = resultCanvas.clientHeight;
      }
      plot_results("result-chart", mlp, testData, maskInput.value);
      trainingTimeMs.value = Date.now() - startTime;
      zeroActivationsResult();
    }

    loading.value = false;
    isTraining.value = false;

  } catch (error) {
    console.error("Training error:", error);
    trainingLog.value += `\nError during training: ${error.message}\n`;
    loading.value = false;
    isTraining.value = false;
  }
};

const plotMask = () => {
  const maskCanvas = document.getElementById("mask-chart");
  if (maskCanvas) {
    maskCanvas.width = maskCanvas.clientWidth;
    maskCanvas.height = maskCanvas.clientHeight;
    plot_mask("mask-chart", maskInput.value);
  }
};

const handleMouseMove = (e) => {
  if (!mlp) return;
  const canvas = e.target;
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const y = (1 - (e.clientY - rect.top) / rect.height) * 2 - 1;
  hoverActivations.value = mlp.forward_collect(Float64Array.from([x, y]));
  hoverActivations.value.unshift([x, y]);
};

const zeroActivationsResult = () => {
  hoverActivations.value = mlp.forward_collect(Float64Array.from([0, 0]));
  hoverActivations.value.unshift([0, 0]);
};

onMounted(async () => {
  loadParamsFromUrl();
  await init();
  startTraining();
});

</script>

<template>
  <div class="common-layout">
    <header class="control-bar">
      <h1>MLP Training Playground</h1>
      <p>Created by John Dews-Flick</p>
      <form class="controls" @submit.prevent="startTraining">
        <div class="param-group">
          <label for="layer-sizes">Layer Sizes</label>
          <input type="text" id="layer-sizes" v-model="layerSizesInput" placeholder="e.g. 2,5,5,1">
          <small>Comma-separated neurons per layer (input, hidden..., output)</small>
        </div>
        <div class="param-group">
          <label for="learning-rate">Learning Rate</label>
          <input type="number" id="learning-rate" v-model.number="learningRate" step="0.001" min="0.001" max="1.0">
          <small>Step size for gradient descent</small>
        </div>
        <div class="param-group">
          <label for="epochs">Epochs</label>
          <input type="number" id="epochs" v-model.number="epochs" step="10" min="10">
          <small>Number of training passes</small>
        </div>
        <div class="param-group">
          <label for="training-data-amount">Training Data</label>
          <input type="number" id="training-data-amount" v-model.number="trainingDataAmount" step="100" min="100">
          <small>Number of training samples</small>
        </div>
        <div class="param-group">
          <label for="batch-size">Batch Size</label>
          <input type="number" id="batch-size" v-model.number="batchSize" step="1" min="1" max="1000">
          <small>Samples per training batch</small>
        </div>
        <div class="param-group">
          <label for="loss-function">Loss Function</label>
          <select id="loss-function" v-model="lossFunction">
            <option value="bce">Binary Cross Entropy (BCE)</option>
            <option value="mse">Mean Squared Error (MSE)</option>
          </select>
        </div>
        <div class="param-group">
          <label for="randomSeed">Random Seed:</label>
          <input type="number" id="randomSeed" v-model.number="randomSeed" min="0" step="1">
          <small>Set a seed for reproducible results</small>
        </div>
        <div class="param-group param-mask">
          <label for="mask-input" title="Mask (e.g., circle(0.5,0,0) & !rec(0,0,0.1,0.1))">Mask</label>
          <textarea id="mask-input" v-model="maskInput" rows="2"
            placeholder="e.g. circle(0.75,0,0)&!rec(-0.25,-0.25,0.25,0.25)"></textarea>
          <small>Define the region for positive class. Use circle(r,x,y), rec(x1,y1,x2,y2), ! for negation, & for
            AND.</small>
        </div>
        <div class="submit-button">
          <button type="submit" :disabled="isTraining">Run Training</button>
        </div>
      </form>
    </header>

    <div v-if="loading" class="loading-message">
      <h2>Training in progress...</h2>
      <div class="progress-container">
        <div class="progress-bar" :style="{ width: trainingProgress + '%' }"></div>
      </div>
      <p class="progress-text">Epoch: {{ currentEpoch }}/{{ totalEpochs }} ({{ Math.round(trainingProgress) }}%)</p>
      <p class="progress-text">Current Loss: {{ currentLoss.toFixed(4) }}</p>
      <button type="button" @click="stopTraining" class="stop-button">Stop Training</button>
    </div>

    <main class="main-content">
      <div class="left-panel">
        <h2>Training Results:</h2>
        <div class="results-summary">
          <table v-if="modelSummary" class="model-summary-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Current Epoch</td>
                <td>{{ currentEpoch }} / {{ totalEpochs }}</td>
              </tr>
              <tr>
                <td>Accuracy</td>
                <td>{{ (modelSummary.accuracy * 100).toFixed(2) }}%</td>
              </tr>
              <tr>
                <td>Loss</td>
                <td>{{ modelSummary.final_loss.toFixed(6) }}</td>
              </tr>
              <tr>
                <td>Parameters</td>
                <td>{{ modelSummary.num_parameters }}</td>
              </tr>
              <tr v-if="trainingTimeMs !== null">
                <td>Training Time</td>
                <td>{{ (trainingTimeMs / 1000).toFixed(2) }} s</td>
              </tr>
            </tbody>
          </table>
        </div>
        <canvas id="loss-chart"></canvas>
        <h2>Training Log:</h2>
        <pre>{{ trainingLog }}</pre>



      </div>
      <div class="right-panel">
        <ActivationVisualizer v-if="hoverActivations" :activations="hoverActivations" />
        <canvas id="result-chart" @mousemove="handleMouseMove"></canvas>
        <canvas id="mask-chart"></canvas>
        <h2>MLP Visualizer:</h2>
        <div class="network-visualizer">
          <div class="network-row">
            <div v-for="(size, idx) in parsedLayerSizes" :key="idx" class="network-layer" :style="{
              height: `${Math.max(30, Math.min(size * 20, 300))}px`,
              lineHeight: `${Math.max(30, Math.min(size * 20, 300))}px`,
              width: '40px',
              background: idx === 0 ? '#4e79a7' : (idx === parsedLayerSizes.length - 1 ? '#f28e2b' : '#76b7b2'),
              alignItems: 'flex-end'
            }" :title="`Layer ${idx + 1}: ${size} neuron${size > 1 ? 's' : ''}`">
              <span class="layer-label">{{ size }}</span>
            </div>
          </div>
        </div>
      </div>
    </main>

    <About />
  </div>
</template>

<style scoped>
.common-layout {
  display: flex;
  flex-direction: column;
  min-height: 100vh;

}

.control-bar {
  display: flex;
  align-items: center;
  padding: 10px 20px;
  background-color: #202b42;
  border-bottom: 1px solid #ccc;
  gap: 20px;
  flex-wrap: wrap;
  /* Allow controls to wrap */
}

.control-bar h1 {
  margin: 0;
  font-size: 1.5em;
}

.controls {
  display: flex;
  gap: 18px;
  align-items: flex-start;
  flex-wrap: wrap;
  padding: 10px 0;
}

.param-group {
  display: flex;
  flex-direction: column;
  min-width: 160px;
  margin-bottom: 0;
}

.param-group label {
  font-weight: bold;
  margin-bottom: 2px;
}

.param-group small {
  font-size: 0.85em;
  color: #bbb;
  margin-top: 2px;
}

.param-group input,
.param-group select,
.param-group textarea {
  padding: 6px;
  border-radius: 4px;
  border: 1px solid #ccc;
  width: 100%;
  font-size: 1em;
  margin-bottom: 0;
}

.param-group textarea {
  resize: vertical;
  min-height: 38px;
  font-family: inherit;
}

.param-mask {
  min-width: 260px;
  flex: 1 1 260px;
}

.loading {
  opacity: 0.3;
  pointer-events: none;
}

.main-content {
  display: flex;
  flex: 1;
  padding: 30px;
  gap: 20px;
  overflow: auto;
  background-color: white;
  color: #333;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.left-panel {
  flex: 1;
  min-width: 300px;
  display: flex;
  flex-direction: column;
  padding: 20px;
  border-radius: 8px;
}

.right-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 400px;
}

pre {
  background-color: #f5f5f5;
  padding: 1em;
  border-radius: 5px;
  white-space: pre-wrap;
  text-align: left;
  color: #333;
  max-height: 200px;
  /* Limit height for training log */
  overflow-y: scroll;
  margin-bottom: 20px;
  border: 1px solid #e0e0e0;
}

.results-summary p {
  font-size: 1.1em;
  font-weight: bold;
  margin-bottom: 10px;
}

.predictions-list {
  list-style: none;
  padding: 0;
}

.predictions-list li {
  margin-bottom: 5px;
}

.prediction-inside {
  color: green;
  font-weight: bold;
}

.prediction-outside {
  color: red;
  font-weight: bold;
}

canvas {
  border: 1px solid #ccc;
  width: 100%;
  min-height: 300px;
  height: auto;
  margin-bottom: 10px;
  max-width: 700px;
  min-height: 500px;
  margin: 0 auto;
}

#mask-chart {
  border: 1px solid #ccc;
  width: 100%;
  height: auto;
  margin-bottom: 10px;
}

@media (max-width: 768px) {
  .control-bar {
    flex-direction: column;
    align-items: flex-start;
  }

  .main-content {
    flex-direction: column;
  }

  .left-panel,
  .right-panel {
    min-width: unset;
    width: 100%;
  }
}

.network-visualizer {
  margin: 20px 0;
  text-align: center;
}

.network-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  min-height: 100px;
}

.network-layer {
  display: flex;
  justify-content: center;
  align-items: center;
  border-radius: 8px 8px 8px 8px;
  position: relative;
  transition: height 0.3s;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.layer-label {
  color: #fff;
  font-weight: bold;
  font-size: 1.1em;
  margin: 0;
  text-align: center;
}

.network-labels {
  display: flex;
  justify-content: center;
  gap: 16px;
  margin-top: 4px;
}

.layer-type {
  font-size: 0.9em;
  color: #666;
}

.loading-message {
  width: 100%;
  text-align: center;
  padding: 60px 0;
  font-size: 1.5em;
  color: #fff;
}

.progress-container {
  width: 80%;
  margin: 20px auto;
  height: 20px;
  background-color: #333;
  border-radius: 10px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background-color: #4CAF50;
  transition: width 0.3s ease;
}

.progress-text {
  margin: 10px 0;
  font-size: 1em;
  color: #fff;
}

.stop-button {
  margin-top: 15px;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
  background-color: #d9534f;
  color: white;
  border: none;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.stop-button:hover {
  background-color: #c9302c;
}

.model-summary-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 1em;
}

.model-summary-table th,
.model-summary-table td {
  border: 1px solid #ccc;
  padding: 6px 12px;
  text-align: left;
}

.model-summary-table thead {
  background: #f5f5f5;
  color: #000;
}

.model-summary-table th {
  background: #f5f5f5;
}

.submit-button {
  flex: 0 0 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
}

button {
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>