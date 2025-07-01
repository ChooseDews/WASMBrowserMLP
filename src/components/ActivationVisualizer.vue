<script setup>


const props = defineProps({
  activations: {
    type: Array,
    required: true
  }
});


//make percent with min opactivty of 0.1
function toPercentWithMinOpacity(v) {
  const minOpacity = 0.15;
  const maxOpacity = 1;
  const range = maxOpacity - minOpacity;
  const value = Math.abs(v);
  const normalizedValue = (value - minOpacity) / range;
  return Math.min(maxOpacity, Math.max(minOpacity, normalizedValue));
}

function blueToRed(value) {
  value = Math.abs(value);
  const max_value = 15;
  value = value / max_value;
  value = Math.max(0, Math.min(1, value));

  const r = Math.round(255 * value);
  const g = 0;
  const b = Math.round(255 * (1 - value));

  return `rgb(${r}, ${g}, ${b})`;
}


function label(lIdx) {
  if (lIdx === 0) return "Input";
  if (lIdx === props.activations.length - 1) return "Output";
  return "";
}


</script>

<template>
  <div class="activation-container">
    <h2>Activation Visualizer</h2>
    <div class="activation-vis">
    <div class="layer" v-for="(layer, lIdx) in activations" :class="{'input': lIdx === 0, 'output': lIdx === activations.length - 1}" :key="lIdx">
      <div class="layer-label" v-if="label(lIdx) !== ''">{{ label(lIdx) }}</div>
      <div
        v-for="(act, nIdx) in layer"
        :key="nIdx"
        class="neuron"
        :style="{ opacity: toPercentWithMinOpacity(act), backgroundColor: blueToRed(act) }"
        :title="`Layer ${lIdx} Neuron ${nIdx + 1}: ${act.toFixed(3)}`"
      >
        {{ act.toFixed(2) }}
      </div>
    </div>
  </div>
  </div>
</template>

<style scoped>
.activation-container {
  padding: 0 20px;
  margin: 10px;
}
.activation-vis {
  display: flex;
  gap: 12px;
  margin-top: 30px;
  margin-bottom: 30px;
  max-width: 600px;
}
.layer {
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
  max-height: 70vh;
  gap: 1px;
  border: 4px solid #ccc;
  margin: auto;
  padding: 5px;
  background-color: rgb(233 233 233);
  border-radius: 4px;
}

.layer.input {
  border-color: #4e79a7;
}

.layer.output {
  border-color: #f28e2b;
}

.neuron {
  width: 40px;
  background-color: #4e79a7;
  color: #fff;
  text-align: center;
  font-size: 0.75em;
  border-radius: 4px;
  flex-wrap: wrap;
}

.output .neuron {
  background-color: #f28e2b;
}

.layer-label {
  text-align: center;
  font-size: 0.75em;
  font-weight: bold;
  margin-bottom: 10px;
}
</style>
