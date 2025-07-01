<script setup>
const props = defineProps({
  activations: {
    type: Array,
    required: true
  }
});
</script>

<template>
  <div class="activation-vis">
    <div class="layer" v-for="(layer, lIdx) in activations" :key="lIdx">
      <div
        v-for="(act, nIdx) in layer"
        :key="nIdx"
        class="neuron"
        :style="{ backgroundColor: `rgba(78, 121, 167, ${act})` }"
        :title="`Layer ${lIdx + 1} Neuron ${nIdx + 1}: ${act.toFixed(3)}`"
      >
        {{ act.toFixed(2) }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.activation-vis {
  display: flex;
  gap: 12px;
  margin-top: 10px;
  flex-wrap: wrap;
}
.layer {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.neuron {
  width: 40px;
  height: 20px;
  background-color: #4e79a7;
  color: #fff;
  text-align: center;
  font-size: 0.75em;
  border-radius: 4px;
}
</style>
