<template>

    <div class="about-section">
        <h2>Training Mask Examples</h2>
        <div class="examples-container">
            <div class="example-item" v-for="example in examples" :key="example.name">
                <a :href="example.link">
                    <img :src="example.image" alt="Example" class="example-image" />
                    <div class="example-name">{{ example.name }}</div>
                </a>
            </div>
        </div>

    </div>
</template>



<script setup>

import face from '../assets/examples/face.png'
import boat from '../assets/examples/boat.png'
import art from '../assets/examples/art.png'
import circles from '../assets/examples/circles.png'
import target from '../assets/examples/target.png'

let examples = [
    {
        name: "Target",
        link: "http://localhost:5173/?maskInput=circle%280.9%2C0%2C0%29%26%21circle%280.6%2C0%2C0%29%26circle%280.4%2C0%2C0%29%0A&batchSize=10&lossFunction=bce&randomSeed=42&learningRate=0.015&epochs=2000&trainingDataAmount=2500&layerSizesInput=2%2C20%2C10%2C1",
        image: target
    },
    {
        name: "Mild Face",
        link: "/?maskInput=circle%280.9%2C0%2C0%29%26%21circle%280.1%2C-0.3%2C0.3%29%26%21circle%280.1%2C0.3%2C0.3%29%26%21circle%280.4%2C0%2C-0.3%29%26circle%280.4%2C0%2C-0.3%29%26%21rec%28-0.4%2C-0.3%2C0.4%2C0%29%0A&batchSize=18&lossFunction=bce&randomSeed=42&learningRate=0.03&epochs=1500&trainingDataAmount=2000&layerSizesInput=2%2C15%2C10%2C5%2C1",
        image: face
    },
    {
        name: "Sail Boat",
        link: "/?maskInput=circle%280.6%2C0%2C-0.6%29+%26+%21rec%28-1%2C-0.6%2C1%2C0%29+%26+rec%28-0.02%2C-0.6%2C0.02%2C0.4%29+%26+circle%280.4%2C0%2C0.1%29+%26+%21rec%28-1%2C0.1%2C0%2C1%29%0A&batchSize=18&lossFunction=bce&randomSeed=42&learningRate=0.03&epochs=1500&trainingDataAmount=2000&layerSizesInput=2%2C15%2C10%2C5%2C1",
        image: boat
    },
    {
        name: "Art",
        link: "/?maskInput=circle%280.5%2C0%2C-0.3%29+%26+circle%280.2%2C0.3%2C0%29+%26+rec%280.35%2C-0.5%2C0.42%2C-1%29+%26+rec%28-0.3%2C-0.5%2C-0.23%2C-1%29+%26+rec%280.2%2C-0.3%2C0.25%2C0.2%29+%26+circle%280.2%2C0.45%2C0.1%29+%26+rec%28-0.48%2C-0.3%2C-0.45%2C-1%29+%26+circle%280.5%2C1%2C1%29+%26+%21circle%280.4%2C1%2C1%29+%26+circle%280.3%2C1%2C1%29+%26+rec%28-0.95%2C-1%2C-0.65%2C0.4%29+%0A%26+%21circle%280.1%2C-0.8%2C0.2%29&batchSize=18&lossFunction=bce&randomSeed=42&learningRate=0.03&epochs=1500&trainingDataAmount=2000&layerSizesInput=2%2C35%2C25%2C15%2C10%2C5%2C1",
        image: art
    },
    {
        name: "Circles",
        link: "http://localhost:5173/?learningRate=0.005&epochs=1500&trainingDataAmount=9000&layerSizesInput=2%2C60%2C30%2C15%2C1&maskInput=circle%281%2C0%2C0%29+%26+%21circle%280.95%2C0%2C0%29+%26+circle%280.9%2C0%2C0%29+%26+%21circle%280.85%2C0%2C0%29+%26+circle%280.8%2C0%2C0%29+%26+%21circle%280.75%2C0%2C0%29+%26+circle%280.7%2C0%2C0%29+%26+%21circle%280.65%2C0%2C0%29+%26+circle%280.6%2C0%2C0%29+%26+%21circle%280.55%2C0%2C0%29+%26+circle%280.5%2C0%2C0%29+%26+%21circle%280.45%2C0%2C0%29+%26+circle%280.4%2C0%2C0%29+%26%21circle%280.35%2C0%2C0%29+%26+circle%280.3%2C0%2C0%29+%26+%21circle%280.25%2C0%2C0%29+%26+circle%280.2%2C0%2C0%29+%26+%21circle%280.15%2C0%2C0%29+%26+circle%280.1%2C0%2C0%29+%26+%21circle%280.05%2C0%2C0%29&batchSize=1000&lossFunction=bce&randomSeed=42",
        image: circles
    }
]



</script>
