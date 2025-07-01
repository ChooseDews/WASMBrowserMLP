use wasm_bindgen::prelude::*;
use rand::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Serialize, Deserialize};
use js_sys::Function;
use plotters::prelude::*;
use plotters_canvas::CanvasBackend;
use std::cmp::min;
use std::cell::RefCell;


thread_local! {
    static GLOBAL_RNG: RefCell<ChaCha8Rng> = RefCell::new(ChaCha8Rng::seed_from_u64(42));
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TrainingData {
    pub inputs: Vec<f64>,
    pub label: Vec<f64>,
}


fn is_in_circle(x: f64, y: f64, r: f64, cx: f64, cy: f64) -> bool {
    (x - cx).powi(2) + (y - cy).powi(2) < r.powi(2)
}


fn is_in_rectangle(x: f64, y: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> bool {
    x >= x1.min(x2) && x <= x1.max(x2) && y >= y1.min(y2) && y <= y1.max(y2)
}

#[wasm_bindgen]
pub fn is_point_in_mask(x: f64, y: f64, mask_str: &str) -> bool {
    if mask_str.is_empty() {
        return false; // No mask means always outside, unless explicitly defined
    }

    let conditions: Vec<&str> = mask_str.split('&').collect();
    let mut overall_result = false;

    for condition in conditions {
        let trimmed_condition = condition.trim();
        let mut negate = false;
        let actual_condition = if trimmed_condition.starts_with('!') {
            negate = true;
            trimmed_condition.trim_start_matches('!').trim()
        } else {
            trimmed_condition
        };

        let mut current_result = false;

        if actual_condition.starts_with("circle(") && actual_condition.ends_with(')') {
            let params_str = actual_condition.trim_start_matches("circle(").trim_end_matches(')');
            let params: Vec<f64> = params_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            if params.len() == 3 {
                current_result = is_in_circle(x, y, params[0], params[1], params[2]);
            }
        } else if actual_condition.starts_with("rec(") && actual_condition.ends_with(')') {
            let params_str = actual_condition.trim_start_matches("rec(").trim_end_matches(')');
            let params: Vec<f64> = params_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            if params.len() == 4 {
                current_result = is_in_rectangle(x, y, params[0], params[1], params[2], params[3]);
            } 
        }

        if negate {
            if current_result {
                overall_result = false;
            }
        } else {
            overall_result = overall_result || current_result;
        }
    }
    overall_result
}

#[wasm_bindgen]
pub fn set_seed(seed: f64) {
    // Convert f64 to u64 for seeding
    let seed_u64 = seed as u64;
    GLOBAL_RNG.with(|rng| {
        *rng.borrow_mut() = ChaCha8Rng::seed_from_u64(seed_u64);
    });
}

#[wasm_bindgen]
pub fn get_training_data(num_samples: usize, mask_str_js: &JsValue) -> JsValue {
    let mut training_data = Vec::new();
    let mask_str = mask_str_js.as_string().unwrap_or_default();

    for _ in 0..num_samples {
        let (x, y) = GLOBAL_RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let x: f64 = rng.gen_range(-1.0..1.0);
            let y: f64 = rng.gen_range(-1.0..1.0);
            (x, y)
        });

        let is_inside = if is_point_in_mask(x, y, &mask_str) { 1.0 } else { 0.0 };
        training_data.push(TrainingData {
            inputs: vec![x, y],
            label: vec![is_inside],
        });
    }

    serde_wasm_bindgen::to_value(&training_data).unwrap()
}

#[wasm_bindgen]
pub fn plot_loss(canvas_id: &str, loss_data: &[f64]) {
    let backend = CanvasBackend::new(canvas_id).expect("cannot find canvas");
    let root = backend.into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss", ("sans-serif", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..loss_data.len(), 0f64..*loss_data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&1.0))
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart.draw_series(LineSeries::new(
        loss_data.iter().enumerate().map(|(i, &v)| (i, v)),
        &RED,
    )).unwrap();
}


enum Activation {
    ReLU,
    Sigmoid,
}


#[derive(Clone, Copy)]
enum LossFunction {
    MSE,
    BCE,
}

#[derive(Serialize, Deserialize)]
pub struct ModelSummary {
    pub accuracy: f64,
    pub final_loss: f64,
    pub num_parameters: usize,
}

#[wasm_bindgen]
pub struct MLP {
    layers: Vec<Layer>,
    loss_function: LossFunction,
}

#[wasm_bindgen]
impl MLP {
    #[wasm_bindgen(constructor)]
    pub fn new(layer_sizes: &[usize]) -> MLP {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let activation = if i == layer_sizes.len() - 2 {
                Activation::Sigmoid
            } else {
                Activation::ReLU
            };
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1], activation));
        }
        MLP { layers, loss_function: LossFunction::MSE }
    }

    
    pub fn set_loss_function(&mut self, loss: &str) {
        self.loss_function = match loss.to_lowercase().as_str() {
            "bce" => LossFunction::BCE,
            _ => LossFunction::MSE,
        };
    }

    
    pub fn train_epoch(&mut self, training_data: &JsValue, learning_rate: f64, batch_size: usize) -> f64 {
        let data: Vec<TrainingData> = serde_wasm_bindgen::from_value(training_data.clone()).unwrap();
        let n = data.len();
        let mut total_loss = 0.0;
        let mut i = 0;
        
        while i < n {
            // Reset gradients at the start of each batch
            self.reset_gradients();
            
            let end = min(i + batch_size, n);
            
            // Process each sample in the batch and accumulate gradients
            for j in i..end {
                let d = &data[j];
                let outputs = self.forward(&d.inputs);
                let loss = self.calculate_loss(&outputs, &d.label);
                total_loss += loss;
                // This now only accumulates gradients without updating weights
                self.backward(&d.inputs, &outputs, &d.label, learning_rate);
            }
            
            // Apply accumulated gradients after processing the entire batch
            self.apply_gradients(learning_rate);
            
            i += batch_size;
        }
        
        // Return the average loss for this epoch
        total_loss / n as f64
    }

    
    pub fn train_epochs(&mut self, training_data: &JsValue, _start_epoch: usize, num_epochs: usize, learning_rate: f64, batch_size: usize) -> Box<[f64]> {
        let mut losses = Vec::with_capacity(num_epochs);
        
        for _ in 0..num_epochs {
            let loss = self.train_epoch(training_data, learning_rate, batch_size);
            losses.push(loss);
        }
        
        losses.into_boxed_slice()
    }

    
    pub fn train_batch(&mut self, training_data: &JsValue, epochs: usize, learning_rate: f64, canvas_id: &str, callback: &Function, batch_size: usize) {
        let mut losses = Vec::new();
        let max_log_entries = min(epochs, 200);
        let log_interval = epochs / max_log_entries;

        for epoch in 0..epochs {
            // Train a single epoch and get the loss
            let avg_loss = self.train_epoch(training_data, learning_rate, batch_size);
            losses.push(avg_loss);

            // Plot the loss chart on the final epoch
            if epoch == epochs - 1 {
                plot_loss(canvas_id, &losses);
            }
            
            // Send progress updates at regular intervals
            if epoch % log_interval == 0 {
                // Create a progress object with more information
                let progress = format!("{{\"epoch\":{},\"totalEpochs\":{},\"loss\":{:.4},\"progress\":{:.2}}}", 
                    epoch + 1, epochs, avg_loss, (epoch + 1) as f64 / epochs as f64 * 100.0);
                
                let this = JsValue::null();
                let _ = callback.call1(&this, &JsValue::from_str(&progress));
            }
        }
    }

    pub fn calculate_accuracy(&mut self, test_data: &JsValue) -> f64 {
        let data: Vec<TrainingData> = serde_wasm_bindgen::from_value(test_data.clone()).unwrap();
        let mut correct_predictions = 0;

        for d in data.iter() {
            let prediction = self.predict(&d.inputs);
            let predicted_class = if prediction[0] > 0.5 { 1.0 } else { 0.0 };
            if (predicted_class - d.label[0]).abs() < f64::EPSILON {
                correct_predictions += 1;
            }
        }

        correct_predictions as f64 / data.len() as f64
    }

    pub fn predict(&mut self, inputs: &[f64]) -> Vec<f64> {
        self.forward(inputs)
    }

    #[wasm_bindgen]
    pub fn forward_collect(&mut self, inputs: &[f64]) -> JsValue {
        let mut activations = Vec::new();
        let mut current_inputs = inputs.to_vec();
        for layer in &mut self.layers {
            current_inputs = layer.forward(&current_inputs);
            activations.push(current_inputs.clone());
        }
        serde_wasm_bindgen::to_value(&activations).unwrap()
    }

    fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        let mut current_inputs = inputs.to_vec();
        for layer in &mut self.layers {
            current_inputs = layer.forward(&current_inputs);
        }
        current_inputs
    }

    fn calculate_loss(&self, outputs: &[f64], targets: &[f64]) -> f64 {
        match self.loss_function {
            LossFunction::MSE => {
                let mut error = 0.0;
                for i in 0..outputs.len() {
                    error += (targets[i] - outputs[i]).powi(2);
                }
                error / outputs.len() as f64
            },
            LossFunction::BCE => {
                let mut error = 0.0;
                for i in 0..outputs.len() {
                    // Clamp outputs to avoid log(0)
                    let o = outputs[i].clamp(1e-7, 1.0 - 1e-7);
                    error += -(targets[i] * o.ln() + (1.0 - targets[i]) * (1.0 - o).ln());
                }
                error / outputs.len() as f64
            }
        }
    }

    fn backward(&mut self, original_inputs: &[f64], outputs: &[f64], targets: &[f64], learning_rate: f64) {
        let mut errors = Vec::with_capacity(outputs.len());
        match self.loss_function {
            LossFunction::MSE => {
                for i in 0..outputs.len() {
                    errors.push(targets[i] - outputs[i]);
                }
            },
            LossFunction::BCE => {
                for i in 0..outputs.len() {
                    // Clamp outputs to avoid division by zero
                    let o = outputs[i].clamp(1e-7, 1.0 - 1e-7);
                    errors.push((targets[i] - o) / (o * (1.0 - o)));
                }
            }
        }
        for i in (0..self.layers.len()).rev() {
            let inputs = if i == 0 {
                original_inputs.to_vec()
            } else {
                self.layers[i - 1].outputs.clone()
            };
            errors = self.layers[i].backward(&inputs, &errors, learning_rate);
        }
    }
    
    
    fn reset_gradients(&mut self) {
        for layer in &mut self.layers {
            layer.reset_gradients();
        }
    }
    
    
    fn apply_gradients(&mut self, learning_rate: f64) {
        for layer in &mut self.layers {
            layer.apply_gradients(learning_rate);
        }
    }

    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|layer| {
            let w = layer.weights.len() * layer.weights[0].len();
            let b = layer.biases.len();
            w + b
        }).sum()
    }

    
    pub fn get_summary(&mut self, test_data: &JsValue) -> JsValue {
        let accuracy = self.calculate_accuracy(test_data);
        // For loss, use the test set and average over all samples
        let data: Vec<TrainingData> = serde_wasm_bindgen::from_value(test_data.clone()).unwrap();
        let mut total_loss = 0.0;
        for d in &data {
            let outputs = self.predict(&d.inputs);
            total_loss += self.calculate_loss(&outputs, &d.label);
        }
        let final_loss = total_loss / data.len() as f64;
        let num_parameters = self.num_parameters();
        let summary = ModelSummary { accuracy, final_loss, num_parameters };
        serde_wasm_bindgen::to_value(&summary).unwrap()
    }
}

#[wasm_bindgen]
pub fn plot_results(canvas_id: &str, mlp: &mut MLP, test_data: &JsValue, mask_str_js: &JsValue) {
    let data: Vec<TrainingData> = serde_wasm_bindgen::from_value(test_data.clone()).unwrap();
    let mask_str = mask_str_js.as_string().unwrap_or_default();

    let backend = CanvasBackend::new(canvas_id).expect("cannot find canvas");
    let root = backend.into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Model Predictions", ("sans-serif", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1f64..1f64, -1f64..1f64) // Updated domain
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart.draw_series(data.iter().map(|d| {
        let prediction = mlp.predict(&d.inputs);
        let predicted_class = if prediction[0] > 0.5 { 1.0 } else { 0.0 };
        let true_class = d.label[0];

        let wrong = predicted_class != true_class;
        let color = if predicted_class == true_class {
            if predicted_class > 0.5 { &GREEN } else { &BLUE }
        } else {
            &RED
        };
        if wrong {
            Circle::new((d.inputs[0], d.inputs[1]), 4, color.filled())
        } else {
            Circle::new((d.inputs[0], d.inputs[1]), 2, color.filled())
        }
    })).unwrap();
}


struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    outputs: Vec<f64>,
    activation: Activation,
    // Gradient accumulators
    weight_grads: Vec<Vec<f64>>,
    bias_grads: Vec<f64>,
    batch_count: usize,
}

impl Layer {
    
    fn new(input_size: usize, output_size: usize, activation: Activation) -> Layer {
        let mut weights = Vec::with_capacity(output_size);
        let mut weight_grads = Vec::with_capacity(output_size);
        
        GLOBAL_RNG.with(|rng_cell| {
            let mut rng = rng_cell.borrow_mut();
            
            for _ in 0..output_size {
                let mut row = Vec::with_capacity(input_size);
                let mut grad_row = Vec::with_capacity(input_size);
                for _ in 0..input_size {
                    row.push(rng.gen_range(-1.0..1.0));
                    grad_row.push(0.0); // Initialize gradient accumulators to zero
                }
                weights.push(row);
                weight_grads.push(grad_row);
            }
        });
        
        let biases = vec![0.0; output_size];
        let bias_grads = vec![0.0; output_size];
        
        Layer {
            weights,
            biases,
            outputs: vec![0.0; output_size],
            activation,
            weight_grads,
            bias_grads,
            batch_count: 0,
        }
    }

    // Update Layer::forward to use the correct activation
    fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        let mut new_outputs = Vec::with_capacity(self.weights.len());
        for (i, weights_row) in self.weights.iter().enumerate() {
            let mut output = 0.0;
            for (j, weight) in weights_row.iter().enumerate() {
                output += inputs[j] * weight;
            }
            output += self.biases[i];
            let activated = match self.activation {
                Activation::ReLU => relu(output),
                Activation::Sigmoid => sigmoid(output),
            };
            new_outputs.push(activated);
        }
        self.outputs = new_outputs.clone();
        new_outputs
    }

    // Update Layer::backward to accumulate gradients without updating weights
    fn backward(&mut self, inputs: &[f64], errors: &[f64], learning_rate: f64) -> Vec<f64> {
        let mut next_errors = vec![0.0; self.weights[0].len()];

        for i in 0..self.weights.len() {
            let error = errors[i];
            let output = self.outputs[i];
            let delta = error * match self.activation {
                Activation::ReLU => relu_derivative(output),
                Activation::Sigmoid => sigmoid_derivative(output),
            };

            for j in 0..self.weights[i].len() {
                next_errors[j] += self.weights[i][j] * delta;
                // Accumulate gradients instead of updating weights directly
                self.weight_grads[i][j] += delta * inputs[j];
            }
            // Accumulate bias gradients
            self.bias_grads[i] += delta;
        }
        
        // Increment batch count
        self.batch_count += 1;

        next_errors
    }
    
    
    fn apply_gradients(&mut self, learning_rate: f64) {
        if self.batch_count == 0 {
            return; // No gradients to apply
        }
        
        // Calculate effective learning rate (scaled by batch size)
        let effective_lr = learning_rate / self.batch_count as f64;
        
        // Apply accumulated gradients to weights and biases
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] += effective_lr * self.weight_grads[i][j];
                self.weight_grads[i][j] = 0.0; // Reset gradient
            }
            self.biases[i] += effective_lr * self.bias_grads[i];
            self.bias_grads[i] = 0.0; // Reset gradient
        }
        
        // Reset batch count
        self.batch_count = 0;
    }
    
    
    fn reset_gradients(&mut self) {
        for i in 0..self.weight_grads.len() {
            for j in 0..self.weight_grads[i].len() {
                self.weight_grads[i][j] = 0.0;
            }
            self.bias_grads[i] = 0.0;
        }
        self.batch_count = 0;
    }
}


fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

#[wasm_bindgen]
pub fn plot_mask(canvas_id: &str, mask_str_js: &JsValue) {
    let mask_str = mask_str_js.as_string().unwrap_or_default();
    let backend = CanvasBackend::new(canvas_id).expect("cannot find canvas");
    let root = backend.into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Mask Visualization", ("sans-serif", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1f64..1f64, -1f64..1f64)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    let grid_size = 200;
    let step = 2.0 / (grid_size as f64);
    let mut points_inside = Vec::new();
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = -1.0 + (i as f64) * step;
            let y = -1.0 + (j as f64) * step;
            if is_point_in_mask(x, y, &mask_str) {
                points_inside.push((x, y));
            }
        }
    }
    chart.draw_series(points_inside.iter().map(|&(x, y)| {
        Circle::new((x, y), 2, GREEN.filled())
    })).unwrap();
}
