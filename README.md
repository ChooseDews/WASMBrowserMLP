# MLP Classifier Demo | Neural Network Visualization in WebAssembly

An interactive 2D binary classification demo using a Multi-Layer Perceptron (MLP) implemented in Rust and compiled to WebAssembly for optimal performance.

![MLP Demo Screenshot](./public/mlp.png)

## Features

- Interactive neural network training visualization
- Custom network architecture configuration
- Real-time decision boundary plotting
- WebAssembly-powered for high performance
- Customizable training masks and parameters

## Tech Stack

- **Frontend**: Vue.js 3 with Vite
- **Core ML**: Rust compiled to WebAssembly
- **Plotting**: Rust Plotters Library with Canvas Backend

## Prerequisites

- [Node.js](https://nodejs.org/) (v16 or later)
- [Rust](https://www.rust-lang.org/tools/install) (stable toolchain)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

## Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/ChooseDews/WASMBrowserMLP.git
   cd WASMBrowserMLP
   ```

2. **Install dependencies**

   ```bash
   npm install
   ```

3. **Build the WebAssembly module**

   ```bash
   npm run build:wasm
   ```

4. **Start the development server**

   ```bash
   npm run dev
   ```

   The application will be available at http://localhost:5173

## Building for Production

```bash
npm run build
```

This will:
1. Compile the Rust code to WebAssembly
2. Build the Vue.js application with Vite

The output will be in the `dist` directory, ready for deployment.

## Deployment

This project is configured to deploy automatically to GitHub Pages using GitHub Actions. When you push to the `main` branch, the workflow will:

1. Build the WebAssembly module and Vue.js application
2. Deploy the built files to GitHub Pages

You can also manually trigger the deployment from the Actions tab in your GitHub repository.

## Project Structure

- `/rust_lib/` - Rust implementation of the neural network
- `/src/` - Vue.js frontend code
- `/public/` - Static assets
- `/.github/workflows/` - GitHub Actions workflow for deployment