# Word Embeddings Educational Dashboard

An interactive dashboard to explore and understand word embeddings, including visualization, similarity calculations, and analogy demonstrations.

## Features

- **Interactive Visualization**: 2D visualization of word embeddings with category coloring
- **Word Similarity Calculator**: Calculate cosine similarity between any two words
- **Analogy Solver**: Solve word analogies using vector arithmetic (e.g., king - man + woman = queen)
- **Educational Content**: Detailed explanations of word embeddings and their properties
- **Famous Examples**: Showcase of well-known word embedding relationships

## Getting Started

### Prerequisites

- Node.js (version 12 or higher)
- npm (usually comes with Node.js)

### Installation

1. Clone the repository (if not already cloned)
2. Navigate to the project directory:
   ```
   cd word-embedding-dashboard
   ```
3. Install dependencies:
   ```
   npm install
   ```

### Running the Dashboard

To start the development server:
```
npm start
```

The dashboard will be available at `http://localhost:3000`

## Using the Dashboard

1. **Visualization Tab**: Explore the 2D word embedding space. Use the search and filter controls to focus on specific words or categories.

2. **How It Works Tab**: Learn about word embeddings through educational content and see famous examples of vector arithmetic.

3. **Similarity & Analogies Tab**: 
   - Calculate similarity between words using the similarity calculator
   - Solve word analogies using the analogy solver
   - Try pre-loaded examples or enter your own word combinations

## Example Interactions

Try these famous examples in the "Similarity & Analogies" tab:

- Similarity: "king" and "queen"
- Analogy: "king" is to "man" as "woman" is to ? (should result in "queen")

## Technology Stack

- **React**: Frontend framework
- **D3.js**: Data visualization library
- **Material-UI**: UI component library
- **CSS3**: Styling

## Project Structure

```
src/
├── components/          # React components
├── data/               # Sample data files
├── App.js              # Main application component
└── index.js            # Entry point
```