// Sample word embeddings data (2D for visualization purposes)
// In a real application, this would come from a model like Word2Vec, GloVe, or BERT
// Values are adjusted to better represent semantic relationships

const sampleEmbeddings = [
  // Royalty terms
  { word: "king", x: 0.85, y: 0.75, category: "royalty" },
  { word: "queen", x: 0.78, y: 0.72, category: "royalty" },
  { word: "prince", x: 0.82, y: 0.68, category: "royalty" },
  { word: "princess", x: 0.75, y: 0.65, category: "royalty" },
  
  // Gender terms
  { word: "man", x: 0.80, y: 0.30, category: "people" },
  { word: "woman", x: 0.72, y: 0.35, category: "people" },
  { word: "boy", x: 0.75, y: 0.25, category: "people" },
  { word: "girl", x: 0.68, y: 0.30, category: "people" },
  
  // Animals
  { word: "cat", x: 0.20, y: 0.80, category: "animals" },
  { word: "dog", x: 0.25, y: 0.78, category: "animals" },
  { word: "bird", x: 0.30, y: 0.75, category: "animals" },
  { word: "fish", x: 0.15, y: 0.82, category: "animals" },
  
  // Transportation
  { word: "car", x: 0.50, y: 0.15, category: "transport" },
  { word: "bus", x: 0.55, y: 0.20, category: "transport" },
  { word: "train", x: 0.45, y: 0.25, category: "transport" },
  { word: "bike", x: 0.40, y: 0.10, category: "transport" },
  
  // Emotions
  { word: "happy", x: 0.90, y: 0.85, category: "emotions" },
  { word: "sad", x: 0.20, y: 0.85, category: "emotions" },
  { word: "angry", x: 0.25, y: 0.15, category: "emotions" },
  { word: "calm", x: 0.85, y: 0.15, category: "emotions" },
  
  // Size terms
  { word: "big", x: 0.90, y: 0.50, category: "size" },
  { word: "small", x: 0.20, y: 0.50, category: "size" },
  { word: "large", x: 0.85, y: 0.55, category: "size" },
  { word: "tiny", x: 0.15, y: 0.45, category: "size" },
  
  // Speed terms
  { word: "fast", x: 0.85, y: 0.25, category: "speed" },
  { word: "slow", x: 0.25, y: 0.25, category: "speed" },
  { word: "quick", x: 0.80, y: 0.30, category: "speed" },
  { word: "sluggish", x: 0.30, y: 0.20, category: "speed" },
  
  // Temperature terms
  { word: "hot", x: 0.90, y: 0.70, category: "temperature" },
  { word: "cold", x: 0.20, y: 0.70, category: "temperature" },
  { word: "warm", x: 0.75, y: 0.65, category: "temperature" },
  { word: "cool", x: 0.35, y: 0.65, category: "temperature" },
  
  // Countries
  { word: "france", x: 0.60, y: 0.90, category: "countries" },
  { word: "germany", x: 0.55, y: 0.88, category: "countries" },
  { word: "italy", x: 0.65, y: 0.85, category: "countries" },
  { word: "spain", x: 0.62, y: 0.82, category: "countries" },
  
  // Capitals
  { word: "paris", x: 0.62, y: 0.92, category: "capitals" },
  { word: "berlin", x: 0.57, y: 0.90, category: "capitals" },
  { word: "rome", x: 0.67, y: 0.87, category: "capitals" },
  { word: "madrid", x: 0.64, y: 0.84, category: "capitals" }
];

export default sampleEmbeddings;