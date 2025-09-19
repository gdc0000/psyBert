import React, { useState, useEffect } from 'react';
import { Paper, Typography, TextField, Button, Box, List, ListItem, ListItemText, Divider, Chip, Grid } from '@mui/material';

const SimilarityDemo = ({ embeddings }) => {
  const [word1, setWord1] = useState('');
  const [word2, setWord2] = useState('');
  const [similarity, setSimilarity] = useState(null);
  const [analogies, setAnalogies] = useState([]);
  const [analogyInput, setAnalogyInput] = useState({ a: '', b: '', c: '' });

  // Calculate cosine similarity between two vectors
  const cosineSimilarity = (vec1, vec2) => {
    // For this demo, we'll use the x,y coordinates as our vectors
    const dotProduct = vec1.x * vec2.x + vec1.y * vec2.y;
    const magnitude1 = Math.sqrt(vec1.x * vec1.x + vec1.y * vec1.y);
    const magnitude2 = Math.sqrt(vec2.x * vec2.x + vec2.y * vec2.y);
    
    if (magnitude1 === 0 || magnitude2 === 0) return 0;
    
    return dotProduct / (magnitude1 * magnitude2);
  };

  // Find embedding by word
  const findEmbedding = (word) => {
    return embeddings.find(e => e.word.toLowerCase() === word.toLowerCase());
  };

  // Handle similarity calculation
  const handleCalculateSimilarity = () => {
    const emb1 = findEmbedding(word1);
    const emb2 = findEmbedding(word2);
    
    if (emb1 && emb2) {
      const sim = cosineSimilarity(emb1, emb2);
      setSimilarity(sim.toFixed(4));
    } else {
      setSimilarity("Word not found in embeddings");
    }
  };

  // Handle analogy calculation (a is to b as c is to ?)
  const handleCalculateAnalogy = () => {
    const embA = findEmbedding(analogyInput.a);
    const embB = findEmbedding(analogyInput.b);
    const embC = findEmbedding(analogyInput.c);
    
    if (embA && embB && embC) {
      // Calculate the vector for the analogy: result = embB - embA + embC
      const resultVector = {
        x: embB.x - embA.x + embC.x,
        y: embB.y - embA.y + embC.y
      };
      
      // Find the closest word to this result vector (excluding the input words)
      let closestWord = null;
      let maxSimilarity = -Infinity;
      const excludeWords = [analogyInput.a.toLowerCase(), analogyInput.b.toLowerCase(), analogyInput.c.toLowerCase()];
      
      embeddings.forEach(emb => {
        // Skip if this is one of the input words
        if (excludeWords.includes(emb.word.toLowerCase())) return;
        
        // Calculate similarity between result vector and this embedding
        const sim = cosineSimilarity(resultVector, emb);
        
        if (sim > maxSimilarity) {
          maxSimilarity = sim;
          closestWord = emb;
        }
      });
      
      if (closestWord) {
        setAnalogies([
          ...analogies,
          {
            a: analogyInput.a,
            b: analogyInput.b,
            c: analogyInput.c,
            result: closestWord.word,
            similarity: maxSimilarity.toFixed(4),
            id: Date.now()
          }
        ]);
      } else {
        alert("Could not find a suitable analogy result");
      }
    } else {
      alert("One or more words not found in embeddings");
    }
  };

  // Predefined examples
  const similarityExamples = [
    { word1: "king", word2: "queen" },
    { word1: "man", word2: "woman" },
    { word1: "cat", word2: "dog" },
    { word1: "car", word2: "bus" }
  ];

  const analogyExamples = [
    { a: "king", b: "man", c: "woman" },
    { a: "man", b: "woman", c: "king" },
    { a: "france", b: "paris", c: "italy" },
    { a: "big", b: "small", c: "hot" }
  ];

  // Function to load example similarity
  const loadSimilarityExample = (example) => {
    setWord1(example.word1);
    setWord2(example.word2);
  };

  // Function to load example analogy
  const loadAnalogyExample = (example) => {
    setAnalogyInput({
      a: example.a,
      b: example.b,
      c: example.c
    });
  };

  // Load a famous example on component mount
  useEffect(() => {
    setAnalogyInput({
      a: "king",
      b: "man",
      c: "woman"
    });
  }, []);

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Word Similarity and Analogies
      </Typography>
      
      <Box sx={{ mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Word Similarity Calculator
        </Typography>
        <Typography variant="body1" paragraph>
          Calculate the cosine similarity between two words based on their embeddings.
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2, flexWrap: 'wrap' }}>
          <TextField
            label="Word 1"
            value={word1}
            onChange={(e) => setWord1(e.target.value)}
            variant="outlined"
            size="small"
          />
          <Typography>and</Typography>
          <TextField
            label="Word 2"
            value={word2}
            onChange={(e) => setWord2(e.target.value)}
            variant="outlined"
            size="small"
          />
          <Button variant="contained" onClick={handleCalculateSimilarity}>
            Calculate Similarity
          </Button>
        </Box>
        
        {similarity !== null && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6">
              Similarity: {similarity}
            </Typography>
            <Typography variant="body2">
              (1 means identical, 0 means orthogonal, -1 means opposite)
            </Typography>
          </Box>
        )}
        
        <Box sx={{ mt: 2 }}>
          <Typography variant="body1" gutterBottom>
            Try these examples:
          </Typography>
          <Grid container spacing={1}>
            {similarityExamples.map((example, index) => (
              <Grid item key={index}>
                <Chip 
                  label={`${example.word1} - ${example.word2}`} 
                  onClick={() => loadSimilarityExample(example)}
                  variant="outlined"
                />
              </Grid>
            ))}
          </Grid>
        </Box>
      </Box>
      
      <Divider sx={{ my: 3 }} />
      
      <Box>
        <Typography variant="h6" gutterBottom>
          Word Analogy Solver
        </Typography>
        <Typography variant="body1" paragraph>
          Solve analogies of the form "A is to B as C is to ?" using vector arithmetic.
          <br />
          <strong>Try the famous example: king - man + woman = queen</strong>
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2, flexWrap: 'wrap' }}>
          <TextField
            label="A"
            value={analogyInput.a}
            onChange={(e) => setAnalogyInput({...analogyInput, a: e.target.value})}
            variant="outlined"
            size="small"
          />
          <Typography>is to</Typography>
          <TextField
            label="B"
            value={analogyInput.b}
            onChange={(e) => setAnalogyInput({...analogyInput, b: e.target.value})}
            variant="outlined"
            size="small"
          />
          <Typography>as</Typography>
          <TextField
            label="C"
            value={analogyInput.c}
            onChange={(e) => setAnalogyInput({...analogyInput, c: e.target.value})}
            variant="outlined"
            size="small"
          />
          <Typography>is to ?</Typography>
          <Button variant="contained" onClick={handleCalculateAnalogy}>
            Solve
          </Button>
        </Box>
        
        <Box sx={{ mt: 2 }}>
          <Typography variant="body1" gutterBottom>
            Try these examples:
          </Typography>
          <Grid container spacing={1}>
            {analogyExamples.map((example, index) => (
              <Grid item key={index}>
                <Chip 
                  label={`${example.a} : ${example.b} :: ${example.c} : ?`} 
                  onClick={() => loadAnalogyExample(example)}
                  variant="outlined"
                />
              </Grid>
            ))}
          </Grid>
        </Box>
        
        {analogies.length > 0 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6" gutterBottom>
              Results:
            </Typography>
            <List>
              {analogies.map((analogy) => (
                <ListItem key={analogy.id} sx={{ py: 1 }}>
                  <ListItemText 
                    primary={`${analogy.a} is to ${analogy.b} as ${analogy.c} is to ${analogy.result}`}
                    secondary={`Similarity: ${analogy.similarity}`}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        )}
      </Box>
    </Paper>
  );
};

export default SimilarityDemo;