import React from 'react';
import { Paper, Typography, Box, Chip } from '@mui/material';

const EducationalContent = () => {
  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Understanding Word Embeddings
      </Typography>
      
      <Typography variant="body1" paragraph>
        Word embeddings are a type of word representation that allows words with similar meanings to have similar representations.
        They are a distributed representation for text that is used in natural language processing (NLP) tasks.
      </Typography>
      
      <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
        How Word Embeddings Work
      </Typography>
      
      <Typography variant="body1" paragraph>
        Traditional approaches to natural language processing represented words as discrete symbols (e.g., "cat" = 1, "dog" = 2).
        Word embeddings, in contrast, represent words as continuous vectors in a high-dimensional space.
      </Typography>
      
      <Typography variant="body1" paragraph>
        The key idea is that words appearing in similar contexts will have similar vector representations.
        This is captured by the distributional hypothesis: "You shall know a word by the company it keeps."
      </Typography>
      
      <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
        Key Properties
      </Typography>
      
      <Box component="ul" sx={{ pl: 3 }}>
        <li>
          <Typography variant="body1">
            <strong>Semantic Similarity:</strong> Words with similar meanings are positioned close to each other in the vector space.
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            <strong>Analogy Relationships:</strong> Vector arithmetic can capture semantic relationships (e.g., king - man + woman ≈ queen).
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            <strong>Dense Representation:</strong> Unlike one-hot encodings, embeddings are dense vectors with mostly non-zero values.
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            <strong>Dimensionality:</strong> Typically 50-300 dimensions, much smaller than vocabulary size.
          </Typography>
        </li>
      </Box>
      
      <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
        Famous Example: King - Man + Woman = Queen
      </Typography>
      
      <Typography variant="body1" paragraph>
        One of the most famous demonstrations of word embeddings is the analogy relationship:
      </Typography>
      
      <Box sx={{ p: 2, bgcolor: 'grey.100', borderRadius: 1, mb: 2 }}>
        <Typography variant="body1" sx={{ fontFamily: 'monospace', textAlign: 'center' }}>
          vector("king") - vector("man") + vector("woman") ≈ vector("queen")
        </Typography>
      </Box>
      
      <Typography variant="body1" paragraph>
        This works because the vector difference between "king" and "man" captures the concept of "royalty" or "gender role",
        which when added to "woman" results in a vector close to "queen". This demonstrates that embeddings can capture
        not just semantic similarity but also semantic relationships.
      </Typography>
      
      <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
        Popular Models
      </Typography>
      
      <Box component="ul" sx={{ pl: 3 }}>
        <li>
          <Typography variant="body1">
            <strong>Word2Vec:</strong> Uses shallow neural networks to learn embeddings, with architectures like Skip-gram and CBOW.
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            <strong>GloVe:</strong> Combines global matrix factorization and local context window methods.
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            <strong>FastText:</strong> Extends Word2Vec by representing words as bags of character n-grams.
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            <strong>BERT:</strong> Contextual embeddings that generate different representations based on context.
          </Typography>
        </li>
      </Box>
      
      <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
        Applications
      </Typography>
      
      <Box component="ul" sx={{ pl: 3 }}>
        <li>
          <Typography variant="body1">
            Machine translation
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            Sentiment analysis
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            Information retrieval
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            Question answering systems
          </Typography>
        </li>
        <li>
          <Typography variant="body1">
            Text summarization
          </Typography>
        </li>
      </Box>
    </Paper>
  );
};

export default EducationalContent;