import React from 'react';
import { Paper, Typography, Box } from '@mui/material';

const FamousExamples = () => {
  return (
    <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
      <Typography variant="h5" gutterBottom>
        Famous Word Embedding Examples
      </Typography>
      
      <Typography variant="body1" paragraph>
        These examples demonstrate the remarkable ability of word embeddings to capture semantic relationships through vector arithmetic.
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Gender Relationships
        </Typography>
        <Box sx={{ p: 2, bgcolor: 'grey.100', borderRadius: 1, mb: 1 }}>
          <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
            king - man + woman = queen
          </Typography>
        </Box>
        <Typography variant="body2">
          This classic example shows how embeddings can capture gender-role relationships.
        </Typography>
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Capital City Relationships
        </Typography>
        <Box sx={{ p: 2, bgcolor: 'grey.100', borderRadius: 1, mb: 1 }}>
          <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
            France - Paris + Rome = Italy
          </Typography>
        </Box>
        <Typography variant="body2">
          Embeddings can capture geographical and political relationships between countries and their capitals.
        </Typography>
      </Box>
      
      <Box>
        <Typography variant="h6" gutterBottom>
          Verb Tense Relationships
        </Typography>
        <Box sx={{ p: 2, bgcolor: 'grey.100', borderRadius: 1, mb: 1 }}>
          <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
            walking - walked + swimming = swam
          </Typography>
        </Box>
        <Typography variant="body2">
          Embeddings can even capture grammatical relationships like verb tenses.
        </Typography>
      </Box>
    </Paper>
  );
};

export default FamousExamples;