import React, { useState } from 'react';
import EmbeddingVisualization from './EmbeddingVisualization';
import EducationalContent from './EducationalContent';
import SimilarityDemo from './SimilarityDemo';
import FamousExamples from './FamousExamples';
import sampleEmbeddings from '../data/sampleEmbeddings';
import { Container, Typography, Box, Tabs, Tab, AppBar } from '@mui/material';

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <div>
      <AppBar position="static">
        <Typography variant="h4" align="center" sx={{ py: 2 }}>
          Word Embeddings Explorer
        </Typography>
      </AppBar>
      
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
          <Tabs value={activeTab} onChange={handleTabChange} variant="fullWidth">
            <Tab label="Visualization" />
            <Tab label="How It Works" />
            <Tab label="Similarity & Analogies" />
          </Tabs>
        </Box>

        {activeTab === 0 && (
          <EmbeddingVisualization embeddings={sampleEmbeddings} />
        )}
        
        {activeTab === 1 && (
          <div>
            <EducationalContent />
            <FamousExamples />
          </div>
        )}
        
        {activeTab === 2 && (
          <SimilarityDemo embeddings={sampleEmbeddings} />
        )}
      </Container>
    </div>
  );
};

export default Dashboard;