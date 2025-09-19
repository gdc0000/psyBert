import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Paper, Typography, Box, TextField, Button, FormControl, InputLabel, Select, MenuItem } from '@mui/material';

const EmbeddingVisualization = ({ embeddings }) => {
  const svgRef = useRef();
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');

  // Get unique categories
  const categories = ['all', ...new Set(embeddings.map(d => d.category))];

  // Filter embeddings based on category and search term
  const filteredEmbeddings = embeddings.filter(emb => {
    const matchesCategory = selectedCategory === 'all' || emb.category === selectedCategory;
    const matchesSearch = searchTerm === '' || emb.word.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  useEffect(() => {
    if (!embeddings || embeddings.length === 0) return;

    // Set up dimensions
    const width = 800;
    const height = 500;
    const margin = { top: 40, right: 40, bottom: 40, left: 40 };

    // Clear previous SVG content
    d3.select(svgRef.current).selectAll("*").remove();

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, 1])
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height - margin.bottom, margin.top]);

    // Add X axis
    svg.append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale));

    // Add Y axis
    svg.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale));

    // Add axis labels
    svg.append("text")
      .attr("text-anchor", "middle")
      .attr("x", width / 2)
      .attr("y", height - 5)
      .text("Dimension 1");

    svg.append("text")
      .attr("text-anchor", "middle")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", 15)
      .text("Dimension 2");

    // Add title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text("Word Embeddings Visualization");

    // Color scale for categories
    const colorScale = d3.scaleOrdinal()
      .domain(categories.filter(c => c !== 'all'))
      .range(d3.schemeCategory10);

    // Add dots
    const circles = svg.append('g')
      .selectAll("dot")
      .data(filteredEmbeddings)
      .enter()
      .append("circle")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", 8)
      .style("fill", d => colorScale(d.category))
      .style("cursor", "pointer")
      .on("mouseover", function(event, d) {
        d3.select(this).attr("r", 12);
        // Show tooltip
        svg.append("text")
          .attr("id", "tooltip")
          .attr("x", xScale(d.x))
          .attr("y", yScale(d.y) - 15)
          .attr("text-anchor", "middle")
          .style("font-size", "12px")
          .style("font-weight", "bold")
          .text(d.word);
      })
      .on("mouseout", function() {
        d3.select(this).attr("r", 8);
        // Remove tooltip
        d3.select("#tooltip").remove();
      });

    // Add labels
    svg.append('g')
      .selectAll("text")
      .data(filteredEmbeddings)
      .enter()
      .append("text")
      .attr("x", d => xScale(d.x))
      .attr("y", d => yScale(d.y) + 20)
      .attr("text-anchor", "middle")
      .style("font-size", "10px")
      .text(d => d.word);

    // Add legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width - 150}, 50)`);

    const legendCategories = categories.filter(c => c !== 'all');
    legendCategories.forEach((category, i) => {
      legend.append("circle")
        .attr("cx", 0)
        .attr("cy", i * 20)
        .attr("r", 6)
        .style("fill", colorScale(category));

      legend.append("text")
        .attr("x", 15)
        .attr("y", i * 20 + 5)
        .text(category)
        .style("font-size", "12px")
        .attr("alignment-baseline", "middle");
    });

  }, [embeddings, filteredEmbeddings, categories, selectedCategory, searchTerm]);

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        2D Visualization of Word Embeddings
      </Typography>
      <Typography variant="body1" paragraph>
        This visualization shows how words with similar meanings are positioned close to each other in the embedding space.
        Each point represents a word, and colors indicate semantic categories.
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}>
        <TextField
          label="Search words"
          variant="outlined"
          size="small"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Category</InputLabel>
          <Select
            value={selectedCategory}
            label="Category"
            onChange={(e) => setSelectedCategory(e.target.value)}
          >
            {categories.map(category => (
              <MenuItem key={category} value={category}>
                {category.charAt(0).toUpperCase() + category.slice(1)}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
      
      <Box sx={{ overflowX: 'auto' }}>
        <svg ref={svgRef}></svg>
      </Box>
      <Typography variant="body2" sx={{ mt: 2 }}>
        <strong>Note:</strong> In real applications, word embeddings exist in high-dimensional spaces (often 100-300 dimensions).
        This 2D representation is for educational purposes and shows a simplified projection.
      </Typography>
    </Paper>
  );
};

export default EmbeddingVisualization;