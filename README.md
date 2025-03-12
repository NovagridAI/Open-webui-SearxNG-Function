# OpenWebUI with Searxng and DeepSeek R1 Integration

This project enhances [OpenWebUI](https://github.com/open-webui/open-webui) by integrating Searxng for web search capabilities and improving the DeepSeek R1 model's functionality. It focuses on seamless search result injection into the context and displaying the model's chain of thought, with fixes for specific issues in versions 0.5.6 and above.

## Features

- **Searxng Integration**: Adds support for Searxng to fetch and insert search results directly into the context without needing to access individual pages.
- **DeepSeek R1 Chain of Thought**: Displays the reasoning process of the DeepSeek R1 model during content generation (supported in v0.5.6+).
- **Bug Fixes**:
  - Resolves the issue where DeepSeek R1 generated content lacks the opening `<think>` tag.
  - Fixes errors in web search and title generation.
  - Implements logic to prevent adding `<think>` tags during non-streaming generation.

## Requirements

- OpenWebUI version **0.5.6 or higher**.
- Searxng installed and configured for web search functionality.
- DeepSeek R1 model integration enabled.
