# CCStory: Context-Consistent Storytelling with Controlled Character Synthesis
This repository contains the official implementation of the paper "CCStory", currently submitted to The Visual Computer.

## 📄 Publication
Journal: The Visual Computer

Key Focus: Character consistency across sequential frames and precise layout-to-image synthesis. (especially Story Visualization Model)

## 🛠 Framework & Acknowledgments
CCStory builds upon and references the following works for specific modules:

- AutoStory: Referenced for the prompt generation template.
- Gen4Gen: Referenced for its multi-concept synthesis logic.

## 🚀 Key Contributions & Implementation
CStory focuses on high-fidelity storytelling where the protagonist's identity and the scene's context remain stable throughout the sequence.

1. Character Identity Injection
We utilize a dedicated Character Prompt to extract and stabilize the protagonist's features. By injecting this identity into the diffusion process, CCStory ensures the character remains recognizable across different frames, regardless of background or action changes.

2. Context-Consistent Diffusion
The architecture facilitates a seamless integration between the Story Prompt and the Diffusion Model, ensuring that each generated frame remains faithful to the overarching narrative context.

3. Integrated Synthesis Pipeline
The core architecture bridges the gap between text-based story prompts and visual consistency, allowing for a seamless transition between frames while preserving the "Context-Consistent" nature of the narrative.

## 🧠 Architecture Workflow
- Input: User provides a Story Prompt and a Character Prompt.
- Structuring: The system generates a Storyboard (visual layout) that aligns with the story's progression.
- Synthesis: The Diffusion model combines the Character Identity with the Storyboard to produce a final, context-consistent story image.

## 📋 Requirements
Detailed dependencies are listed in requirements.txt.

To install:

```Bash
pip install -r requirements.txt
```

## 🔢 DOI & Data Notice
Code DOI: [![DOI](https://zenodo.org/badge/1012396713.svg)](https://doi.org/10.5281/zenodo.18466684)

Data Notice: Training datasets and specific character weights are private. This repository includes the inference code and the core architectural implementation.

## 📚 References
% CCStory (Submitted to The Visual Computer)
```bibtex
@article{ccstory2026,
  title={CCStory: Context-Consistent Storytelling with Controlled Character Synthesis},
  author={Yang, Hyemin and others},
  journal={The Visual Computer (Under Review)},
  year={2026}
}
```
```bibtex
% Referenced Frameworks
@article{shao2023autostory,
  title={AutoStory: Generating Multi-frame Consistent Storytelling with Text-to-Image Models},
  author={Shao, Wenqi and others},
  journal={arXiv preprint arXiv:2311.11243},
  year={2023}
}
```
```bibtex
@article{yen2024gen4gen,
  title={Gen4Gen: Generative Data Pipeline for Generative Multi-Concept Synthesis},
  author={Yen, Louis and others},
  journal={arXiv preprint arXiv:2402.11130},
  year={2024}
}
```
