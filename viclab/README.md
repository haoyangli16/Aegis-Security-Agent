# viclab

**viclab** (Video & Voice Intelligence Center, or "Victory" Lab) is a toolkit for common video analysis and voice generation tasks.

## Features

- Video understanding and analysis tools (frame extraction, video QA, temporal grounding, dense captioning, etc.)
- Voice generation utilities (TTS, audio synthesis, etc.) *(planned)*
- Modular, extensible Python codebase
- Easy integration with Seed1.5-VL and other AI models

## Getting Started

1. **Clone the repo:**
   ```sh
   git clone https://github.com/haoyangli16/viclab.git
   cd viclab/viclab
   ```

2. **Install as a package:**
   ```sh
   pip install .
   ```

3. **Set your API key:**
   ```sh
   export OPENAI_API_KEY="your_volcengine_or_openai_api_key"
   ```

4. **Usage in your Python code:**
   ```python
   from viclab.video import ComplexVideoUnderstander, Strategy

   understander = ComplexVideoUnderstander()
   result = understander.analyze_video(
       video_path="path/to/video.mp4",
       prompt="Describe this video in detail.",
       extraction_strategy=Strategy.EVEN_INTERVAL,
       max_frames=5
   )
   print(result)
   ```

## Folder Structure

- `video/` — Video analysis tools and examples
- `voice/` — Voice generation tools (coming soon)
- `.gitignore` — Common ignores, including `.cursor`
- `README.md` — This file

## License

[Apache-2.0](LICENSE)

---

*viclab: Video, Voice, Victory!* 