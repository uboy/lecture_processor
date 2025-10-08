# lecture_processor
Process audio/video lecture with presentation and prepare summary

1. Install requirements
``` bash
sudo apt update
sudo apt install ffmpeg
```

2. Run a local Ollama (or another local LLM server) and make sure that it is accessible via HTTP (in the example in the code — http://localhost:11434/api/generate ).
(If you are using a different local LLM infrastructure, you can adapt the call_ollama function in the script.)

3. `python3 lecture_processor.py --input lecture.mp4 --slides slides.pdf --ollama-model llama2 --outdir ./out
`

