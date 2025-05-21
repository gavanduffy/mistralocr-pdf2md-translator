# Mistral OCR PDF-to-Markdown Translation Tool

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/dodo13114arch/mistral-ocr-translator-demo)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dodo13114arch/mistralocr-pdf2md-translator)

**[Try the Online Demo (Hugging Face Spaces)](https://huggingface.co/spaces/dodo13114arch/mistral-ocr-translator-demo)**

---

This tool can automatically convert PDF documents into Markdown format and provides the following features:

1. Uses the **Mistral OCR** model to recognize text and images in PDF files.
2. Assembles the recognition results into a Markdown file with images.
3. Uses the **Gemini** or **OpenAI** model to translate English content into **Taiwan Traditional Chinese** (or other languages by modifying the prompt).
4. Optionally uses **Pixtral**, **Gemini**, or **OpenAI** models to perform structured OCR on image content.
5. Exports Markdown files (original + translated versions) and corresponding images.

## Installation Requirements

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file and set up your API keys:

```
# Mistral AI API Key
# Obtain from https://console.mistral.ai/
MISTRAL_API_KEY=your_mistral_api_key_here

# Google Gemini API Key (required to use Gemini model)
# Obtain from https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API Key (required to use OpenAI model)
# Obtain from https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Using the Gradio Interface

Run the following command to start the Gradio interface:

```bash
python mistralocr_app.py
```

Then open the displayed URL in your browser (typically http://127.0.0.1:7860).

### Interface Instructions

1. Upload a PDF file (drag and drop or click to upload).
2. Basic settings:
    * Specify the output directory (optional; leave blank to save to the desktop `MistralOCR_Output` folder).
    * Choose whether to use existing checkpoints (enabled by default).
    * Choose output format (**multiple options allowed**: "Chinese Translation", "English Original"; both are selected by default).
3. Processing options:
    * Choose whether to process image OCR (enabled by default).
    * *(Translation is controlled by the "Output Format" selection)*
4. Model settings (optional):
    * Select the OCR model (currently only Mistral is supported).
    * Select the structured model (Pixtral, Gemini, OpenAI).
    * Select the translation model (Gemini, OpenAI).
5. Advanced settings (optional):
    * Modify the translation system prompt (can adjust translation to other languages).
6. Click the "Start Processing" button.
7. During processing, you can view detailed progress in the "Processing Log" tab.
8. After processing is complete, results will be shown in the "Output Results" tab, and files will be automatically saved to the specified directory.

## File Descriptions

- `mistralocr_app.py`: Main Gradio application script.
- `requirements.txt`: List of required Python packages.
- `mistralocr_pdf2md.ipynb`: Modified study notebook based on the [Mistral official notebook](https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/ocr/structured_ocr.ipynb), with translation and image saving features added.
- `mistralocr_pdf2md_claude_refined.ipynb`: (Optional) Another development/testing notebook version.
- **Output files (located in the specified output directory):**
    * `[filename]_original.md`: Processed original (English) Markdown file (if selected).
    * `[filename]_translated.md`: Translated (Traditional Chinese) Markdown file (if selected).
    * `images_[filename]/`: Folder containing images extracted and saved from the PDF.
    * `checkpoints_[filename]/`: Folder for checkpoints during processing, containing intermediate results.

## Notes

- During processing, a `checkpoints_[filename]` folder will be created in the **output directory** to store intermediate results, so you can resume processing after interruption and avoid repeated API requests. To force reprocessing, uncheck "Use Existing Checkpoints" or delete the corresponding checkpoint folder.
- Extracted images will be saved in the `images_[filename]` folder in the **output directory**.
- The final Markdown files (`_original.md`, `_translated.md`) will be saved in the user-specified **output directory** (if not specified, defaults to the `MistralOCR_Output` folder on the desktop).
- Make sure the `.env` file is correctly configured with your Mistral AI and Google Gemini API keys.

## Technical Sources & References

This project is integrated and adapted from the following technologies and official examples:

- [Mistral Document Processing Capabilities](https://docs.mistral.ai/capabilities/document/)
- [Mistral Official Colab Notebook](https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/ocr/structured_ocr.ipynb)

The `mistralocr_pdf2md.ipynb` notebook is for learning purposes, extended and modified based on the official example, with added translation and local image saving features.

This tool also integrates the following third-party APIs/tools:

- [Mistral API](https://mistral.ai/)
- [Google Gemini API](https://ai.google.dev/)
- [OpenAI API](https://openai.com/)
- [Gradio](https://www.gradio.app/)

> This project is for personal development and learning purposes only, and has no official affiliation with the aforementioned service providers. Users must prepare their own legal API keys and comply with the terms of service of each API provider ([Mistral](https://mistral.ai/terms), [Gemini](https://ai.google.dev/terms), [OpenAI](https://openai.com/policies)).
