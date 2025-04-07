#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF Mistral OCR 匯出工具

本程式可將 PDF 文件自動化轉換為 Markdown 格式，包含以下流程：

1. 使用 Mistral OCR 模型辨識 PDF 內文與圖片
2. 將辨識結果組成含圖片的 Markdown 檔
3. 使用 Gemini 模型將英文內容翻譯為台灣繁體中文
4. 匯出 Markdown 檔（原文版 + 翻譯版）與對應圖片

新增功能：
- 處理過程中的檢查點，可以保存中間結果
- Gradio 介面，方便調整參數和選擇輸出格式
"""

# Standard libraries
import os
import json
import base64
import time
import tempfile
from pathlib import Path
import pickle

# Third-party libraries
from IPython.display import Markdown, display
from pydantic import BaseModel
from dotenv import load_dotenv
import gradio as gr

# Mistral AI
from mistralai import Mistral
from mistralai.models import OCRResponse, ImageURLChunk, DocumentURLChunk, TextChunk

# Google Gemini
from google import genai
from google.genai import types

# ===== Pydantic Models =====

class StructuredOCR(BaseModel):
    file_name: str
    topics: list[str]
    languages: str
    ocr_contents: dict

# ===== Utility Functions =====

def retry_with_backoff(func, retries=5, base_delay=1.5):
    """Retry a function with exponential backoff."""
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e):
                wait_time = base_delay * (2 ** attempt)
                print(f"⚠️ API rate limit hit. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("❌ Failed after multiple retries.")

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """Replace image placeholders in markdown with base64-encoded images."""
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """Combine OCR text and images into a single markdown document."""
    markdowns: list[str] = []
    for page in ocr_response.pages:
        image_data = {img.id: img.image_base64 for img in page.images}
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))
    return "\n\n".join(markdowns)

def insert_ocr_below_images(markdown_str, ocr_img_map, page_idx):
    """Insert OCR results below images in markdown."""
    for img_id, ocr_text in ocr_img_map.get(page_idx, {}).items():
        markdown_str = markdown_str.replace(
            f"![{img_id}]({img_id})",
            f"![{img_id}]({img_id})\n\n> 📄 Image OCR Result：\n\n```json\n{ocr_text}\n```"
        )
    return markdown_str

def save_images_and_replace_links(markdown_str, images_dict, page_idx, image_folder="images"):
    """Save base64 images to files and update markdown links."""
    os.makedirs(image_folder, exist_ok=True)
    image_id_to_path = {}

    for i, (img_id, base64_str) in enumerate(images_dict.items()):
        img_bytes = base64.b64decode(base64_str.split(",")[-1])
        img_path = f"{image_folder}/page_{page_idx+1}_img_{i+1}.png"
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        image_id_to_path[img_id] = img_path

    for img_id, img_path in image_id_to_path.items():
        markdown_str = markdown_str.replace(
            f"![{img_id}]({img_id})", f"![{img_id}]({img_path})"
        )

    return markdown_str

# ===== Translation Functions =====

# Default translation system prompt
DEFAULT_TRANSLATION_SYSTEM_INSTRUCTION = """
你是一位專業的技術文件翻譯者。請將我提供的英文 Markdown 內容翻譯成**台灣繁體中文**。

**核心要求：**
1.  **翻譯所有英文文字：** 你的主要工作是翻譯內容中的英文敘述性文字（段落、列表、表格等）。
2.  **保持結構與程式碼不變：**
    * **不要**更改任何 Markdown 標記（如 `#`, `*`, `-`, `[]()`, `![]()`, ``` ```, ` `` `, `---`）。
    * **不要**翻譯或修改程式碼區塊 (``` ... ```) 和行內程式碼 (`code`) 裡的任何內容。
    * 若有 JSON，**不要**更改鍵（key），僅翻譯字串值（value）。
3.  **處理專有名詞：** 對於普遍接受的英文技術術語、縮寫或專有名詞（例如 API, SDK, CPU, Google, Python 等），傾向於**保留英文原文**。但請確保翻譯了其他所有非術語的常規英文文字。
4.  **直接輸出結果：** 請直接回傳翻譯後的完整 Markdown 文件，不要添加任何額外說明。
"""

def translate_markdown_pages(pages, gemini_client, model="gemini-2.0-flash", system_instruction=None):
    """Translate markdown pages using Gemini API."""
    if system_instruction is None:
        system_instruction = DEFAULT_TRANSLATION_SYSTEM_INSTRUCTION
        
    translated_pages = []
    
    for idx, page in enumerate(pages):
        try:
            print(f"🔁 正在翻譯第 {idx+1} 頁...")
            
            response = gemini_client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
                contents=page
            )
            
            translated_md = response.text.strip()
            translated_pages.append(translated_md)
            
        except Exception as e:
            print(f"⚠️ 翻譯第 {idx+1} 頁失敗：{e}")
            translated_pages.append(page)
    
    return translated_pages

# ===== PDF Processing Functions =====

def process_pdf_with_mistral_ocr(pdf_path, client, model="mistral-ocr-latest"):
    """Process PDF with Mistral OCR."""
    pdf_file = Path(pdf_path)
    
    # Upload to mistral
    uploaded_file = client.files.upload(
        file={
            "file_name": pdf_file.stem,
            "content": pdf_file.read_bytes(),
        },
        purpose="ocr"
    )
    
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
    
    # OCR analyze PDF
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model=model,
        include_image_base64=True
    )
    
    return pdf_response

def process_images_with_ocr(pdf_response, mistral_client, model="pixtral-12b-latest"):
    """Process images from PDF pages with OCR."""
    image_ocr_results = {}
    
    for page_idx, page in enumerate(pdf_response.pages):
        for i, img in enumerate(page.images):
            base64_data_url = img.image_base64
            
            def run_ocr_and_parse():
                # Step 1: basic OCR
                image_response = mistral_client.ocr.process(
                    document=ImageURLChunk(image_url=base64_data_url),
                    model="mistral-ocr-latest"
                )
                image_ocr_markdown = image_response.pages[0].markdown
                
                # Step 2: structure the OCR markdown
                structured = mistral_client.chat.parse(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                ImageURLChunk(image_url=base64_data_url),
                                TextChunk(text=(
                                    f"This is the image's OCR in markdown:\n{image_ocr_markdown}\n. "
                                    "Convert this into a structured JSON response with the OCR contents in a sensible dictionary."
                                ))
                            ]
                        }
                    ],
                    response_format=StructuredOCR,
                    temperature=0
                )
                
                structured_data = structured.choices[0].message.parsed
                pretty_text = json.dumps(structured_data.ocr_contents, indent=2, ensure_ascii=False)
                return pretty_text
            
            try:
                result = retry_with_backoff(run_ocr_and_parse, retries=4)
                image_ocr_results[(page_idx, img.id)] = result
            except Exception as e:
                print(f"❌ Failed at page {page_idx+1}, image {i+1}: {e}")
    
    # Reorganize results by page
    ocr_by_page = {}
    for (page_idx, img_id), ocr_text in image_ocr_results.items():
        ocr_by_page.setdefault(page_idx, {})[img_id] = ocr_text
    
    return ocr_by_page

# ===== Checkpoint Functions =====

def save_checkpoint(data, filename, console_output=None):
    """Save data to a checkpoint file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    message = f"✅ 已儲存檢查點：{filename}"
    print(message)
    if console_output is not None:
        console_output.append(message)

def load_checkpoint(filename, console_output=None):
    """Load data from a checkpoint file."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        message = f"✅ 已載入檢查點：{filename}"
        print(message)
        if console_output is not None:
            console_output.append(message)
        return data
    return None

# ===== Main Processing Function =====

def process_pdf_to_markdown(
    pdf_path, 
    mistral_client, 
    gemini_client,
    ocr_model="mistral-ocr-latest",
    structure_model="pixtral-12b-latest",
    translation_model="gemini-2.0-flash",
    translation_system_prompt=None,
    process_images=True,
    translate=True,
    output_dir=None,
    checkpoint_dir=None,
    console_output=None,
    use_existing_checkpoints=True
):
    """Main function to process PDF to markdown with translation."""
    pdf_file = Path(pdf_path)
    filename_stem = pdf_file.stem
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.getcwd()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(output_dir, f"checkpoints_{filename_stem}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Log function for both console and UI
    def log(message):
        print(message)
        if console_output is not None:
            console_output.append(message)
    
    # Checkpoint files
    pdf_ocr_checkpoint = os.path.join(checkpoint_dir, f"{filename_stem}_pdf_ocr.pkl")
    image_ocr_checkpoint = os.path.join(checkpoint_dir, f"{filename_stem}_image_ocr.pkl")
    markdown_checkpoint = os.path.join(checkpoint_dir, f"{filename_stem}_markdown.pkl")
    
    # Step 1: Process PDF with OCR (with checkpoint)
    pdf_response = None
    if use_existing_checkpoints:
        pdf_response = load_checkpoint(pdf_ocr_checkpoint, console_output)
    
    if pdf_response is None:
        log("🔍 Processing PDF with OCR...")
        pdf_response = process_pdf_with_mistral_ocr(pdf_path, mistral_client, model=ocr_model)
        save_checkpoint(pdf_response, pdf_ocr_checkpoint, console_output)
    
    # Step 2: Process images with OCR (with checkpoint)
    ocr_by_page = {}
    if process_images:
        if use_existing_checkpoints:
            ocr_by_page = load_checkpoint(image_ocr_checkpoint, console_output)
        
        if ocr_by_page is None:
            log("🖼️ Processing images with OCR...")
            ocr_by_page = process_images_with_ocr(pdf_response, mistral_client, model=structure_model)
            save_checkpoint(ocr_by_page, image_ocr_checkpoint, console_output)
    
    # Step 3: Create markdown pages (with checkpoint)
    markdown_pages = None
    if use_existing_checkpoints:
        markdown_pages = load_checkpoint(markdown_checkpoint, console_output)
    
    if markdown_pages is None:
        log("📝 Creating markdown pages with images...")
        markdown_pages = []
        image_folder_name = os.path.join(output_dir, f"images_{filename_stem}")
        
        for page_idx, page in enumerate(pdf_response.pages):
            images_dict = {img.id: img.image_base64 for img in page.images}
            
            md = page.markdown
            if process_images:
                md = insert_ocr_below_images(md, ocr_by_page, page_idx)
            md = save_images_and_replace_links(md, images_dict, page_idx, image_folder=image_folder_name)
            
            markdown_pages.append(md)
        save_checkpoint(markdown_pages, markdown_checkpoint, console_output)
    
    # Step 4: Translate markdown pages (optional)
    translated_markdown_pages = markdown_pages
    if translate:
        log("🔄 Translating markdown pages...")
        translated_markdown_pages = translate_markdown_pages(
            markdown_pages, 
            gemini_client, 
            model=translation_model,
            system_instruction=translation_system_prompt
        )
    
    # Step 5: Combine pages into complete markdown
    final_markdown_translated = "\n\n---\n\n".join(translated_markdown_pages)
    final_markdown_original = "\n\n---\n\n".join(markdown_pages)
    
    # Step 6: Save files
    translated_md_name = os.path.join(output_dir, f"{filename_stem}_translated.md")
    original_md_name = os.path.join(output_dir, f"{filename_stem}_original.md")
    
    with open(translated_md_name, "w", encoding="utf-8") as f:
        f.write(final_markdown_translated)
    
    with open(original_md_name, "w", encoding="utf-8") as f:
        f.write(final_markdown_original)
    
    image_folder_name = os.path.join(output_dir, f"images_{filename_stem}")
    log(f"✅ 已儲存翻譯版：{translated_md_name}")
    log(f"✅ 已儲存原始英文版：{original_md_name}")
    log(f"✅ 圖片資料夾：{image_folder_name}")
    
    return {
        "translated_file": translated_md_name,
        "original_file": original_md_name,
        "image_folder": image_folder_name,
        "translated_content": final_markdown_translated,
        "original_content": final_markdown_original
    }

# ===== Gradio Interface =====

def create_gradio_interface():
    """Create a Gradio interface for the PDF to Markdown tool."""
    
    # Initialize clients
    load_dotenv()
    
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("❌ 找不到 MISTRAL_API_KEY，請檢查 .env 是否正確設置。")
    mistral_client = Mistral(api_key=mistral_api_key)
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("❌ 未在 .env 找到 GEMINI_API_KEY，請確認已正確設置。")
    gemini_client = genai.Client(api_key=gemini_api_key)
    
    # Define processing function for Gradio
    def process_pdf(
        pdf_file, 
        ocr_model, 
        structure_model, 
        translation_model,
        translation_system_prompt,
        process_images,
        translate,
        output_format,
        output_dir,
        use_existing_checkpoints,
        console_output
    ):
        # Handle the uploaded PDF file
        if pdf_file is None:
            console_output.append("❌ 請先上傳 PDF 檔案")
            return "請先上傳 PDF 檔案", console_output
        
        # Create checkpoint directory
        if output_dir:
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
        else:
            temp_dir = tempfile.mkdtemp()
            checkpoint_dir = os.path.join(temp_dir, "checkpoints")
            output_dir = temp_dir
        
        # Clear console output
        console_output = []
        console_output.append(f"🚀 開始處理 PDF: {os.path.basename(pdf_file)}")
        console_output.append(f"📂 輸出目錄: {output_dir}")
        console_output.append(f"💾 檢查點目錄: {checkpoint_dir}")
        
        # Determine if translation is needed based on output format
        need_translation = translate and (output_format != "英文原文")
        if need_translation:
            console_output.append("✅ 將進行中文翻譯")
        else:
            console_output.append("ℹ️ 跳過中文翻譯步驟")
        
        if use_existing_checkpoints:
            console_output.append("✅ 使用現有檢查點（如果存在）")
        else:
            console_output.append("🔄 重新處理所有步驟（不使用現有檢查點）")
        
        # Process the PDF
        try:
            result = process_pdf_to_markdown(
                pdf_path=pdf_file,
                mistral_client=mistral_client,
                gemini_client=gemini_client,
                ocr_model=ocr_model,
                structure_model=structure_model,
                translation_model=translation_model,
                translation_system_prompt=translation_system_prompt if translation_system_prompt.strip() else None,
                process_images=process_images,
                translate=need_translation,  # Use need_translation flag instead of translate
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                console_output=console_output,
                use_existing_checkpoints=use_existing_checkpoints
            )
            
            # Determine which content to return based on output format
            console_output.append("✅ 處理完成！")
            
            if output_format == "中文翻譯":
                return result["translated_content"], console_output
            elif output_format == "英文原文":
                return result["original_content"], console_output
            else:  # Both
                return f"# 英文原文\n\n{result['original_content']}\n\n# 中文翻譯\n\n{result['translated_content']}", console_output
                
        except Exception as e:
            error_message = f"❌ 處理過程中發生錯誤: {str(e)}"
            console_output.append(error_message)
            return error_message, console_output
    
    # Create Gradio interface
    with gr.Blocks(title="PDF Mistral OCR 匯出工具") as demo:
        gr.Markdown("# PDF Mistral OCR 匯出工具")
        gr.Markdown("將 PDF 文件自動化轉換為 Markdown 格式，支援圖片 OCR 與中文翻譯")
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_file = gr.File(label="上傳 PDF 檔案", file_types=[".pdf"])
                
                with gr.Accordion("基本設定", open=True):
                    output_dir = gr.Textbox(
                        label="輸出目錄（留空使用預設目錄）", 
                        placeholder="例如: C:/Users/Documents/output"
                    )
                    
                    use_existing_checkpoints = gr.Checkbox(
                        label="使用現有檢查點（如果存在）", 
                        value=True,
                        info="啟用後，如果檢查點存在，將跳過已完成的步驟"
                    )
                    
                    output_format = gr.Radio(
                        label="輸出格式", 
                        choices=["中文翻譯", "英文原文", "中英對照"], 
                        value="中文翻譯"
                    )
                
                with gr.Accordion("處理選項", open=True):
                    process_images = gr.Checkbox(
                        label="處理圖片 OCR", 
                        value=True,
                        info="啟用後，將對 PDF 中的圖片進行 OCR 辨識"
                    )
                    
                    translate = gr.Checkbox(
                        label="翻譯成中文", 
                        value=True,
                        info="啟用後，將英文內容翻譯為中文"
                    )
                
                with gr.Accordion("模型設定", open=False):
                    ocr_model = gr.Dropdown(
                        label="OCR 模型", 
                        choices=["mistral-ocr-latest"], 
                        value="mistral-ocr-latest"
                    )
                    structure_model = gr.Dropdown(
                        label="結構化模型", 
                        choices=["pixtral-12b-latest", "mistral-large-latest"], 
                        value="pixtral-12b-latest"
                    )
                    translation_model = gr.Dropdown(
                        label="翻譯模型", 
                        choices=["gemini-2.0-flash", "gemini-2.0-pro"], 
                        value="gemini-2.0-flash"
                    )
                
                with gr.Accordion("進階設定", open=False):
                    translation_system_prompt = gr.Textbox(
                        label="翻譯系統提示詞", 
                        value=DEFAULT_TRANSLATION_SYSTEM_INSTRUCTION,
                        lines=10
                    )
                
                process_button = gr.Button("開始處理", variant="primary")
            
            with gr.Column(scale=2):
                with gr.Tab("輸出結果"):
                    output = gr.Markdown(label="輸出結果")
                
                with gr.Tab("處理日誌"):
                    console_output = gr.Textbox(
                        label="處理進度", 
                        lines=20,
                        max_lines=50,
                        interactive=False
                    )
        
        process_button.click(
            fn=process_pdf,
            inputs=[
                pdf_file, 
                ocr_model, 
                structure_model, 
                translation_model,
                translation_system_prompt,
                process_images,
                translate,
                output_format,
                output_dir,
                use_existing_checkpoints,
                console_output
            ],
            outputs=[output, console_output]
        )
        
        gr.Markdown("""
        ## 使用說明
        
        1. 上傳 PDF 檔案（可拖曳或點擊上傳）
        2. 基本設定：
           - 指定輸出目錄（可選，留空使用預設目錄）
           - 選擇是否使用現有檢查點（如果存在）
           - 選擇輸出格式（中文翻譯、英文原文、中英對照）
        3. 處理選項：
           - 選擇是否處理圖片 OCR
           - 選擇是否翻譯成中文（注意：如果輸出格式選擇「英文原文」，則不會進行翻譯）
        4. 點擊「開始處理」按鈕
        5. 處理過程中，可在「處理日誌」標籤頁查看進度
        6. 處理完成後，結果將顯示在「輸出結果」標籤頁，並自動儲存檔案到指定目錄
        
        ## 檢查點說明
        
        本工具會在處理過程中建立檢查點，以便在中斷後繼續處理，避免重複請求 API：
        
        - **PDF OCR 檢查點**：儲存 PDF 文件的 OCR 結果
        - **圖片 OCR 檢查點**：儲存 PDF 中圖片的 OCR 結果
        - **Markdown 檢查點**：儲存生成的 Markdown 頁面
        
        如果您想重新處理特定步驟，可以取消勾選「使用現有檢查點」選項，或手動刪除檢查點目錄。
        
        ## 輸出檔案
        
        - `[檔名]_translated.md`：翻譯後的 Markdown 檔案
        - `[檔名]_original.md`：原始英文 Markdown 檔案
        - `images_[檔名]/`：儲存的圖片資料夾
        - `checkpoints/`：處理過程中的檢查點資料夾
        """)
    
    return demo

# ===== Main Execution =====

if __name__ == "__main__":
    # Create and launch Gradio interface
    demo = create_gradio_interface()
    demo.launch()
