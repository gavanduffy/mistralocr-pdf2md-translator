# PDF Mistral OCR 匯出工具

本工具可將 PDF 文件自動化轉換為 Markdown 格式，包含以下功能：

1. 使用 **Mistral OCR** 模型辨識 PDF 內文與圖片
2. 將辨識結果組成含圖片的 Markdown 檔
3. 使用 **Gemini** 模型將英文內容翻譯為**台灣繁體中文**
4. 匯出 Markdown 檔（原文版 + 翻譯版）與對應圖片

## 新增功能

- **處理檢查點**：處理過程中會建立檢查點，可以在中斷後繼續處理，避免重複請求 API
- **Gradio 介面**：提供友善的使用者介面，方便調整參數和選擇輸出格式
- **彈性輸出**：可選擇只輸出英文、只輸出中文或中英對照
- **參數調整**：可調整使用的模型、系統提示詞等參數
- **處理日誌**：即時顯示處理進度和狀態
- **自訂輸出目錄**：可指定檔案輸出的位置
- **檢查點控制**：可選擇是否使用現有檢查點，方便重新處理特定步驟

## 安裝需求

1. 安裝必要的 Python 套件：

```bash
pip install -r requirements.txt
```

2. 建立 `.env` 檔案，並設定 API 金鑰：

```
# Mistral AI API Key
# 請從 https://console.mistral.ai/ 獲取
MISTRAL_API_KEY=your_mistral_api_key_here

# Google Gemini API Key
# 請從 https://aistudio.google.com/app/apikey 獲取
GEMINI_API_KEY=your_gemini_api_key_here
```

## 使用方法

### 使用 Gradio 介面

執行以下命令啟動 Gradio 介面：

```bash
python mistralocr_app.py
```

然後在瀏覽器中開啟顯示的網址（通常是 http://127.0.0.1:7860）。

### 介面使用說明

1. 上傳 PDF 檔案（可拖曳或點擊上傳）
2. 基本設定：
   - 指定輸出目錄（可選，留空使用預設目錄）
   - 選擇是否使用現有檢查點（如果存在）
   - 選擇輸出格式（中文翻譯、英文原文、中英對照）
3. 處理選項：
   - 選擇是否處理圖片 OCR
   - 選擇是否翻譯成中文
4. 模型設定（可選）：
   - 選擇 OCR 模型
   - 選擇結構化模型
   - 選擇翻譯模型
5. 進階設定（可選）：
   - 修改翻譯系統提示詞
6. 點擊「開始處理」按鈕
7. 處理過程中，可在「處理日誌」標籤頁查看進度
8. 處理完成後，結果將顯示在「輸出結果」標籤頁，並自動儲存檔案到指定目錄

## 檔案說明

- `mistralocr_app.py`：主程式檔案
- `requirements.txt`：所需的 Python 套件
- `mistralocr_pdf2md.ipynb`：根據 [Mistral 官方 Notebook](https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/ocr/structured_ocr.ipynb) 修改的學習筆記，加入翻譯與圖片儲存功能。
- 處理結果：
  - `[檔名]_translated.md`：翻譯後的 Markdown 檔案
  - `[檔名]_original.md`：原始英文 Markdown 檔案
  - `images_[檔名]/`：儲存的圖片資料夾
  - `checkpoints_[檔名]/`：處理過程中的檢查點資料夾

## 注意事項

- 處理過程中會建立檢查點，可以在中斷後繼續處理，避免重複請求 API
- 圖片會儲存在 `images_[檔名]` 資料夾中
- 翻譯和原文 Markdown 檔案會儲存在程式執行目錄中

## 技術來源與引用

本專案整合並改作自以下技術或官方範例：

- [Mistral 文件處理功能說明](https://docs.mistral.ai/capabilities/document/)
- [Mistral 官方 Colab Notebook](https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/ocr/structured_ocr.ipynb)

其中 `mistralocr_pdf2md.ipynb` 為學習用途所建立，根據官方範例進行延伸與修改，加入了中英文翻譯流程與本地圖片儲存功能等。

本工具亦整合以下第三方 API/工具：

- [Mistral API](https://mistral.ai/)
- [Google Gemini API](https://ai.google.dev/)
- [Gradio](https://www.gradio.app/)

> 本專案為個人開發與學習用途，與上述服務提供者無任何官方關聯。請使用者自行準備合法的 API 金鑰，並遵守各 API 提供商的使用條款。
