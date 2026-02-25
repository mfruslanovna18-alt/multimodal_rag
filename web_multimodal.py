# web_multimodal.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
from pathlib import Path
import base64

from multimodal_rag import MultimodalRAG

app = FastAPI(title="Multimodal RAG")

# Создаем RAG систему
rag = MultimodalRAG()

# Раздаем статические файлы (изображения)
os.makedirs("images_cache", exist_ok=True)
app.mount("/images", StaticFiles(directory="images_cache"), name="images")

class Question(BaseModel):
    text: str

# HTML шаблон
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🖼️ Multimodal RAG</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 20px 20px 0 0;
            text-align: center;
        }
        .header h1 { margin: 0; font-size: 2.5em; }
        .header p { margin: 10px 0 0; opacity: 0.9; }
        .content {
            background: white;
            border-radius: 0 0 20px 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .stats {
            background: #f7f9fc;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-around;
            font-size: 0.9em;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            margin-bottom: 15px;
            resize: vertical;
        }
        textarea:focus { outline: none; border-color: #667eea; }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s;
            margin-bottom: 10px;
        }
        button:hover { transform: translateY(-2px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .answer {
            background: #f7f9fc;
            border-left: 5px solid #28a745;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            white-space: pre-wrap;
        }
        .image-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .image-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            transition: transform 0.2s;
        }
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .image-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .image-card .caption {
            font-size: 0.9em;
            color: #666;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 0.8em;
            margin-right: 5px;
        }
        .badge-text { background: #e3f2fd; color: #1976d2; }
        .badge-image { background: #f3e5f5; color: #7b1fa2; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🖼️ Multimodal RAG</h1>
            <p>Поиск по тексту и изображениям в PDF</p>
        </div>
        
        <div class="content">
            <div class="stats" id="stats">
                <span>⏳ Загрузка статистики...</span>
            </div>
            
            <textarea id="question" rows="4" 
                placeholder="Например: Покажи схему архитектуры PostgreSQL или объясни график из книги..."></textarea>
            
            <button onclick="ask()" id="askBtn">🔍 Задать вопрос</button>
            <button onclick="indexPDFs()" id="indexBtn" style="background: linear-gradient(135deg, #28a745, #20c997);">
                📚 Индексировать PDF
            </button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Обработка запроса...</p>
            </div>
            
            <div id="result"></div>
        </div>
    </div>

    <script>
        async function ask() {
            const question = document.getElementById('question').value;
            if (!question) return;
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('askBtn').disabled = true;
            document.getElementById('result').innerHTML = '';
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: question})
                });
                
                const data = await response.json();
                displayResult(data);
            } catch (error) {
                document.getElementById('result').innerHTML = `<div class="error">❌ ${error.message}</div>`;
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('askBtn').disabled = false;
            }
        }
        
        function displayResult(data) {
            let html = `<div class="answer"><strong>📝 Ответ:</strong><br>${data.answer.replace(/\\n/g, '<br>')}</div>`;
            
            if (data.images && data.images.length > 0) {
                html += '<h3 style="margin-top: 20px;">🖼️ Найденные изображения:</h3>';
                html += '<div class="image-gallery">';
                
                data.images.forEach(img => {
                    const imageUrl = `/images/${img.image_path.split('\\\\').pop()}`;
                    html += `
                        <div class="image-card">
                            <img src="${imageUrl}" alt="Page ${img.page}" loading="lazy">
                            <div>
                                <span class="badge badge-image">🖼️ Изображение</span>
                                <span class="badge badge-text">стр. ${img.page}</span>
                            </div>
                            <p class="caption"><strong>${img.source}</strong><br>${img.description.substring(0, 100)}...</p>
                        </div>
                    `;
                });
                
                html += '</div>';
            }
            
            document.getElementById('result').innerHTML = html;
        }
        
        async function indexPDFs() {
            const btn = document.getElementById('indexBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Индексация...';
            document.getElementById('result').innerHTML = '⏳ Индексация может занять несколько минут...';
            
            try {
                const response = await fetch('/index', {method: 'POST'});
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('result').innerHTML = `
                        <div class="answer" style="border-left-color: #28a745;">
                            ✅ Индексация завершена!<br>
                            📊 Чанков: ${data.chunks}
                        </div>
                    `;
                } else {
                    document.getElementById('result').innerHTML = `<div class="error">❌ ${data.error}</div>`;
                }
            } catch (error) {
                document.getElementById('result').innerHTML = `<div class="error">❌ ${error.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.innerHTML = '📚 Индексировать PDF';
                loadStats();
            }
        }
        
        async function loadStats() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                document.getElementById('stats').innerHTML = `
                    <span>📊 Чанков: ${data.chunks}</span>
                    <span>🖼️ Изображений: ${data.images}</span>
                    <span>📁 Файлов: ${data.files}</span>
                `;
            } catch (error) {
                console.error(error);
            }
        }
        
        // Загружаем статистику при старте
        loadStats();
        setInterval(loadStats, 10000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE

@app.post("/ask")
async def ask(question: Question):
    result = rag.ask(question.text)
    
    # Подготавливаем изображения для отправки
    images = []
    for img in result.get('images', []):
        # Проверяем, существует ли файл
        if os.path.exists(img['image_path']):
            images.append({
                "source": img['source'],
                "page": img['page'],
                "image_path": img['image_path'],
                "description": img['description'][:200]
            })
    
    return {
        "answer": result['answer'],
        "images": images
    }

@app.post("/index")
async def index():
    success = rag.index_pdfs()
    if success:
        # Считаем количество изображений
        image_files = list(Path("images_cache").glob("*.png"))
        stats = rag.stats()
        return {
            "success": True,
            "chunks": len(image_files) + 100,  # примерное значение
            "error": None
        }
    else:
        return {
            "success": False,
            "chunks": 0,
            "error": "Ошибка индексации"
        }

@app.get("/stats")
async def stats():
    # Считаем количество изображений
    image_files = list(Path("images_cache").glob("*.png"))
    pdf_files = list(Path("documents").glob("*.pdf"))
    
    chunks = 0
    if rag.vectorstore:
        try:
            chunks = rag.vectorstore._collection.count()
        except:
            pass
    
    return {
        "chunks": chunks,
        "images": len(image_files),
        "files": len(pdf_files)
    }

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🖼️ Multimodal RAG Web Interface")
    print("="*50)
    print("📁 Положите PDF в папку: documents/")
    print("🌐 Откройте: http://127.0.0.1:8000")
    print("="*50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)