# multimodal_rag.py
import os
import base64
import logging
import shutil
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# PDF обработка
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Ollama клиент для мультимодальных моделей
import ollama

# Прогресс-бары
from tqdm import tqdm

# Явно указываем путь к Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Добавляем путь в PATH для надежности
os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR'

# Проверка наличия Tesseract
def check_tesseract():
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True)
        print(f"✅ Tesseract найден: {result.stdout.splitlines()[0]}")
        return True
    except:
        print("⚠️ Tesseract не найден. OCR будет отключен.")
        return False

HAS_TESSERACT = check_tesseract()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class MultimodalRAG:
    """RAG система с поддержкой текста и изображений"""
    
    def __init__(self):
        self.docs_dir = "documents"
        self.images_cache = "images_cache"
        self.db_dir = "chroma_db"
        self.text_model = "gemma3:27b"  # для текста
        self.vision_model = "llava"  # для изображений (должна быть установлена в Ollama)
        
        # Создаем папки
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.images_cache, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Инициализация
        self._init_components()
    
    def _init_components(self):
        """Инициализация компонентов"""
        print("\n🔄 Загрузка компонентов...")
        
        # Эмбеддинги для текста
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small"
        )
        
        # Текстовая LLM
        self.llm = Ollama(
            model=self.text_model,
            temperature=0.1
        )
        
        # Проверяем наличие vision модели в Ollama
        try:
            ollama.list()
            print(f"✅ Ollama доступен")
        except:
            print("⚠️ Ollama не запущен. Запустите 'ollama serve'")
        
        # Загружаем существующую базу если есть
        if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.db_dir,
                    embedding_function=self.embeddings
                )
                count = self.vectorstore._collection.count()
                print(f"✅ Загружена база с {count} чанками")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки базы: {e}")
                self.vectorstore = None
        else:
            self.vectorstore = None
            print("📁 База не найдена")
    
    def _extract_images_from_pdf(self, pdf_path: str, pdf_name: str) -> List[Dict]:
        """Извлечение изображений из PDF"""
        images_data = []
        
        # Путь к Poppler
        poppler_path = r"C:\poppler\poppler-25.12.0\Library\bin"
        
        try:
            print(f"   🖼️ Извлечение изображений...")
            
            # Добавляем путь в окружение для надежности
            os.environ['PATH'] = poppler_path + ';' + os.environ.get('PATH', '')
            
            pages = convert_from_path(
                pdf_path, 
                dpi=150,
                poppler_path=poppler_path
            )
            
            for page_num, page_image in enumerate(pages, 1):
                # Сохраняем изображение страницы
                image_filename = f"{pdf_name}_page_{page_num}.png"
                image_path = os.path.join(self.images_cache, image_filename)
                page_image.save(image_path, "PNG")
                
                # Пробуем извлечь текст с изображения через OCR
                if HAS_TESSERACT:
                    try:
                        ocr_text = pytesseract.image_to_string(page_image, lang='rus+eng')
                    except Exception as e:
                        print(f"   ⚠️ OCR ошибка для страницы {page_num}: {e}")
                        ocr_text = ""
                else:
                    ocr_text = ""
                
                images_data.append({
                    "path": image_path,
                    "page": page_num,
                    "ocr_text": ocr_text.strip(),
                    "source": pdf_name
                })
                
                if page_num % 50 == 0:
                    print(f"      ...обработано {page_num} страниц")
            
            print(f"   ✅ Извлечено {len(pages)} страниц как изображения")
            
        except Exception as e:
            print(f"   ⚠️ Ошибка извлечения изображений: {e}")
            import traceback
            traceback.print_exc()
        
        return images_data
    
    def _describe_image_with_llava(self, image_path: str) -> str:
        """Получение описания изображения через LLaVA"""
        try:
            # Читаем изображение и кодируем в base64
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Отправляем запрос к LLaVA
            response = ollama.generate(
                model=self.vision_model,
                prompt="""Опиши подробно, что изображено на этой картинке. 
                Если это график или диаграмма, объясни, что на нем показано.
                Если это схема или архитектура, опиши компоненты и связи.
                Ответ дай на русском языке.""",
                images=[image_base64]
            )
            
            return response['response']
            
        except Exception as e:
            print(f"   ⚠️ Ошибка описания изображения: {e}")
            return "[Описание изображения не удалось получить]"
    
    def _extract_text_from_pdf(self, pdf_path: str, pdf_name: str) -> List[Document]:
        """Извлечение текста из PDF"""
        text_docs = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text.strip():
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": pdf_name,
                                "page": page_num + 1,
                                "type": "text"
                            }
                        )
                        text_docs.append(doc)
                        
        except Exception as e:
            print(f"   ⚠️ Ошибка чтения текста: {e}")
        
        return text_docs
    
    def index_pdfs(self):
        """Индексация PDF файлов с текстом и изображениями"""
        start = time.time()
        
        try:
            # Поиск PDF файлов
            pdf_files = list(Path(self.docs_dir).glob("*.pdf"))
            
            if not pdf_files:
                print("❌ Нет PDF файлов в папке documents/")
                return False
            
            print(f"\n📚 Найдено файлов: {len(pdf_files)}")
            for f in pdf_files:
                size = f.stat().st_size / (1024 * 1024)
                print(f"   📄 {f.name} ({size:.1f} MB)")
            
            all_chunks = []
            
            # Обрабатываем каждый PDF
            for pdf_file in tqdm(pdf_files, desc="Обработка PDF"):
                pdf_name = pdf_file.stem
                print(f"\n📖 Обработка: {pdf_file.name}")
                
                # 1. ИЗВЛЕКАЕМ ТЕКСТ
                text_docs = self._extract_text_from_pdf(str(pdf_file), pdf_name)
                print(f"   📝 Текст: {len(text_docs)} страниц")
                
                # Разделяем текст на чанки
                if text_docs:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=100
                    )
                    text_chunks = splitter.split_documents(text_docs)
                    all_chunks.extend(text_chunks)
                    print(f"   🔢 Текстовых чанков: {len(text_chunks)}")
                
                # 2. ИЗВЛЕКАЕМ ИЗОБРАЖЕНИЯ
                images = self._extract_images_from_pdf(str(pdf_file), pdf_name)
                
                # 3. ОПИСЫВАЕМ ИЗОБРАЖЕНИЯ через LLaVA
                for img in tqdm(images, desc="   🖼️ Описание изображений"):
                    # Получаем описание
                    description = self._describe_image_with_llava(img["path"])
                    
                    # Создаем документ из описания
                    doc = Document(
                        page_content=f"[ИЗОБРАЖЕНИЕ: {description}]\n\nOCR текст с картинки: {img['ocr_text']}",
                        metadata={
                            "source": pdf_name,
                            "page": img["page"],
                            "type": "image",
                            "image_path": img["path"],
                            "ocr_text": img["ocr_text"][:200]  # для превью
                        }
                    )
                    all_chunks.append(doc)
                
                print(f"   🖼️ Изображений обработано: {len(images)}")
            
            print(f"\n📊 Всего создано чанков: {len(all_chunks)}")
            
            # Создаем векторную базу
            print("\n💾 Создание векторной базы...")
            
            # Удаляем старую базу если есть
            if os.path.exists(self.db_dir):
                shutil.rmtree(self.db_dir)
            
            # Создаем новую базу
            self.vectorstore = Chroma.from_documents(
                documents=all_chunks,
                embedding=self.embeddings,
                persist_directory=self.db_dir
            )
            self.vectorstore.persist()
            
            elapsed = time.time() - start
            print(f"\n✅ Индексация завершена за {elapsed:.1f} сек")
            print(f"📊 Всего чанков в базе: {len(all_chunks)}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Задать вопрос с поддержкой изображений"""
        try:
            if not self.vectorstore:
                return {
                    "answer": "❌ База не найдена. Сначала выполните индексацию.",
                    "sources": []
                }
            
            # Поиск релевантных чанков
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 8}  # Больше чанков для поиска
            )
            docs = retriever.get_relevant_documents(question)
            
            # Разделяем текстовые и графические источники
            text_context = []
            image_sources = []
            
            for doc in docs:
                if doc.metadata.get("type") == "image":
                    image_sources.append({
                        "source": doc.metadata["source"],
                        "page": doc.metadata["page"],
                        "image_path": doc.metadata["image_path"],
                        "ocr_text": doc.metadata.get("ocr_text", ""),
                        "description": doc.page_content.replace("[ИЗОБРАЖЕНИЕ: ", "").split("]\n\n")[0]
                    })
                else:
                    text_context.append(doc.page_content)
            
            # Формируем промпт
            context = "\n\n".join(text_context)
            
            prompt_template = """Ты - эксперт по техническим книгам. 
            Используй контекст для ответа на вопрос.
            Если в ответе есть информация из изображений, обязательно укажи это.

            Контекст:
            {context}

            Вопрос: {question}

            Подробный ответ:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Создаем цепочку
            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            
            print(f"\n❓ Вопрос: {question}")
            result = chain.invoke({"query": question})
            
            return {
                "answer": result["result"],
                "sources": docs,
                "images": image_sources
            }
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"❌ Ошибка: {e}",
                "sources": [],
                "images": []
            }
    
    def stats(self):
        """Статистика базы"""
        if self.vectorstore:
            try:
                count = self.vectorstore._collection.count()
                return f"✅ База готова, всего чанков: {count}"
            except Exception as e:
                return f"❌ Ошибка получения статистики: {e}"
        return "❌ База не найдена"

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", "-a", type=str, help="Задать вопрос")
    parser.add_argument("--index", "-i", action="store_true", help="Индексировать PDF")
    parser.add_argument("--stats", "-s", action="store_true", help="Статистика")
    
    args = parser.parse_args()
    
    rag = MultimodalRAG()
    
    if args.stats:
        print(rag.stats())
    
    elif args.index:
        rag.index_pdfs()
    
    elif args.ask:
        result = rag.ask(args.ask)
        print(f"\n📝 Ответ:\n{result['answer']}")
        
        if result['images']:
            print(f"\n🖼️ Найдено изображений: {len(result['images'])}")
            for img in result['images']:
                print(f"   📍 {img['source']}, стр. {img['page']}")
                print(f"   📝 {img['description'][:100]}...")
    
    else:
        print("\n" + "="*50)
        print("🖼️ Multimodal RAG System")
        print("="*50)
        print("Команды:")
        print("  python multimodal_rag.py --ask 'вопрос'")
        print("  python multimodal_rag.py --index")
        print("  python multimodal_rag.py --stats")
        print("="*50)

if __name__ == "__main__":
    main()