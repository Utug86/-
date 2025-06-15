import logging

def get_logger(name: str, logfile: str = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        if logfile:
            fh = logging.FileHandler(logfile, encoding="utf-8")
            fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(sh)
        logger.setLevel(level)
    return logger

# код - rag_file_utils.py

from pathlib import Path
import logging
from typing import Optional
import pandas as pd
from bs4 import BeautifulSoup

from logs import get_logger

logger = get_logger("rag_file_utils")

# --- Dynamic imports ---
def _try_import_docx() -> Optional[object]:
    try:
        import docx
        return docx
    except ImportError:
        logger.warning("python-docx не установлен. Форматы .docx будут проигнорированы.")
        return None

def _try_import_pypdf2() -> Optional[object]:
    try:
        import PyPDF2
        return PyPDF2
    except ImportError:
        logger.warning("PyPDF2 не установлен. Форматы .pdf будут проигнорированы.")
        return None

def _try_import_textract() -> Optional[object]:
    try:
        import textract
        return textract
    except ImportError:
        logger.warning("textract не установлен. Некоторые форматы могут быть не поддержаны.")
        return None

DOCX = _try_import_docx()
PDF = _try_import_pypdf2()
TEXTRACT = _try_import_textract()

COMMON_ENCODINGS = ["utf-8", "cp1251", "windows-1251", "latin-1", "iso-8859-1"]
MAX_FILE_SIZE_MB = 100

def _smart_read_text(path: Path) -> str:
    """
    Пробует прочитать текстовый файл с помощью популярных кодировок.
    Возвращает содержимое файла или пустую строку при ошибке.
    """
    for encoding in COMMON_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except Exception as e:
            logger.debug(f"Проблема с кодировкой {encoding} для {path}: {e}")
    logger.error(f"Не удалось прочитать файл {path} в поддерживаемых кодировках: {COMMON_ENCODINGS}")
    return ""

def extract_text_from_file(path: Path) -> str:
    """
    Универсальный парсер для извлечения текста из файлов различных форматов.
    Поддерживает: txt, html, csv, xlsx, xlsm, docx, doc, pdf.
    Возвращает текст или специальную метку при ошибке/неподдерживаемом формате.
    """
    ext = path.suffix.lower()
    if not path.exists():
        logger.error(f"Файл не найден: {path}")
        return "[Файл не найден]"

    if path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        logger.warning(f"Файл слишком большой (> {MAX_FILE_SIZE_MB} МБ): {path}")
        return "[Файл слишком большой для обработки]"

    try:
        if ext == ".txt":
            logger.info(f"Extracting text from TXT file: {path}")
            return _smart_read_text(path)

        elif ext == ".html":
            logger.info(f"Extracting text from HTML file: {path}")
            html_content = _smart_read_text(path)
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text(separator=" ")

        elif ext == ".csv":
            logger.info(f"Extracting text from CSV file: {path}")
            try:
                df = pd.read_csv(path)
                return df.to_csv(sep="\t", index=False)
            except Exception as e:
                logger.warning(f"Ошибка чтения CSV через pandas: {e}. Пробуем как текст.")
                return _smart_read_text(path)

        elif ext in [".xlsx", ".xls", ".xlsm"]:
            logger.info(f"Extracting text from Excel file: {path}")
            try:
                df = pd.read_excel(path)
                return df.to_csv(sep="\t", index=False)
            except Exception as e:
                logger.error(f"Ошибка чтения Excel через pandas: {e}")
                return "[Ошибка чтения Excel]"

        elif ext == ".docx":
            logger.info(f"Extracting text from DOCX file: {path}")
            if DOCX is not None:
                try:
                    doc = DOCX.Document(path)
                    return "\n".join([p.text for p in doc.paragraphs])
                except Exception as e:
                    logger.error(f"Ошибка чтения DOCX: {e}")
                    return "[Ошибка чтения DOCX]"
            else:
                logger.warning(f"Модуль python-docx не установлен. DOCX не поддерживается.")
                return "[Формат DOCX не поддерживается]"

        elif ext == ".doc":
            logger.info(f"Extracting text from DOC file: {path}")
            if TEXTRACT is not None:
                try:
                    return TEXTRACT.process(str(path)).decode("utf-8", errors="ignore")
                except Exception as e:
                    logger.error(f"Ошибка чтения DOC через textract: {e}")
                    return "[Ошибка чтения DOC]"
            else:
                logger.warning(f"textract не установлен. DOC не поддерживается.")
                return "[Формат DOC не поддерживается]"

        elif ext == ".pdf":
            logger.info(f"Extracting text from PDF file: {path}")
            if PDF is not None:
                try:
                    text = []
                    with open(path, "rb") as f:
                        reader = PDF.PdfReader(f)
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text.append(page_text)
                    return "\n".join(text)
                except Exception as e:
                    logger.warning(f"Ошибка чтения PDF через PyPDF2: {e}. Пробуем textract.")
            if TEXTRACT is not None:
                try:
                    return TEXTRACT.process(str(path)).decode("utf-8", errors="ignore")
                except Exception as e:
                    logger.error(f"Ошибка чтения PDF через textract: {e}")
                    return "[Ошибка чтения PDF]"
            logger.warning(f"PyPDF2 и textract не установлены. PDF не поддерживается.")
            return "[Формат PDF не поддерживается]"

        else:
            logger.warning(f"Неподдерживаемый тип файла: {path}")
            return f"[Неподдерживаемый тип файла: {ext}]"

    except Exception as e:
        logger.error(f"Критическая ошибка при извлечении текста из {path}: {e}")
        return "[Критическая ошибка при извлечении текста]"

def clean_html_from_cell(cell_value) -> str:
    """
    Очищает строку/ячейку от HTML-тегов.
    """
    if isinstance(cell_value, str):
        return BeautifulSoup(cell_value, "html.parser").get_text(separator=" ")
    return str(cell_value)
