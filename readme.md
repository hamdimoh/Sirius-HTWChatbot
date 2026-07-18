# 🌟 Sirius – KI-gestützter RAG-Chatbot für HTW-Studierende

**Bachelorarbeit, Note 1,7** — Ein LLM-basierter Chatbot (GPT-4o), der Studierenden der HTW Berlin per natürlicher Sprache Auskunft über Verwaltungsdaten gibt und Dokumente (PDFs, URLs) durchsuchbar macht.

## Überblick

Sirius kombiniert zwei Retrieval-Strategien in einer Streamlit-Oberfläche:

1. **Natural-Language-to-SQL**: Fragen zu Dozent:innen, Terminen oder Studiengängen werden von einem LLM in SQL-Abfragen gegen eine MySQL-Datenbank übersetzt und die Antwort in natürlicher Sprache formuliert.
2. **RAG über PDFs & Webseiten**: Hochgeladene PDF-Dokumente oder eingegebene URLs werden in Chunks zerlegt, per Embeddings vektorisiert (FAISS/Chroma) und für die Beantwortung von Fragen mittels Retrieval-Augmented Generation herangezogen.

Chatverläufe werden persistent in der Datenbank gespeichert, sind pro Sitzung abrufbar und können gelöscht werden.

## Funktionen

- **Natural Language Processing**: GPT-4o interpretiert und beantwortet Anfragen in natürlicher Sprache
- **Text-to-SQL**: Dynamische Erstellung von SQL-Abfragen aus natürlichsprachlichen Eingaben
- **RAG-Pipeline**: PDF- und URL-Verarbeitung mit Text-Chunking, Embeddings und Ähnlichkeitssuche
- **Chatverlauf**: Laden, Fortsetzen und Löschen vergangener Konversationen
- **Streamlit-GUI**: Übersichtliche, sofort nutzbare Weboberfläche

## Architektur

```
Nutzer:in ─▶ Streamlit-UI ─▶ LangChain-Chain ─▶ GPT-4o
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
            MySQL (Text-to-SQL)      FAISS/Chroma (RAG über PDFs & URLs)
```

## Installation

Voraussetzung: Python 3.9+ und eine laufende MySQL-Instanz.

```bash
pip install -r requirements.txt
```

Lege eine `.env`-Datei im Projekt-Hauptverzeichnis an:

```
OPENAI_API_KEY=dein-openai-api-key

DB_USER=root
DB_PASSWORD=dein-db-passwort
DB_HOST=localhost
DB_PORT=3306
DB_NAME=HTW
```

## Nutzung

```bash
streamlit run 🏠Homepage.py
```

Die App verbindet sich beim Start automatisch mit der Datenbank (Zugangsdaten aus `.env`) und ist unter `http://localhost:8501` erreichbar.

## Tech-Stack

Python · LangChain · OpenAI GPT-4o · Streamlit · SQLAlchemy · MySQL · FAISS/Chroma · BeautifulSoup

## Lizenz

MIT — siehe [LICENSE](LICENSE).
