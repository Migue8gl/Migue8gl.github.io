---
layout: default
title: Miguel García López CV
description: AI Engineer y Data Scientist especializado en agentes de IA, NLP y sistemas de búsqueda semántica
---

# Miguel García López
**AI Engineer & Data Scientist**

- Email: [migue8gl@gmail.com](mailto:migue8gl@gmail.com)
- Location: Granada, España
- LinkedIn: [miguel-garcía-lópez-b5bb69273](https://linkedin.com/in/miguel-garcía-lópez-b5bb69273)
- GitHub: [Migue8gl](https://github.com/Migue8gl)

---

## Perfil

AI Engineer y Data Scientist. Actualmente desarrollo plataformas de agentes de IA integradas con los sistemas de la empresa, con control de acceso por permisos, monitorización en producción y una capa de skills reutilizable. Antes, varios años construyendo modelos de clasificación en producción (F1 hasta 0.95), pipelines ETL a escala y sistemas de búsqueda semántica sobre +500K documentos. Especializado en NLP (fine-tuning de transformers, RAG) y modelado predictivo con ML clásico (Scikit-Learn, XGBoost), Deep Learning (PyTorch) y análisis de datos con Pandas. Enfoque en análisis exploratorio, feature engineering, validación estadística e impacto de negocio.

---

## Experiencia

### **AI Engineer**
*Omnicom* · Granada, España (remoto)  
*May 2026 – Actualidad*

- Desarrollo de una plataforma interna de agentes integrada con los sistemas de la empresa, con filtrado de acceso por permisos para que cada agente opere solo dentro del ámbito autorizado del usuario.
- Servicios de monitorización del comportamiento y el rendimiento de los agentes en producción.
- Diseño de una capa de skills reutilizable que permite ampliar los agentes con nuevas capacidades.
- Trabajo con otros departamentos para identificar sus procesos y automatizarlos con agentes de IA.

### **Data Scientist & AI Developer**
*GPU Solutions* · Granada, España  
*Feb 2025 – May 2026*

- Entrenamiento de un clasificador de prompt injection sobre dataset curado a partir de múltiples fuentes públicas y aumentado con LLM: +90% en accuracy y F1, con +10% de mejora vs. LLM con prompt especializado, inferencia en CPU en milisegundos. Demo comparativa con Streamlit.
- Fine-tuning de Gemma con QLoRA sobre documentación técnica, evaluando con ROUGE, BLEU y BERTScore. +18% ROUGE-L y +22% BERTScore F1 vs. modelo base, validado con evaluación humana en QA.
- Sistemas RAG escalables sobre +500K documentos en Qdrant, desplegados en AKS (Azure). Agentes conversacionales con LangGraph integrados con WhatsApp, evaluados con Phoenix (observabilidad) y LLM-as-judge, con score personalizado basado en flujo de nodos, ejecución de herramientas y calidad de respuestas (85% coherencia).
- Pipeline de scraping a gran escala (multiprocessing, Beautiful Soup) para la extracción diaria de datos públicos (HTML, PDF), que funciona como ETL del sistema de recuperación de información: aceleración ~10x vs. enfoque secuencial.
- Visión por computadora clásica (OpenCV) para eliminación de líneas de recorte en imágenes destinadas a software de visualización 3D.
- Instalación y configuración de SLURM en un clúster de servidores GPU para cargas de trabajo internas y experimentación. Despliegue con Docker y Kubernetes; interfaces con React (frontend) y FastAPI (backend).

### **Data Scientist**
*EDUCA EDTECH Group* · Granada, España  
*May 2024 – Feb 2025*

- Clasificación jerárquica de texto (+100 categorías, fuerte desbalance de clases) con DistilBERT: F1 0.95 (nivel alto) / 0.80 (granular). Comparación con FastText: métricas equivalentes con mayor eficiencia en inferencia.
- Modelo XGBoost de predicción de conversión de cupones, usado por el equipo comercial para optimizar campañas y flujo de conversión de la plataforma.
- Búsqueda semántica híbrida (embeddings + BM25) en Pinecone sobre +1.000 categorías. Reducción de dimensionalidad y clustering jerárquico por densidad de los embeddings para evaluar su separabilidad, con LLM para consolidar taxonomías automáticamente.
- Pipelines ETL con Airflow (varios GB/día): limpieza, preprocesado y feature engineering. Despliegue de modelos con FastAPI. Validación estadística rigurosa: Wilcoxon, intervalos de confianza, validación cruzada estratificada.
- Refactorización de un proyecto basado en Elasticsearch, contenerizado con Docker y desplegado en Google Cloud.
- Diseño de esquemas de base de datos normalizados para integridad y eficiencia en consultas analíticas.
- Colaboración con equipos de ingeniería y producto de empresas líderes en IA en el diseño de soluciones escalables para producción.

### **Desarrollador de Software Junior**
*EDUCA EDTECH Group* · Granada, España · Media jornada  
*Mar 2023 – May 2024*

- Servicios backend con PHP (Symfony): diseño de endpoints RESTful (OpenAPI/Swagger) para la consulta y gestión de datos.
- Paneles de administración con EasyAdmin y gestión de bases de datos relacionales mediante ORM. Frontend con Twig y Bootstrap.
- Trabajo en equipos ágiles (Jira, Git) siguiendo el patrón MVC para mantener el código modular y mantenible.

### **Prácticas de Desarrollo de Software**
*EDUCA EDTECH Group* · Granada, España  
*Ene 2023 – Mar 2023*

- Desarrollo web con PHP y ORM.

---

## Educación

### **Universidad de Granada**
*Máster en Ciencia de Datos* · Granada, España  
*2024 – 2025*

### **Universidad de Granada**
*Grado en Ingeniería Informática* · Granada, España  
*2019 – 2024*

---

## Habilidades Técnicas

**Modelado & ML:** Scikit-Learn, XGBoost, PyTorch, HuggingFace Transformers, clasificación, regresión, clustering, feature engineering, validación cruzada, tests estadísticos

**NLP, LLMs & Agentes:** Agentes de IA (control de acceso, monitorización, skills), LangGraph, Fine-tuning (QLoRA/PEFT), RAG, embeddings, búsqueda semántica, evaluación ROUGE/BLEU/BERTScore, LLM-as-judge

**Datos & Infra:** Pandas, Polars, NumPy, SQL, Matplotlib, Seaborn, Airflow, Beautiful Soup, Docker, Kubernetes (AKS), SLURM, GCP, Azure, FastAPI, Pinecone, Qdrant, Elasticsearch

**Lenguajes:** Python, SQL, C++, PHP, JavaScript

---

## Proyectos

### **Metaheurísticas para Selección de Características**
*Trabajo de Fin de Grado, 2024*

Comparación de Algoritmos Genéticos, PSO y evolutivos novedosos sobre +10 datasets de alta dimensión. Reducciones del 80% en features con pérdida de accuracy <5%, validado con tests no paramétricos.

---

## Información Adicional

**Idiomas:** Español (nativo), Inglés (B2 – competencia profesional)

**Certificaciones:** LangGraph · DeepLearning.AI — LLMOps · NVIDIA — CUDA with Python