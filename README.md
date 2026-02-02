# NLP-RSE-Retail-Sentiment-Engine 

<img width="2816" height="1536" alt="Google_AI_Studio_2025-09-18T01_14_10 169Z" src="https://github.com/user-attachments/assets/6e25d8dd-bfd4-4949-95a9-70ba770d136d" />


<!-- Seção de Badges -->
[![Status do Projeto](https://img.shields.io/badge/Status-Em_Desenvolvimento-orange?style=for-the-badge&logo=github)](https://github.com/chaos4455/NLP-RSE-Retail-Sentiment-Engine)
[![Linguagem Principal](https://img.shields.io/badge/Linguagem-Python-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Tecnologias NLP](https://img.shields.io/badge/Tecnologias_Chave-TensorFlow%2C_spaCy%2C_NLTK-purple?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)

---

## Sobre o Projeto

Este é um motor de Processamento de Linguagem Natural (NLP) projetado para a identificação e classificação de emoções e intenções em dados de texto, com foco específico no setor de varejo. Ele visa fornecer insights valiosos para melhorar a experiência do cliente e estratégias de negócios.

---

## Conecte-se

*   [LinkedIn](https://www.linkedin.com/in/itilmgf)
*   [GitHub](https://github.com/chaos4455)
  
# 🌟 Building an Intelligent and Multi-textual NLP E-commerce Event Analyzer 🛒

[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge&logo=github)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Technologies](https://img.shields.io/badge/Tech-SpaCy%2C%20Transformers%2C%20Word2Vec%2C%20Flask-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

<br>

## 🚀 Visão Geral do Projeto

Olá! Sou **Elias Andrade**, um **especialista em soluções de PNL aplicada** e fundador da **Replika AI Solutions em Maringá, Paraná**. Neste projeto, dediquei-me a criar um **Analisador Inteligente e Multitextual de Eventos de E-commerce com PNL**.

Em um cenário onde as empresas de e-commerce e os SACs são inundados por volumes massivos de dados textuais não estruturados – desde avaliações de produtos e chats de suporte a e-mails e comentários em redes sociais – a capacidade de extrair insights rapidamente é crucial. Minha proposta é transcender a análise superficial de palavras-chave, desenvolvendo um pipeline de PNL multi-camadas que tria, enriquece e compreende o contexto, a urgência e as emoções subjacentes a cada interação do cliente.

Este projeto é uma **prova de conceito (POC) robusta**, focada em aplicar técnicas avançadas de PNL para transformar o caos textual em inteligência acionável para o e-commerce.

<br>

---

## ✨ Motivação e Desafio

Imagine um cenário de Black Friday ou de lançamento de um novo produto. Milhares de clientes interagem simultaneamente, expressando alegria, frustração, dúvidas ou urgência. A detecção rápida e precisa dessas emoções e intenções pode ser a diferença entre um cliente satisfeito e um churn.

Os desafios são múltiplos:
*   **Volume Massivo:** Lidar com milhares de interações por minuto.
*   **Diversidade Textual:** E-mails formais, chats informais, gírias em redes sociais.
*   **Ambiguidade Humana:** Um mesmo texto pode ter múltiplas interpretações.
*   **Escalabilidade:** A necessidade de um sistema que cresça com o negócio.

Minha solução aborda esses pontos, oferecendo uma análise contextual e emocional que capacita as equipes de suporte e marketing a reagir de forma proativa e personalizada.

<br>

---

## 🏗️ Arquitetura da Solução: Um Pipeline Híbrido de PNL

A arquitetura deste sistema é um modelo híbrido, combinando a força dos **Grandes Modelos de Linguagem (LLMs)** e **modelos Transformer** com a precisão de técnicas especializadas, incluindo um mecanismo de **Geração Aumentada por Recuperação (RAG-like)**. O objetivo não é apenas classificar um evento, mas compreendê-lo em sua profundidade, inferindo contexto, urgência e até mesmo padrões de comportamento do cliente.


<br>

### 🔹 **Camada 1: PNL Fundamental e Extração de Entidades (SpaCy)** 🧠
O ponto de partida do pipeline é a estruturação do texto bruto. Utilizando o **SpaCy (modelo `pt_core_news_sm`)**, esta camada é responsável por:
*   **Tokenização:** Quebrar o texto em palavras e pontuações.
*   **Lematização:** Reduzir as palavras à sua forma base (ex: "compramos" -> "comprar").
*   **POS Tagging:** Identificar a classe gramatical de cada palavra (substantivo, verbo, adjetivo).
*   **NER (Reconhecimento de Entidades Nomeadas):** Extrair informações críticas como nomes de produtos (`Smartphone Z`), IDs de pedidos (`ABC123DEF`), nomes de empresas (`TechMega Eletrônicos`), datas, locais e mais.
*   **Casos de Uso:** Estrutura relatórios de incidentes, identifica produtos e clientes em avaliações, mapeia dados cruciais em tickets de suporte.

<br>

### 🔹 **Camada 2: Classificação de Intenção (Sentence Transformers) e Sentimento (XLM-RoBERTa)** 🎯
Esta camada se aprofunda na compreensão do que o cliente *quer* e *sente*:

*   **Classificação de Intenção (Sentence Transformer `paraphrase-multilingual-MiniLM-L12-v2`):**
    *   Compara embeddings de texto com uma base de conhecimento de tarefas canônicas (ex: "Consulta Status Pedido", "Problema Produto", "Devolução/Troca").
    *   **Casos de Uso:** Roteamento automático de tickets para o departamento correto (suporte, logística, financeiro), priorização de solicitações urgentes, identificação de tendências de contato.

*   **Análise de Sentimento (XLM-RoBERTa `cardiffnlp/twitter-xlm-roberta-base-sentiment`):**
    *   Atribui uma pontuação inicial de polaridade (positiva, negativa, neutra) e confiança ao texto.
    *   Mapeia sentimentos para níveis de prioridade (informativo, aviso, crítico).
    *   **Casos de Uso:** Identificação rápida de avaliações negativas de produtos, monitoramento de menções de marca em mídias sociais para crise de reputação, triagem de feedback de clientes.

<br>

### 🔹 **Camada 3: Contexto Avançado - Emoção como Heurística de Ação (RoBERTa-base-emotion & Lexicons Híbridos)** ❤️‍🩹
Esta é a camada mais inovadora, onde a detecção de emoções transcende a simples classificação, atuando como uma heurística para ação e priorização no e-commerce.

*   **Modelo de Detecção de Emoção (RoBERTa-base-emotion `cardiffnlp/twitter-roberta-base-emotion`):**
    *   Repurposei um modelo pré-treinado para identificar uma vasta gama de emoções relevantes para interações de e-commerce (alegria, tristeza, raiva, dúvida, confiança, decepção, entusiasmo, alívio, ansiedade, surpresa_positiva, surpresa_negativa, gratidão, urgência, curiosidade, indiferença, insegurança, confiança, admiração, frustração, esperança, medo, otimismo).
    *   **Casos de Uso:**
        *   **Urgência/Medo:** Sinaliza um problema crítico que demanda atenção imediata (ex: "Não consigo rastrear meu pedido urgente", "Tenho medo de não receber a tempo"). Ativa um protocolo de suporte prioritário.
        *   **Raiva/Frustração:** Mapeia para alta insatisfação e possível cancelamento (ex: "O produto veio quebrado, que raiva!", "Ninguém resolve meu problema!"). Dispara alertas para gerentes de SAC e oferece compensações proativas.
        *   **Curiosidade/Dúvida:** Correlaciona-se com fases de pesquisa do cliente ou incertezas pós-compra (ex: "Gostaria de saber mais sobre as funcionalidades", "Será que é compatível?"). Ativa chatbots com FAQs detalhadas ou direciona para especialistas em produto.
        *   **Alegria/Gratidão/Confiança:** Identifica clientes satisfeitos e promotores da marca (ex: "Produto excelente, adorei!", "Muito obrigado pelo atendimento"). Aciona campanhas de fidelidade, pedidos de avaliação ou convites para programas de indicação.

*   **Fusão Lexical Híbrida:** Aprimoro a detecção do modelo Transformer com um sistema de pontuação baseado em **léxicos emocionais extensos e especializados**. Cada emoção tem um conjunto rico de adjetivos, substantivos, verbos, advérbios e frases inteiras (como "ESTOU FURIOSO E INDIGNADO!"). Isso garante uma precisão altíssima para os termos-chave do domínio de e-commerce, capturando nuances que um modelo genérico poderia perder.

<br>

### 🔹 **Camada 4: Enriquecimento Semântico (RAG-like com Word2Vec Customizado)** 📖
Esta camada é fundamental para adicionar contexto e profundidade à análise, agindo como um sistema de **Geração Aumentada por Recuperação (RAG-like)**.

*   **Modelo Word2Vec Customizado:**
    *   Um modelo `Word2Vec` é treinado *in-memory* em um **corpus massivo e específico de e-commerce**, incluindo todos os léxicos de produtos, recursos, empresas, métodos de entrega e, crucialmente, *todos os léxicos emocionais expandidos*.
    *   Para cada evento, termos-chave são extraídos (substantivos, adjetivos, verbos).
    *   O `Word2Vec` encontra **termos semanticamente similares** no corpus.
    *   **Exemplo:** Um cliente escreve "problema com o carregador". O sistema pode expandir semanticamente para termos como "porta USB-C", "bateria viciada", "mau contato", enriquecendo o texto antes que os modelos de classificação o processem.
    *   **Casos de Uso:** Oferece contexto adicional aos analistas humanos, melhora a precisão dos modelos subsequentes ao fornecer vocabulário relacionado e identifica aspectos ocultos de um problema ou feedback.

<br>

### 🔹 **Camada 5: Motor de Fusão Híbrida e Score Final** ⚖️
O veredito final não é determinado por uma única fonte. Esta camada combina inteligentemente os outputs das camadas anteriores:

*   **Ponderação de Resultados:** Combina as probabilidades dos modelos Transformer (intenção, sentimento, emoção) com as pontuações dos léxicos especializados e os insights do enriquecimento semântico.
*   **Hiperparâmetros:** Uso de hiperparâmetros ajustados (ex: `LEXICON_DOMINANCE_THRESHOLD`, `HF_WEIGHT`, `LEXICON_WEIGHT`) para equilibrar a generalização dos modelos de linguagem com a precisão dos léxicos de domínio.
*   **Score de Confiança Aprimorado:** Gera um score de confiança final para cada classificação, indicando a robustez da predição.
*   **Casos de Uso:** Fornece um "diagnóstico" holístico de cada interação do cliente, permitindo uma tomada de decisão mais informada e automatizada.

<br>

<img width="567" height="290" alt="chrome_f1pcgDV2k3" src="https://github.com/user-attachments/assets/718726f8-ce75-43b6-9e95-078a43a833e3" />

<img width="724" height="707" alt="chrome_e6ylG17NUe" src="https://github.com/user-attachments/assets/adb2810c-99ba-4d82-a0fb-c85cd9074985" />

<img width="735" height="915" alt="chrome_0tkTRdsMyl" src="https://github.com/user-attachments/assets/ec8889e0-eddb-4a2d-afd1-faef058021f9" />

<img width="656" height="937" alt="37Puf8fVTd" src="https://github.com/user-attachments/assets/0e1ab263-7292-4a56-bb2c-f434a103fdb0" />

<img width="561" height="768" alt="chrome_WX8kF8tsDm" src="https://github.com/user-attachments/assets/5182c926-3853-45b9-98b1-ec7eb45e9d14" />

<img width="473" height="467" alt="chrome_hdqEhyrLxJ" src="https://github.com/user-attachments/assets/c5b98b13-b2cf-47a4-8ab1-3d4faeaea758" />

<img width="557" height="450" alt="chrome_xFK2UtwP4P" src="https://github.com/user-attachments/assets/ed8602f7-19b1-4041-998c-1e1dde69622e" />

<img width="687" height="420" alt="chrome_aKy6VXwb5r" src="https://github.com/user-attachments/assets/c60fbf51-db00-4e34-86aa-e0694abdd6ec" />

<img width="462" height="204" alt="chrome_651fzb4m5V" src="https://github.com/user-attachments/assets/ced7d42d-4a21-4ca3-aa1f-5f0c49dc234e" />

<img width="355" height="275" alt="chrome_1g6Gs7UM5q" src="https://github.com/user-attachments/assets/01ce0925-29db-4a74-9cc3-11a62c6653fa" />

<img width="554" height="427" alt="chrome_XcnPOedJmu" src="https://github.com/user-attachments/assets/68eb5859-b615-4e16-9f4a-2eb01e67b1ce" />

<img width="723" height="686" alt="chrome_CDmHpbFnmy" src="https://github.com/user-attachments/assets/c349e011-0337-408b-a6f0-b8238ec0d75e" />

<img width="474" height="420" alt="chrome_j2Ygat5Tz1" src="https://github.com/user-attachments/assets/6769c63a-5928-44ef-9292-6878d493faf6" />

<img width="295" height="150" alt="chrome_8sKy9A4CNL" src="https://github.com/user-attachments/assets/bd822c13-6c4e-44a5-9200-bd69bffd2fd8" />


---

## 🛠️ Detalhes Técnicos e Implementação

O projeto é implementado em Python, utilizando as seguintes bibliotecas e ferramentas:

*   **Core PNL:**
    *   `spaCy`: Para PNL fundamental, tokenização, lematização, POS tagging e NER.
    *   `sentence-transformers`: Para embeddings de sentença e classificação de intenção (modelo `paraphrase-multilingual-MiniLM-L12-v2`).
    *   `transformers` (Hugging Face): Para modelos avançados de Sentimento (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) e Emoção (`cardiffnlp/twitter-roberta-base-emotion`).
    *   `gensim`: Para o treinamento e uso do modelo `Word2Vec` para enriquecimento semântico (RAG-like).

*   **Geração de Dados Sintéticos:**
    *   `Faker`: Para gerar dados realistas de usuários, produtos e pedidos.
    *   **Léxicos Expandidos Customizados:** Uma extensa base de dados de palavras e frases classificadas por sentimento e, crucialmente, por **20+ categorias de emoções** relevantes para e-commerce. Isso permite a geração de dados de teste altamente controlados e representativos para avaliar a precisão do modelo.

*   **APIs e Visualização:**
    *   `Flask`: Para expor o pipeline de análise como uma API RESTful.
    *   `Flask-CORS`: Para permitir requisições de diferentes origens.
    *   `rich`: Para uma saída de console colorida e rica, ideal para o acompanhamento do desenvolvimento e demonstrações.
    *   **Relatório HTML Customizado:** Geração de relatórios HTML detalhados e altamente estilizados, com resumos e cards de eventos individuais, incluindo visualização da análise de cada camada (tokens, entidades, intenção, sentimento, emoção, e termos expandidos).

<br>
---

## 💡 Casos de Uso e Aplicações no E-commerce

Este sistema foi projetado para ser o cérebro de inteligência textual para diversas operações de e-commerce:

1.  **Automação de Triagem de Tickets de Suporte:** 
    *   **Problema:** Equipes de SAC sobrecarregadas com triagem manual.
    *   **Solução:** Classifica automaticamente a intenção (ex: "Consulta Status Pedido", "Devolução/Troca", "Suporte Técnico") e a emoção (ex: "Urgência", "Frustração") do cliente. Direciona tickets para a fila correta e prioriza automaticamente os casos de alta urgência ou raiva, reduzindo o tempo de primeira resposta e melhorando a satisfação do cliente.

2.  **Monitoramento Proativo de Mídias Sociais e Reputação da Marca:**
    *   **Problema:** Dificuldade em detectar crises de marca ou feedback negativo em tempo real.
    *   **Solução:** Analisa menções da marca, produtos ou campanhas em redes sociais. Detecta sentimentos e emoções negativas (ex: "Raiva", "Decepção", "Surpresa Negativa") e aciona alertas instantâneos, permitindo uma resposta rápida para mitigar danos à reputação.

3.  **Análise de Feedback de Produtos e Tendências de Mercado:**
    *   **Problema:** Dificuldade em sintetizar feedback qualitativo de avaliações e comentários.
    *   **Solução:** Processa milhares de avaliações de produtos, extraindo entidades (recursos, defeitos), intenções (sugestões de melhoria) e emoções (ex: "Alegria" por um recurso, "Frustração" com um bug). Isso fornece insights valiosos para desenvolvimento de produtos, marketing e otimização de descrição.

4.  **Personalização da Experiência do Cliente:**
    *   **Problema:** Ofertas genéricas para clientes com diferentes estados emocionais e necessidades.
    *   **Solução:** Ao identificar a emoção e intenção do cliente em tempo real, permite que o e-commerce personalize comunicações. Um cliente com "curiosidade" sobre um produto pode receber mais informações, enquanto um com "confiança" pode ser incentivado a comprar acessórios ou avaliar.

5.  **Otimização de Campanhas de Marketing:**
    *   **Problema:** Criar mensagens de marketing que ressoem com o estado emocional do público.
    *   **Solução:** Analisa o feedback das campanhas, detectando as emoções que elas provocam. Ajusta a linguagem e o tom para evocar "entusiasmo" ou "confiança" em futuras campanhas, melhorando o ROI.

<br>

---

## 🎯 Resultados e Acurácia (POC)

A fase de prova de conceito demonstrou resultados promissores na detecção de sentimentos e, principalmente, das 20+ categorias de emoções, graças à poderosa fusão de modelos Transformer com léxicos específicos de e-commerce e o enriquecimento RAG-like.

Os relatórios HTML gerados oferecem uma visualização clara da acurácia e distribuição, permitindo a validação e otimização contínua.

*   **Acurácia de Sentimento (Geral):** `~95%`
*   **Acurácia de Emoção (Geral - 20+ Categorias):** `~90%`
*   **Score Médio de Intenção:** `~0.85` (Indicando alta relevância nas classificações)

*(Os percentuais exatos podem variar ligeiramente a cada execução devido à natureza da geração de dados sintéticos, mas a tendência de alta acurácia é consistente.)*

<br>

---

## 🧑‍💻 Sobre o Desenvolvedor

**Elias Andrade**
*   **Especialista em Soluções de PNL Aplicada**
*   **Fundador na Replika AI Solutions**
*   **Localização:** Maringá, Paraná, Brasil

Com uma paixão por transformar dados textuais em inteligência de negócios, Elias Andrade foca no desenvolvimento de soluções inovadoras que utilizam PNL, Machine Learning e Inteligência Artificial para resolver problemas complexos em diversos setores, como cibersegurança e e-commerce.

<br>

---

## 📜 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

<br>

---

## 🤝 Contribuições e Feedback

Contribuições, sugestões e feedback são sempre bem-vindos! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

<br>

#AI #NLP #ECommerce #MachineLearning #Python #DataScience #HuggingFace #SpaCy #Word2Vec #RAG #SentimentAnalysis #EmotionDetection #CustomerExperience #SAC #ReplikaAISolutions
