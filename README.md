# BareNet

> **BareNet** é um navegador / motor de busca em modo texto (CLI/TUI) orientado à **análise passiva de sinais de segurança**, projetado para identificar e navegar apenas por páginas que apresentem **indícios técnicos de fragilidade** (misconfigurações, exposições acidentais, ausência de boas práticas), sem realizar qualquer forma de exploração ativa.

Este documento descreve **o objetivo do projeto, suas características principais e a arquitetura adotada**. O BareNet encontra-se em **estágio de prototipagem**, com foco em clareza arquitetural, modularidade e segurança por padrão.

---

## Objetivo do Projeto

O BareNet tem como objetivo permitir que pesquisadores e estudantes de segurança:

* Naveguem na web **a partir do terminal**
* Realizem buscas por termos comuns (ex.: "login", "admin", "páscoa")
* Visualizem **apenas páginas que apresentem algum sinal técnico relevante**
* Inspecionem conteúdo HTML de forma **segura, passiva e sanitizada**

O projeto **não executa ataques**, não realiza exploração ativa e não envia payloads maliciosos. Ele funciona como um **filtro de visibilidade técnica** sobre a web.

---

## Princípios Fundamentais

* **Passividade absoluta**: nenhuma ação invasiva é executada
* **Safe by default**: JS nunca é executado; submissões são bloqueadas
* **Separação de responsabilidades**: UI, engine, detecção e renderização isolados
* **Extensibilidade**: novos sinais, fontes e renderizadores podem ser adicionados como plugins
* **Persistência local**: dados armazenados localmente para análise e auditoria

---

## O que o BareNet é (e não é)

### É

* Um navegador CLI/TUI
* Um motor de busca orientado a sinais
* Uma ferramenta de estudo e observação
* Um indexador passivo de páginas públicas

### Não é

* Um scanner ativo
* Um fuzzer
* Uma ferramenta de exploração
* Um substituto de navegador tradicional

---

## Arquitetura Geral

O fluxo principal do BareNet segue o modelo:

```
Usuário → UI (Textual) → Engine
                      → Searcher
                      → Signals
                      → Storage
                      → Renderer (on-demand)
```

Cada camada possui responsabilidades bem definidas e não acessa camadas inferiores diretamente.

---

## Estrutura de Diretórios

```
barenet/
├── __main__.py           # bootstrap + inicialização
├── bootstrap.py          # verificação de dependências de runtime
├── cli.py                # entrada CLI
├── engine.py             # orquestração do fluxo
├── httpclient.py         # cliente HTTP passivo
├── indexer.py            # descoberta de páginas
├── parser/               # parsers de HTML e TLS
├── signals/              # detectores de sinais (plug-and-play)
├── signals_registry.py   # registro automático de detectores
├── db/                   # persistência (SQLite + FTS5)
├── cache.py              # cache local de conteúdo
├── searcher.py           # consultas, ranking e filtros
├── inspect.py            # inspeção HTML sanitizada
├── renderers/            # renderização externa (ex.: w3m)
├── plugins/              # fontes de descoberta
├── ui/                   # interface TUI
├── exporters.py          # exportação de dados
├── tools/                # utilitários de desenvolvimento
├── docs/                 # documentação
└── README.md
```

---

## Camadas Principais

### UI (Textual)

* Interface interativa no terminal
* Campo de busca, lista de resultados e painel lateral
* Navegação por teclado

### Engine

* Coordena o fluxo de busca, análise e apresentação
* Aplica filtros selecionados pelo usuário

### Signals

* Detectores independentes
* Cada arquivo representa um sinal técnico
* Exemplo: HTTP sem TLS, headers ausentes, mensagens de erro

### Storage

* Banco de dados local SQLite
* Full-text search (FTS5)
* Persistência para histórico e auditoria

### Renderer

* Renderização HTML sob demanda
* `w3m` utilizado apenas quando o usuário solicita visualização

---

## Segurança por Design

* Nenhum JavaScript é executado
* HTML é sanitizado antes da renderização
* Limites rígidos de tamanho e tempo
* Respeito a políticas de uso local

---

## Estado Atual do Projeto

* Estrutura arquitetural definida
* Interfaces em definição
* UI e engine em fase inicial
* Detectores básicos planejados

Este README serve como **guia de arquitetura** e **referência conceitual** durante a fase de prototipagem.

---

## Próximos Passos

* Implementar interfaces base (Signal, Storage, Renderer)
* Criar UI mínima funcional
* Adicionar primeiros detectores passivos
* Documentar contratos internos

---

## Licença

A definir.
