BareNet

BareNet é uma ferramenta CLI escrita em C++ para OSINT e análise passiva de recursos web, projetada com foco em arquitetura modular, determinismo e controle total do fluxo de execução.

O projeto prioriza organização, extensibilidade e previsibilidade, evitando abordagens baseadas em scripts monolíticos.

Objetivo

Fornecer um motor de descoberta e inspeção de URLs a partir de mecanismos de busca, aplicando normalização, deduplicação e filtros configuráveis, de forma ética e controlada.

Características

Arquitetura modular baseada em interfaces (.hpp)

CLI simples e determinística

Suporte a múltiplos motores de busca (plug-and-play)

Normalização e deduplicação de URLs

Sistema de filtros encadeáveis

Código organizado em camadas claras

Build controlado via CMake

Implementação em C++ moderno (C++20)

Estrutura do Projeto
barenet/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── cli/        # Interface de linha de comando
│   ├── core/       # Orquestração e fluxo principal
│   ├── search/     # Motores de busca
│   ├── filters/    # Filtros de resultados
│   ├── net/        # Comunicação HTTP
│   ├── utils/      # Utilitários (URL, logging, etc)
│   └── config/     # Configuração
└── include/

Fluxo de Execução
CLI
 ↓
Engine
 ↓
Motores de Busca
 ↓
Normalização de URLs
 ↓
Deduplicação
 ↓
Filtros
 ↓
Resultados

Exemplo de Uso (prototipal)
barenet search "site:example.com login"
barenet engines list
barenet inspect https://example.com


Observação: Alguns comandos podem não estar implementados no estágio inicial.

Build
Requisitos

Compilador compatível com C++20 (GCC / Clang)

CMake ≥ 3.16

Ambiente Linux recomendado

Compilação
mkdir build
cd build
cmake ..
make

Princípios de Design

Separação clara de responsabilidades

Nenhuma lógica em main.cpp

Interfaces explícitas

Execução determinística

Falhas isoladas (graceful degradation)

Estado do Projeto

⚠️ Prototipo / MVP inicial

O BareNet ainda está em fase inicial de desenvolvimento.
Interfaces, estrutura e fluxo podem mudar conforme o projeto evolui.

Considerações Éticas

BareNet foi projetado para uso defensivo e passivo.

Não realiza exploração automática

Não executa ataques ativos por padrão

Qualquer inspeção ativa deve ser iniciada explicitamente pelo usuário

O uso é de inteira responsabilidade do operador.
